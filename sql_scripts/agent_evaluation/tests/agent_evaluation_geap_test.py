# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the GEAP BigQuery Python UDF module."""

import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import threading
import types
from typing import Any, Callable, Optional
import unittest
from unittest import mock


class FakeResponse:
  """Fake requests.Response for unit tests."""

  def __init__(self, status_code=200, json_data=None, text="", headers=None):
    self.status_code = status_code
    self._json_data = json_data if json_data is not None else {}
    self.text = text or json.dumps(self._json_data)
    self.headers = headers or {}

  def json(self):
    if isinstance(self._json_data, Exception):
      raise self._json_data
    return self._json_data


class FakeSession:
  """Fake requests.Session for unit tests."""

  def __init__(self):
    self.headers = {}
    self.mounted_adapters = {}

  def mount(self, prefix, adapter):
    self.mounted_adapters[prefix] = adapter

  def post(self, url, headers=None, json=None, timeout=None):
    return FakeResponse()


class FakeHTTPAdapter:
  """Fake requests.adapters.HTTPAdapter for unit tests."""

  def __init__(self, pool_connections=10, pool_maxsize=10):
    self.pool_connections = pool_connections
    self.pool_maxsize = pool_maxsize


def _build_fake_modules() -> dict[str, Any]:
  """Constructs hermetic fake modules for google.auth and requests."""
  fake_google: Any = types.ModuleType("google")
  google_auth_error = type("GoogleAuthError", (Exception,), {})
  fake_auth_exceptions: Any = types.ModuleType("google.auth.exceptions")
  fake_auth_exceptions.GoogleAuthError = google_auth_error

  fake_auth_requests: Any = types.ModuleType("google.auth.transport.requests")
  fake_auth_requests.Request = type("FakeAuthRequest", (), {})

  fake_auth_transport: Any = types.ModuleType("google.auth.transport")
  fake_auth_transport.requests = fake_auth_requests

  fake_auth: Any = types.ModuleType("google.auth")
  fake_auth.exceptions = fake_auth_exceptions
  fake_auth.transport = fake_auth_transport
  fake_auth.default = lambda scopes=None: (None, None)
  fake_google.auth = fake_auth

  req_exception = type("RequestException", (Exception,), {})
  invalid_url = type("InvalidURL", (req_exception,), {})
  invalid_header = type("InvalidHeader", (req_exception,), {})
  invalid_json = type("InvalidJSONError", (req_exception,), {})
  timeout = type("Timeout", (req_exception,), {})
  connection_error = type("ConnectionError", (req_exception,), {})
  http_error = type("HTTPError", (req_exception,), {})

  fake_req_exceptions: Any = types.ModuleType("requests.exceptions")
  fake_req_exceptions.RequestException = req_exception
  fake_req_exceptions.InvalidURL = invalid_url
  fake_req_exceptions.InvalidHeader = invalid_header
  fake_req_exceptions.InvalidJSONError = invalid_json
  fake_req_exceptions.Timeout = timeout
  fake_req_exceptions.ConnectionError = connection_error
  fake_req_exceptions.HTTPError = http_error

  fake_adapters: Any = types.ModuleType("requests.adapters")
  fake_adapters.HTTPAdapter = FakeHTTPAdapter

  fake_requests: Any = types.ModuleType("requests")
  fake_requests.exceptions = fake_req_exceptions
  fake_requests.adapters = fake_adapters
  fake_requests.Response = FakeResponse
  fake_requests.Session = FakeSession
  fake_requests.post = lambda *args, **kwargs: FakeResponse()

  return {
      "google": fake_google,
      "google.auth": fake_auth,
      "google.auth.exceptions": fake_auth_exceptions,
      "google.auth.transport": fake_auth_transport,
      "google.auth.transport.requests": fake_auth_requests,
      "requests": fake_requests,
      "requests.exceptions": fake_req_exceptions,
      "requests.adapters": fake_adapters,
  }


def _find_geap_dir() -> pathlib.Path:
  """Finds the directory containing the geap_evaluate_udf module.

  Returns:
    pathlib.Path of the directory containing geap_evaluate_udf.py.

  Raises:
    FileNotFoundError: If the geap directory cannot be found.
  """
  for base in (
      pathlib.Path(__file__).parent.parent,
      pathlib.Path(__file__).resolve().parent.parent,
  ):
    candidate = base / "geap"
    if (candidate / "geap_evaluate_udf.py").is_file():
      return candidate
  raise FileNotFoundError("Could not locate geap directory.")


def _load_module_from_path(module_name: str, file_path: pathlib.Path) -> Any:
  """Dynamically loads a Python module from disk by file path."""
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {file_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _load_udf_module() -> Any:
  """Dynamically loads the geap_evaluate_udf module from disk."""
  return _load_module_from_path(
      "geap_evaluate_udf", _find_geap_dir() / "geap_evaluate_udf.py"
  )


def _load_generator_module() -> Any:
  """Dynamically loads the generate_udf_sql module from disk."""
  return _load_module_from_path(
      "generate_udf_sql", _find_geap_dir() / "tools" / "generate_udf_sql.py"
  )


def _extract_udf_python_bodies(sql_text: str) -> list[str]:
  """Extracts all Python UDF definition bodies from raw string SQL options."""
  bodies = re.findall(r"AS\s+r'''\n(.*?)\n''';", sql_text, flags=re.DOTALL)
  if not bodies:
    raise ValueError("Could not extract Python UDF body from SQL text.")
  return bodies


def _load_fixture(filename: str) -> dict[str, Any]:
  """Loads a JSON test fixture from testdata/geap.

  Args:
    filename: Base filename of the fixture file (e.g. single_turn_trace.json).

  Returns:
    Deserialized JSON content dictionary.

  Raises:
    FileNotFoundError: If the fixture file cannot be located.
  """
  for base in (
      pathlib.Path(__file__).parent,
      pathlib.Path(__file__).resolve().parent,
  ):
    candidate = base / "testdata" / "geap" / filename
    if candidate.is_file():
      with open(candidate, "r", encoding="utf-8") as f:
        return json.load(f)
  raise FileNotFoundError(f"Could not locate fixture {filename}")


class AgentEvaluationGeapTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    fake_modules = _build_fake_modules()
    patcher = mock.patch.dict(sys.modules, fake_modules)
    patcher.start()
    cls.addClassCleanup(patcher.stop)
    cls.udf = _load_udf_module()

  def setUp(self):
    super().setUp()
    self.udf._CACHED_CREDENTIALS = None
    self.udf._HTTP_SESSION = None
    # Simulate an installed UDF; install.sh fills this placeholder.
    old_project = self.udf._DEFAULT_PROJECT_ID
    self.udf._DEFAULT_PROJECT_ID = "test-proj"
    self.addCleanup(setattr, self.udf, "_DEFAULT_PROJECT_ID", old_project)

  def _run_retry_test(
      self,
      responses: list[Any],
      random_fn: Callable[[float, float], float] = lambda a, b: 0.0,
  ) -> tuple[Any, int, Optional[str], int, float, list[float]]:
    """Executes evaluate_with_retries across an injected sequence of responses.

    Args:
      responses: Sequence of response objects or Exception instances to return
        or raise from fake_send.
      random_fn: Function providing jitter values.

    Returns:
      A 6-tuple of (response, http_code, error, attempts, fake_time,
      slept_durations).
    """
    attempts = 0
    fake_time = 0.0
    slept_durations = []

    def fake_send():
      nonlocal attempts
      val = responses[min(attempts, len(responses) - 1)]
      attempts += 1
      if isinstance(val, Exception):
        raise val
      return val

    def fake_sleep(duration):
      nonlocal fake_time
      fake_time += duration
      slept_durations.append(duration)

    def fake_clock():
      return fake_time

    resp, http_code, error = self.udf.evaluate_with_retries(
        send_fn=fake_send,
        sleep_fn=fake_sleep,
        clock_fn=fake_clock,
        random_fn=random_fn,
    )
    return resp, http_code, error, attempts, fake_time, slept_durations

  def _evaluate_with_mocks(
      self,
      trace_str: str,
      creds_return: Any = None,
      retry_return: Any = None,
      managed_metric: Optional[str] = "tool_use_quality_v1",
      prompt_template: Optional[str] = None,
  ) -> dict[str, Any]:
    """Executes geap_evaluate with hermetic credentials and transport mocking.

    Args:
      trace_str: Serialized JSON conversation trace string.
      creds_return: Return value to inject for udf.get_credentials.
      retry_return: Return value to inject for udf.evaluate_with_retries.
      managed_metric: Optional GEAP managed metric name.
      prompt_template: Optional custom prompt template string.

    Returns:
      Dictionary returned by geap_evaluate.
    """
    with mock.patch.object(
        self.udf, "get_credentials", return_value=creds_return
    ), mock.patch.object(
        self.udf, "evaluate_with_retries", return_value=retry_return
    ):
      return self.udf.geap_evaluate(
          trace_str=trace_str,
          managed_metric=managed_metric,
          prompt_template=prompt_template,
      )

  def test_fixtures_load(self):
    single_turn = _load_fixture("single_turn_trace.json")
    self.assertIn("turns", single_turn)
    self.assertEqual(len(single_turn["turns"]), 1)

  def test_module_loaded(self):
    self.assertTrue(hasattr(self.udf, "VALID_MANAGED_METRICS"))
    self.assertIn("tool_use_quality_v1", self.udf.VALID_MANAGED_METRICS)
    self.assertIn("tool_use_quality_v2", self.udf.VALID_MANAGED_METRICS)
    self.assertIn("final_response_quality_v2", self.udf.VALID_MANAGED_METRICS)
    self.assertIn("hallucination_v2", self.udf.VALID_MANAGED_METRICS)

  def test_oneof_managed_metric_or_prompt_template_provided(self):
    single_turn = json.dumps(_load_fixture("single_turn_trace.json"))
    with self.assertRaisesRegex(
        ValueError,
        "Exactly one of managed_metric or prompt_template must be provided.",
    ):
      self.udf.validate_inputs(
          trace_str=single_turn,
          managed_metric=None,
          prompt_template=None,
      )

    with self.assertRaisesRegex(
        ValueError,
        "Exactly one of managed_metric or prompt_template must be provided.",
    ):
      self.udf.validate_inputs(
          trace_str=single_turn,
          managed_metric="tool_use_quality_v1",
          prompt_template="Is this polite: {agent_data}?",
      )

  def test_unknown_managed_metric_raises(self):
    single_turn = json.dumps(_load_fixture("single_turn_trace.json"))
    with self.assertRaisesRegex(
        ValueError, "Managed metric unknown_metric is not supported."
    ):
      self.udf.validate_inputs(
          trace_str=single_turn,
          managed_metric="unknown_metric",
          prompt_template=None,
      )

  def test_prompt_without_placeholder_raises(self):
    single_turn = json.dumps(_load_fixture("single_turn_trace.json"))
    with self.assertRaisesRegex(
        ValueError,
        "Custom prompt_template must contain at least one placeholder",
    ):
      self.udf.validate_inputs(
          trace_str=single_turn,
          managed_metric=None,
          prompt_template="prompt without placeholder",
      )

  def test_empty_turns_in_trace_raises(self):
    with self.assertRaisesRegex(
        ValueError, "Trace must contain at least one turn."
    ):
      self.udf.validate_inputs(
          trace_str='{"turns": []}',
          managed_metric="multi_turn_task_success_v1",
          prompt_template=None,
      )

  def test_non_object_json_trace_raises_value_error(self):
    for bad_trace in ('"null"', "null", "[]", "123", '"plain_string"'):
      with self.subTest(bad_trace=bad_trace):
        with self.assertRaisesRegex(
            ValueError, "Trace must be a valid JSON object."
        ):
          self.udf.validate_inputs(
              trace_str=bad_trace,
              managed_metric="tool_use_quality_v1",
              prompt_template=None,
          )

  def test_non_list_turns_in_trace_raises_value_error(self):
    for bad_turns in ('{"turns": 5}', '{"turns": "a"}', '{"turns": {"0": 1}}'):
      with self.subTest(bad_turns=bad_turns):
        with self.assertRaisesRegex(
            ValueError, "Trace must contain at least one turn."
        ):
          self.udf.validate_inputs(
              trace_str=bad_turns,
              managed_metric="tool_use_quality_v1",
              prompt_template=None,
          )

  def test_valid_inputs_pass_validation(self):
    single_turn = json.dumps(_load_fixture("single_turn_trace.json"))
    multi_turn = json.dumps(_load_fixture("multi_turn_trace.json"))

    # Valid single-turn managed metric
    self.assertEqual(
        self.udf.validate_inputs(
            trace_str=single_turn,
            managed_metric="tool_use_quality_v1",
            prompt_template=None,
        ),
        json.loads(single_turn),
    )
    # Valid multi-turn managed metric
    self.assertEqual(
        self.udf.validate_inputs(
            trace_str=multi_turn,
            managed_metric="multi_turn_task_success_v1",
            prompt_template=None,
        ),
        json.loads(multi_turn),
    )
    # Valid custom prompt template
    self.assertEqual(
        self.udf.validate_inputs(
            trace_str=single_turn,
            managed_metric=None,
            prompt_template="Check {agent_data}",
        ),
        json.loads(single_turn),
    )
    # Valid None trace (only validates metrics)
    self.assertIsNone(
        self.udf.validate_inputs(
            trace_str=None,
            managed_metric="tool_use_quality_v1",
            prompt_template=None,
        )
    )

  def test_single_turn_managed_metric_rejects_multi_turn_trace(self):
    multi_turn = _load_fixture("multi_turn_trace.json")
    with self.assertRaises(ValueError):
      self.udf.validate_inputs(
          trace_str=json.dumps(multi_turn),
          managed_metric="tool_use_quality_v1",
          prompt_template=None,
      )

  def test_multi_turn_trace_rejects_prompt_or_response_placeholders(self):
    multi_turn = _load_fixture("multi_turn_trace.json")
    for bad_template in (
        "Check if {prompt} matches {response}",
        "Check if {prompt.text} matches {events}",
        "Check if {response[0]} matches {events}",
    ):
      with self.subTest(bad_template=bad_template):
        with self.assertRaises(ValueError):
          self.udf.validate_inputs(
              trace_str=json.dumps(multi_turn),
              managed_metric=None,
              prompt_template=bad_template,
          )

  def test_request_builder_generates_valid_predefined_metric_spec(self):
    single_turn = _load_fixture("single_turn_trace.json")
    request = self.udf.build_evaluation_request(
        project_id="test-project",
        region="us-central1",
        trace_dict=single_turn,
        managed_metric="tool_use_quality_v1",
        prompt_template=None,
    )
    self.assertEqual(
        request["location"], "projects/test-project/locations/us-central1"
    )
    self.assertEqual(
        request["metrics"][0]["predefined_metric_spec"]["metric_spec_name"],
        "tool_use_quality_v1",
    )
    self.assertEqual(request["instance"]["agent_eval_data"], single_turn)

  def test_request_builder_generates_valid_llm_based_metric_spec(self):
    single_turn = _load_fixture("single_turn_trace.json")
    prompt = "Is this polite: {agent_data}?"
    request = self.udf.build_evaluation_request(
        project_id="test-project",
        region="us-central1",
        trace_dict=single_turn,
        managed_metric=None,
        prompt_template=prompt,
    )
    self.assertEqual(
        request["location"], "projects/test-project/locations/us-central1"
    )
    self.assertEqual(
        request["metrics"][0]["llm_based_metric_spec"][
            "metric_prompt_template"
        ],
        prompt,
    )
    self.assertEqual(request["instance"]["agent_eval_data"], single_turn)

  def test_response_builder_extracts_score_from_response(self):
    response = _load_fixture("resp_success_custom.json")
    parsed = self.udf.parse_evaluation_response(response)
    self.assertEqual(parsed["score"], 1.0)
    self.assertIsNone(parsed["status"])

  def test_response_parser_converts_missing_rubric_verdict_to_false(self):
    response = {
        "metricResults": [{
            "score": 0.0,
            "explanation": "Tone was aggressive.",
            "rubricVerdicts": [{
                "evaluatedRubric": {
                    "rubricId": "politeness",
                    "content": {
                        "property": {
                            "description": "Must speak politely.",
                        }
                    },
                    "type": "CUSTOM",
                    "importance": "HIGH",
                },
                "reasoning": "Agent insulted the user.",
            }],
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertEqual(len(parsed["rubric_verdicts"]), 1)
    verdict = parsed["rubric_verdicts"][0]
    self.assertEqual(verdict["rubric_id"], "politeness")
    self.assertEqual(verdict["rubric_content"], "Must speak politely.")
    self.assertEqual(verdict["type"], "CUSTOM")
    self.assertEqual(verdict["importance"], "HIGH")
    self.assertEqual(verdict["reasoning"], "Agent insulted the user.")
    self.assertFalse(verdict["verdict"])

  def test_response_parser_suppresses_score_to_none_when_error_present(self):
    response = {
        "metricResults": [{
            "score": 0.0,
            "explanation": "Something failed during evaluation",
            "error": {
                "code": 3,
                "message": "Model generated invalid output",
            },
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertIsNone(parsed["score"])
    self.assertIsNone(parsed["explanation"])
    self.assertEqual(parsed["rubric_verdicts"], [])
    self.assertEqual(
        parsed["status"], "INVALID_ARGUMENT: Model generated invalid output"
    )

  def test_response_parser_ignores_empty_or_zero_code_error(self):
    response = {
        "metricResults": [{
            "score": 0.9,
            "explanation": "Good output.",
            "error": {
                "code": 0,
                "message": "",
            },
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertEqual(parsed["score"], 0.9)
    self.assertEqual(parsed["explanation"], "Good output.")
    self.assertIsNone(parsed["status"])

  def test_response_parser_handles_top_level_error_response(self):
    response = _load_fixture("resp_error_json_parse.json")
    parsed = self.udf.parse_evaluation_response(response, http_status=400)
    self.assertIsNone(parsed["score"])
    self.assertIsNone(parsed["explanation"])
    self.assertEqual(parsed["rubric_verdicts"], [])
    self.assertEqual(
        parsed["status"],
        "INVALID_ARGUMENT: Error parsing JSON. Last error: Expecting property"
        " name enclosed in double quotes: line 1 column 2 (char 1)",
    )

  def test_response_parser_uses_error_status_name_for_non_400_errors(self):
    response = {
        "error": {
            "code": 429,
            "message": (
                "Quota exceeded for quota metric 'evaluate_instances_requests'"
            ),
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    parsed = self.udf.parse_evaluation_response(response, http_status=429)
    self.assertIsNone(parsed["score"])
    self.assertEqual(
        parsed["status"],
        "RESOURCE_EXHAUSTED: Quota exceeded for quota metric"
        " 'evaluate_instances_requests'",
    )

  def test_response_parser_handles_top_level_error_without_status_key(self):
    response = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for evaluate requests.",
        }
    }
    parsed = self.udf.parse_evaluation_response(response, http_status=429)
    self.assertIsNone(parsed["score"])
    self.assertEqual(
        parsed["status"],
        "RESOURCE_EXHAUSTED: Quota exceeded for evaluate requests.",
    )

  def test_response_parser_handles_explicit_null_error_payload(self):
    response = {"error": None, "metricResults": []}
    parsed = self.udf.parse_evaluation_response(response, http_status=500)
    self.assertIsNone(parsed["score"])
    self.assertEqual(
        parsed["status"],
        "INTERNAL: empty metric results in response (HTTP 500)",
    )

  def test_response_parser_handles_empty_results_fallback(self):
    response = {"metricResults": []}
    parsed = self.udf.parse_evaluation_response(response, http_status=200)
    self.assertIsNone(parsed["score"])
    self.assertEqual(
        parsed["status"],
        "INTERNAL: empty metric results in response (HTTP 200)",
    )

  def test_response_parser_handles_per_metric_error_with_zero_code_and_message(
      self,
  ):
    response = {
        "metricResults": [{
            "score": 0.0,
            "explanation": None,
            "error": {
                "code": 0,
                "message": "Evaluation pipeline failed internally.",
            },
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertIsNone(parsed["score"])
    self.assertEqual(
        parsed["status"], "UNKNOWN: Evaluation pipeline failed internally."
    )

  def test_response_parser_maps_unrecognized_error_code_to_unknown(self):
    response = {
        "metricResults": [{
            "score": 0.0,
            "error": {"code": 99, "message": "Unexpected failure."},
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertIsNone(parsed["score"])
    self.assertEqual(parsed["status"], "UNKNOWN: Unexpected failure.")

  def test_response_parser_reads_recorded_managed_metric_response(self):
    # Recorded from a live final_response_quality_v2 call on 2026-11-14. GEAP
    # sent an empty explanation, an empty rubric type, no rubric ID, and an
    # empty error object on success.
    response = _load_fixture(
        "resp_success_final_response_quality_v2_recorded.json"
    )
    parsed = self.udf.parse_evaluation_response(response)
    self.assertEqual(
        parsed,
        {
            "score": 1.0,
            "explanation": None,
            "rubric_verdicts": [{
                "rubric_id": None,
                "rubric_content": (
                    "The final answer states the correct capital of France."
                ),
                "type": None,
                "importance": "HIGH",
                "verdict": True,
                "reasoning": (
                    "The final answer correctly states that the capital of"
                    " France is Paris."
                ),
            }],
            "status": None,
        },
    )

  def test_response_parser_reads_recorded_v1_judge_not_found_error(self):
    # Recorded from a live final_response_quality_v1 call on 2026-11-14 in a
    # project that hasn't used Gemini 2.5 Pro, the judge for v1 metrics. The
    # project number is replaced and GEAP's debug details are removed.
    response = _load_fixture("resp_error_v1_judge_not_found.json")
    parsed = self.udf.parse_evaluation_response(response, http_status=404)
    self.assertIsNone(parsed["score"])
    self.assertEqual(parsed["rubric_verdicts"], [])
    self.assertRegex(
        parsed["status"], r"^NOT_FOUND: .*Autorater model not found"
    )

  def test_response_parser_handles_non_mapping_first_result(self):
    for bad_response in (
        {"metricResults": [None]},
        {"metricResults": ["string"]},
        {"metricResults": [123]},
    ):
      with self.subTest(bad_response=bad_response):
        parsed = self.udf.parse_evaluation_response(bad_response)
        self.assertIsNone(parsed["score"])
        self.assertIsNone(parsed["explanation"])
        self.assertEqual(parsed["rubric_verdicts"], [])
        self.assertEqual(
            parsed["status"],
            "INTERNAL: empty metric results in response (HTTP 200)",
        )

  def test_response_parser_handles_explicit_null_rubric_verdicts(self):
    response = {
        "metricResults": [{
            "score": 0.8,
            "explanation": "Mostly good.",
            "rubricVerdicts": None,
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertEqual(parsed["score"], 0.8)
    self.assertEqual(parsed["rubric_verdicts"], [])
    self.assertIsNone(parsed["status"])

  def test_response_parser_rubric_verdicts_edge_cases(self):
    response = {
        "metricResults": [{
            "score": 0.5,
            "explanation": "Mixed performance.",
            "rubricVerdicts": [
                {
                    "evaluatedRubric": {
                        "rubricId": "r1",
                        "content": {
                            "property": {"description": "Property description."}
                        },
                        "type": "CUSTOM",
                        "importance": "IMPORTANCE_UNSPECIFIED",
                    },
                    "reasoning": "Criteria was fully met.",
                    "verdict": True,
                },
                {
                    "evaluatedRubric": {
                        "rubricId": "",
                        "content": {"property": {"description": ""}},
                        "type": "",
                    },
                    "reasoning": "",
                    "verdict": False,
                },
            ],
        }]
    }
    parsed = self.udf.parse_evaluation_response(response)
    self.assertEqual(len(parsed["rubric_verdicts"]), 2)

    v1 = parsed["rubric_verdicts"][0]
    self.assertEqual(v1["rubric_id"], "r1")
    self.assertEqual(v1["rubric_content"], "Property description.")
    self.assertEqual(v1["type"], "CUSTOM")
    self.assertIsNone(v1["importance"])
    self.assertTrue(v1["verdict"])
    self.assertEqual(v1["reasoning"], "Criteria was fully met.")

    v2 = parsed["rubric_verdicts"][1]
    self.assertIsNone(v2["rubric_id"])
    self.assertIsNone(v2["rubric_content"])
    self.assertIsNone(v2["type"])
    self.assertIsNone(v2["importance"])
    self.assertFalse(v2["verdict"])
    self.assertIsNone(v2["reasoning"])

  def test_prompt_template_placeholders_validation(self):
    single_turn = json.dumps(_load_fixture("single_turn_trace.json"))
    multi_turn = json.dumps(_load_fixture("multi_turn_trace.json"))

    # Valid whitespace, dotted paths, and filter expressions
    for valid_template in (
        "Check { agent_data }",
        "Evaluate {agent_data.turns}",
        "Trace: {agent_data | trim}",
        "Check {events} and {agent_data}",
    ):
      with self.subTest(template=valid_template):
        self.udf.validate_inputs(
            trace_str=single_turn,
            managed_metric=None,
            prompt_template=valid_template,
        )

    # Rejection of {question_block} alone (rubric injection variable)
    with self.assertRaisesRegex(
        ValueError,
        "Custom prompt_template must contain at least one placeholder",
    ):
      self.udf.validate_inputs(
          trace_str=single_turn,
          managed_metric=None,
          prompt_template="Answer the question: {question_block}",
      )

    # Rejection of numeric literals like {1}
    with self.assertRaisesRegex(
        ValueError,
        "Custom prompt_template must contain at least one placeholder",
    ):
      self.udf.validate_inputs(
          trace_str=single_turn,
          managed_metric=None,
          prompt_template="Template with number {1}",
      )

    # Multi-turn trace rejects { prompt } with spaces or filter
    for invalid_multi_template in (
        "Check { prompt }",
        "Check {response | trim}",
    ):
      with self.subTest(template=invalid_multi_template):
        with self.assertRaisesRegex(
            ValueError,
            "Multi-turn metrics cannot be used with {prompt} or {response}",
        ):
          self.udf.validate_inputs(
              trace_str=multi_turn,
              managed_metric=None,
              prompt_template=invalid_multi_template,
          )

  def test_get_credentials_with_default_scope(self):
    class FakeCred:
      token = "fake-token"
      valid = True

    with mock.patch.object(
        self.udf.google.auth,
        "default",
        return_value=(FakeCred(), "test-project"),
        create=True,
    ) as mock_default:
      token, status = self.udf.get_credentials()
      mock_default.assert_called_once_with(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
      self.assertEqual(token, "fake-token")
      self.assertIsNone(status)

  def test_get_credentials_missing_token_returns_unauthenticated_status(self):
    class FakeCred:
      token = None
      valid = True

    with mock.patch.object(
        self.udf.google.auth,
        "default",
        return_value=(FakeCred(), "test-project"),
        create=True,
    ):
      token, status = self.udf.get_credentials()
      self.assertIsNone(token)
      self.assertEqual(
          status,
          "UNAUTHENTICATED: Failed to obtain authentication token from"
          " credentials.",
      )

  def test_get_credentials_library_unavailable_returns_unauthenticated_status(
      self,
  ):
    with mock.patch.object(self.udf.google, "auth", new=object()):
      token, status = self.udf.get_credentials()
      self.assertIsNone(token)
      self.assertEqual(
          status, "UNAUTHENTICATED: google.auth library not available."
      )

  def test_get_credentials_auth_error_returns_unauthenticated_status(self):
    def failing_default(scopes=None):
      del scopes
      raise self.udf.google.auth.exceptions.GoogleAuthError("ADC not found")

    with mock.patch.object(
        self.udf.google.auth,
        "default",
        side_effect=failing_default,
        create=True,
    ):
      token, status = self.udf.get_credentials()
      self.assertIsNone(token)
      self.assertEqual(status, "UNAUTHENTICATED: ADC not found")

  def test_get_credentials_returns_token_and_caches_creds(self):
    class FakeCred:

      def __init__(self):
        self.token = "initial-token"
        self.valid = True

    call_count = 0

    def fake_default(scopes=None):
      del scopes
      nonlocal call_count
      call_count += 1
      return FakeCred(), "test-project-123"

    with mock.patch.object(
        self.udf.google.auth,
        "default",
        side_effect=fake_default,
        create=True,
    ):
      token1, status1 = self.udf.get_credentials()
      token2, status2 = self.udf.get_credentials()
      self.assertEqual(token1, "initial-token")
      self.assertIsNone(status1)
      self.assertEqual(token2, "initial-token")
      self.assertIsNone(status2)
      self.assertEqual(call_count, 1)

  def test_evaluate_with_retries_retries_429_until_success(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.Response(
            status_code=429,
            json_data={
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "rate limit",
                }
            },
        ),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 0.8}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)

  def test_evaluate_with_retries_retries_500_until_success(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.Response(
            status_code=500,
            json_data={
                "error": {
                    "status": "INTERNAL",
                    "message": "server error",
                }
            },
        ),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 0.8}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)

  def test_evaluate_with_retries_retries_timeout_until_success(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.exceptions.Timeout("Read timed out"),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 0.9}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)

  def test_evaluate_with_retries_aborts_without_retry_on_non_retryable_error(
      self,
  ):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.Response(
            status_code=404,
            json_data={
                "error": {
                    "status": "NOT_FOUND",
                    "message": "Not found",
                }
            },
        )
    ])
    self.assertEqual(attempts, 1)
    self.assertEqual(http_code, 404)
    self.assertEqual(error, "NOT_FOUND: Not found")
    self.assertIsNone(resp)

  def test_evaluate_with_retries_aborts_after_max_attempts(self):
    resp, http_code, error, attempts, _, slept = self._run_retry_test([
        self.udf.requests.Response(
            status_code=429,
            json_data={
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "rate limit exceeded",
                }
            },
        )
    ])
    self.assertEqual(attempts, 5)
    self.assertEqual(http_code, 429)
    self.assertEqual(error, "RESOURCE_EXHAUSTED: rate limit exceeded")
    self.assertIsNone(resp)
    self.assertEqual(slept, [1.0, 2.0, 4.0, 8.0])

  def test_evaluate_with_retries_honors_retry_after_header(self):
    resp, http_code, error, attempts, _, slept = self._run_retry_test([
        self.udf.requests.Response(
            status_code=429,
            json_data={"error": {"message": "rate limit"}},
            headers={"Retry-After": "15"},
        ),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 1.0}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)
    self.assertTrue(slept)
    self.assertGreaterEqual(slept[0], 15.0)

  def test_evaluate_with_retries_aborts_on_wait_budget_exceeded(self):
    resp, http_code, error, attempts, _, slept = self._run_retry_test([
        self.udf.requests.Response(
            status_code=429,
            json_data={
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "rate limit exceeded",
                }
            },
            headers={"Retry-After": "65"},
        )
    ])
    self.assertEqual(attempts, 1)
    self.assertEqual(http_code, 429)
    self.assertEqual(error, "RESOURCE_EXHAUSTED: rate limit exceeded")
    self.assertIsNone(resp)
    self.assertEqual(slept, [])

  def test_evaluate_with_retries_aborts_on_elapsed_wall_clock_timeout_without_sleeping(
      self,
  ):
    fake_time = 0.0
    slept_durations = []

    def fake_send():
      nonlocal fake_time
      # Simulate a request that consumes the full 60s timeout budget.
      fake_time += 60.0
      raise self.udf.requests.exceptions.Timeout("Read timed out")

    def fake_sleep(duration):
      nonlocal fake_time
      fake_time += duration
      slept_durations.append(duration)

    resp, http_code, error = self.udf.evaluate_with_retries(
        send_fn=fake_send,
        sleep_fn=fake_sleep,
        clock_fn=lambda: fake_time,
    )
    self.assertIsNone(resp)
    self.assertEqual(http_code, 408)
    self.assertEqual(error, "DEADLINE_EXCEEDED: Read timed out")
    self.assertEqual(slept_durations, [])

  def test_evaluate_with_retries_aborts_when_sleep_would_exceed_wall_clock_budget(
      self,
  ):
    fake_time = 0.0
    slept_durations = []

    def fake_send():
      nonlocal fake_time
      # Request took 59s, leaving only 1s in the 60s wall-clock budget.
      fake_time += 59.0
      return self.udf.requests.Response(
          status_code=503,
          json_data={"error": {"message": "Service Unavailable"}},
      )

    def fake_sleep(duration):
      nonlocal fake_time
      fake_time += duration
      slept_durations.append(duration)

    # Initial backoff is 1.0s + jitter 1.0s = 2.0s wait, which exceeds 1s
    # remaining.
    resp, http_code, error = self.udf.evaluate_with_retries(
        send_fn=fake_send,
        sleep_fn=fake_sleep,
        clock_fn=lambda: fake_time,
        random_fn=lambda a, b: b,
    )
    self.assertIsNone(resp)
    self.assertEqual(http_code, 503)
    self.assertEqual(error, "UNAVAILABLE: Service Unavailable")
    self.assertEqual(slept_durations, [])

  def test_geap_evaluate_validation_failure_returns_invalid_argument_status(
      self,
  ):
    result = self.udf.geap_evaluate(
        trace_str='{"turns": []}',
        managed_metric="tool_use_quality_v1",
        prompt_template=None,
    )
    self.assertIsNone(result["score"])
    self.assertIsNone(result["explanation"])
    self.assertEqual(result["rubric_verdicts"], [])
    self.assertTrue(result["status"].startswith("INVALID_ARGUMENT"))

  def test_geap_evaluate_success(self):
    single_turn = _load_fixture("single_turn_trace.json")
    success_resp = {
        "metricResults": [{
            "score": 1.0,
            "explanation": "Agent followed instructions.",
            "rubricVerdicts": [],
        }]
    }
    fake_resp = self.udf.requests.Response(
        status_code=200,
        json_data=success_resp,
    )
    result = self._evaluate_with_mocks(
        trace_str=json.dumps(single_turn),
        creds_return=("test-token", None),
        retry_return=(fake_resp, 200, None),
    )
    self.assertIsNotNone(result)
    self.assertEqual(result["score"], 1.0)
    self.assertIsNotNone(result["explanation"])
    self.assertEqual(result["rubric_verdicts"], [])
    self.assertIsNone(result["status"])

  def test_geap_evaluate_prefers_configured_project_over_adc(self):
    single_turn = _load_fixture("single_turn_trace.json")
    self.udf._DEFAULT_PROJECT_ID = "configured-corp-proj"

    captured_url: Optional[str] = None
    captured_payload: Optional[dict[str, Any]] = None

    def fake_send(url, payload, token):
      nonlocal captured_url, captured_payload
      captured_url = url
      captured_payload = payload
      return self.udf.requests.Response(
          status_code=200,
          json_data={"metricResults": [{"score": 1.0, "rubricVerdicts": []}]},
      )

    old_creds = self.udf._CACHED_CREDENTIALS
    self.udf._CACHED_CREDENTIALS = None
    self.addCleanup(setattr, self.udf, "_CACHED_CREDENTIALS", old_creds)

    fake_creds = types.SimpleNamespace(
        valid=True,
        token="fake-token",
        service_account_email="my-sa@developer.gserviceaccount.com",
    )

    with mock.patch.object(
        self.udf.google.auth,
        "default",
        return_value=(fake_creds, "adc-worker-runtime-proj"),
    ), mock.patch.object(
        self.udf,
        "execute_http_request",
        side_effect=fake_send,
    ):
      result = self.udf.geap_evaluate(
          trace_str=json.dumps(single_turn),
          managed_metric="tool_use_quality_v1",
      )

    self.assertIsNone(result["status"])
    self.assertIsNotNone(captured_url)
    self.assertIn("projects/configured-corp-proj/locations/", captured_url)
    self.assertIsNotNone(captured_payload)
    self.assertEqual(
        captured_payload["location"],
        f"projects/configured-corp-proj/locations/{self.udf._DEFAULT_GEAP_REGION}",
    )

  def test_geap_evaluate_rejects_unfilled_project_without_calling(self):
    self.udf._DEFAULT_PROJECT_ID = self.udf._UNFILLED_PROJECT_ID
    single_turn = _load_fixture("single_turn_trace.json")
    with mock.patch.object(
        self.udf, "get_credentials"
    ) as mock_creds, mock.patch.object(
        self.udf, "execute_http_request"
    ) as mock_send:
      result = self.udf.geap_evaluate(
          trace_str=json.dumps(single_turn),
          managed_metric="tool_use_quality_v1",
      )
    self.assertEqual(
        result["status"],
        "INTERNAL: install project ID was never filled in; rerun install.sh.",
    )
    mock_creds.assert_not_called()
    mock_send.assert_not_called()

  def test_unfilled_project_sentinel_survives_placeholder_substitution(self):
    # install.sh replaces YOUR_PROJECT_ID everywhere in the module text. The
    # sentinel must keep the placeholder name so an installed module counts as
    # filled in.
    source = (_find_geap_dir() / "geap_evaluate_udf.py").read_text(
        encoding="utf-8"
    )
    installed_source = source.replace("YOUR_PROJECT_ID", "configured-proj")
    with tempfile.TemporaryDirectory() as tmpdir:
      installed_path = pathlib.Path(tmpdir) / "installed_geap_evaluate_udf.py"
      installed_path.write_text(installed_source, encoding="utf-8")
      spec = importlib.util.spec_from_file_location(
          "installed_geap_evaluate_udf", installed_path
      )
      self.assertIsNotNone(spec)
      self.assertIsNotNone(spec.loader)
      module = importlib.util.module_from_spec(spec)
      with mock.patch.dict(sys.modules, _build_fake_modules()):
        spec.loader.exec_module(module)
    self.assertEqual(module._DEFAULT_PROJECT_ID, "configured-proj")
    self.assertEqual(module._UNFILLED_PROJECT_ID, "YOUR_PROJECT_ID")

  def test_evaluate_with_retries_aborts_immediately_on_deterministic_request_exceptions(
      self,
  ):
    resp, http_code, error, attempts, _, slept = self._run_retry_test([
        self.udf.requests.exceptions.InvalidURL("Invalid URL format"),
    ])
    self.assertEqual(attempts, 1)
    self.assertEqual(http_code, 400)
    self.assertEqual(error, "INVALID_ARGUMENT: Invalid URL format")
    self.assertIsNone(resp)
    self.assertEqual(slept, [])

  def test_validate_inputs_rejects_non_finite_floats(self):
    nan_trace = '{"turns": [{"role": "user", "score": NaN}]}'
    with self.assertRaisesRegex(ValueError, "Failed to parse trace"):
      self.udf.validate_inputs(
          nan_trace, managed_metric="tool_use_quality_v1", prompt_template=None
      )
    result = self.udf.geap_evaluate(
        trace_str=nan_trace, managed_metric="tool_use_quality_v1"
    )
    self.assertTrue(result["status"].startswith("INVALID_ARGUMENT"))

  def test_geap_evaluate_rejects_none_trace(self):
    result = self.udf.geap_evaluate(
        trace_str=None, managed_metric="tool_use_quality_v1"
    )
    self.assertEqual(
        result["status"], "INVALID_ARGUMENT: trace cannot be None."
    )

  def test_geap_evaluate_handles_non_mapping_json_response(self):
    single_turn = _load_fixture("single_turn_trace.json")
    fake_resp = self.udf.requests.Response(status_code=200)
    fake_resp.json = lambda: ["unexpected-list"]
    result = self._evaluate_with_mocks(
        trace_str=json.dumps(single_turn),
        creds_return=("test-token", None),
        retry_return=(fake_resp, 200, None),
    )
    self.assertIsNone(result["score"])
    self.assertEqual(
        result["status"],
        "INTERNAL: empty metric results in response (HTTP 200)",
    )

  def test_validate_inputs_rejects_deeply_nested_json(self):
    with mock.patch.object(
        self.udf.json,
        "loads",
        side_effect=RecursionError("maximum recursion depth exceeded"),
    ):
      with self.assertRaisesRegex(ValueError, "Failed to parse trace"):
        self.udf.validate_inputs(
            "{}", managed_metric="tool_use_quality_v1", prompt_template=None
        )
      result = self.udf.geap_evaluate(
          trace_str="{}", managed_metric="tool_use_quality_v1"
      )
    self.assertTrue(result["status"].startswith("INVALID_ARGUMENT"))

  def test_http_session_mounts_https_adapter_with_pool_size_100(self):
    session = self.udf._get_http_session()
    self.assertIn("https://", session.mounted_adapters)
    self.assertEqual(session.mounted_adapters["https://"].pool_maxsize, 100)

  def test_concurrent_initialization_initializes_once(self):
    old_creds = self.udf._CACHED_CREDENTIALS
    old_session = self.udf._HTTP_SESSION
    self.udf._CACHED_CREDENTIALS = None
    self.udf._HTTP_SESSION = None
    self.addCleanup(setattr, self.udf, "_CACHED_CREDENTIALS", old_creds)
    self.addCleanup(setattr, self.udf, "_HTTP_SESSION", old_session)

    cred_calls = 0
    session_calls = 0
    cred_barrier = threading.Barrier(2)
    session_barrier = threading.Barrier(2)

    def slow_auth_default(scopes=None):
      del scopes
      nonlocal cred_calls
      cred_calls += 1
      if cred_calls == 1:
        # Hold the lock only if another thread can enter; a 2-party barrier
        # would deadlock if locked, so use a short yield instead.
        threading.Event().wait(0.02)
      return (
          types.SimpleNamespace(
              valid=True, token="tok", service_account_email=""
          ),
          "proj",
      )

    def make_session():
      nonlocal session_calls
      session_calls += 1
      if session_calls == 1:
        threading.Event().wait(0.02)
      return FakeSession()

    with mock.patch.object(
        self.udf.google.auth, "default", side_effect=slow_auth_default
    ), mock.patch.object(
        self.udf.requests, "Session", side_effect=make_session
    ):

      cred_results: list[Any] = []
      session_results: list[Any] = []
      thread_errors: list[Exception] = []

      def run_creds():
        try:
          cred_barrier.wait(timeout=5.0)
          cred_results.append(self.udf.get_credentials())
        except Exception as exc:  # pylint: disable=broad-except
          thread_errors.append(exc)

      def run_session():
        try:
          session_barrier.wait(timeout=5.0)
          session_results.append(self.udf._get_http_session())
        except Exception as exc:  # pylint: disable=broad-except
          thread_errors.append(exc)

      threads = [
          threading.Thread(target=run_creds),
          threading.Thread(target=run_creds),
          threading.Thread(target=run_session),
          threading.Thread(target=run_session),
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())

    self.assertEqual(thread_errors, [])
    self.assertEqual(cred_calls, 1)
    self.assertEqual(session_calls, 1)
    self.assertEqual(cred_results, [("tok", None), ("tok", None)])
    self.assertEqual(len(session_results), 2)
    self.assertIs(session_results[0], session_results[1])

  def test_execute_http_request_posts_with_headers_and_payload(self):
    fake_resp = self.udf.requests.Response(
        status_code=200, json_data={"ok": True}
    )
    mock_session = mock.MagicMock()
    mock_session.post.return_value = fake_resp
    resp = self.udf.execute_http_request(
        url="https://test.endpoint/eval",
        payload={"key": "val"},
        token="test-token",
        timeout=30.0,
        session=mock_session,
    )
    self.assertEqual(resp.status_code, 200)
    mock_session.post.assert_called_once_with(
        "https://test.endpoint/eval",
        headers={
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
        },
        json={"key": "val"},
        timeout=30.0,
    )

  def test_execute_http_request_raises_timeout_on_timeout(self):
    mock_session = mock.MagicMock()
    mock_session.post.side_effect = self.udf.requests.exceptions.Timeout(
        "timed out"
    )
    with self.assertRaises(self.udf.requests.exceptions.Timeout):
      self.udf.execute_http_request(
          url="https://test.endpoint/eval",
          payload={},
          token="test-token",
          session=mock_session,
      )

  def test_execute_http_request_raises_request_exception_on_failure(self):
    mock_session = mock.MagicMock()
    mock_session.post.side_effect = (
        self.udf.requests.exceptions.RequestException("connection failed")
    )
    with self.assertRaises(self.udf.requests.exceptions.RequestException):
      self.udf.execute_http_request(
          url="https://test.endpoint/eval",
          payload={},
          token="test-token",
          session=mock_session,
      )

  def test_execute_http_request_preserves_connection_error(self):
    mock_session = mock.MagicMock()
    mock_session.post.side_effect = (
        self.udf.requests.exceptions.ConnectionError("connection dropped")
    )
    with self.assertRaises(self.udf.requests.exceptions.ConnectionError):
      self.udf.execute_http_request(
          url="https://test.endpoint/eval",
          payload={},
          token="test-token",
          session=mock_session,
      )

  def test_get_credentials_refreshes_invalid_token(self):
    class ExpiredCred:

      def __init__(self):
        self.token = "initial-token"
        self.valid = False
        self.refresh_called = False

      def refresh(self, request):
        del request
        self.token = "refreshed-token"
        self.valid = True
        self.refresh_called = True

    cred = ExpiredCred()
    with mock.patch.object(
        self.udf.google.auth,
        "default",
        return_value=(cred, "test-project-123"),
        create=True,
    ):
      token, status = self.udf.get_credentials()
      self.assertEqual(token, "refreshed-token")
      self.assertIsNone(status)
      self.assertTrue(cred.refresh_called)

  def test_get_credentials_does_not_cache_when_refresh_fails(self):
    requests_exc = self.udf.requests.exceptions

    class FailingCred:

      def __init__(self):
        self.token = "initial-token"
        self.valid = False

      def refresh(self, request):
        del request
        raise requests_exc.ConnectionError("Auth server down")

    cred = FailingCred()
    with mock.patch.object(
        self.udf.google.auth,
        "default",
        return_value=(cred, "test-project-123"),
        create=True,
    ):
      token, status = self.udf.get_credentials()
      self.assertIsNone(token)
      self.assertEqual(status, "UNAUTHENTICATED: Auth server down")
      self.assertIsNone(self.udf._CACHED_CREDENTIALS)

  def test_get_credentials_catches_request_exception_and_resets_cache(self):
    requests_exc = self.udf.requests.exceptions

    class CachedCred:

      def __init__(self):
        self.token = "valid-token"
        self.valid = False

      def refresh(self, request):
        del request
        raise requests_exc.Timeout("Auth token refresh timeout")

    self.udf._CACHED_CREDENTIALS = CachedCred()
    token, status = self.udf.get_credentials()
    self.assertIsNone(token)
    self.assertEqual(status, "UNAUTHENTICATED: Auth token refresh timeout")
    self.assertIsNone(self.udf._CACHED_CREDENTIALS)

  def test_evaluate_with_retries_retries_connection_error_until_success(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.exceptions.ConnectionError("connection reset"),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 1.0}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)

  def test_evaluate_with_retries_retries_generic_request_exception(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.exceptions.RequestException("stream error"),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 1.0}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)

  def test_evaluate_with_retries_handles_invalid_retry_after_header(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test([
        self.udf.requests.Response(
            status_code=429,
            json_data={"error": {"message": "rate limit"}},
            headers={"Retry-After": "invalid-float"},
        ),
        self.udf.requests.Response(
            status_code=200,
            json_data={"metricResults": [{"score": 1.0}]},
        ),
    ])
    self.assertEqual(attempts, 2)
    self.assertEqual(http_code, 200)
    self.assertIsNone(error)
    self.assertIsNotNone(resp)

  def test_evaluate_with_retries_falls_back_to_response_text_on_malformed_json(
      self,
  ):
    class MalformedResponse:
      status_code = 404
      text = "Not Found Plaintext"
      headers = {}

      def json(self):
        raise ValueError("Invalid JSON")

    resp, http_code, error, attempts, _, _ = self._run_retry_test(
        [MalformedResponse()]
    )
    self.assertEqual(attempts, 1)
    self.assertEqual(http_code, 404)
    self.assertEqual(error, "NOT_FOUND: Not Found Plaintext")
    self.assertIsNone(resp)

    long_resp = types.SimpleNamespace(
        status_code=503,
        text="x" * 700,
        json=lambda: (_ for _ in ()).throw(ValueError("Invalid JSON")),
    )
    long_error = self.udf._extract_response_error(long_resp, 503)
    self.assertEqual(long_error, f"UNAVAILABLE: {'x' * 512}")

  def test_extract_response_error_preserves_canonical_status(self):
    resp = self.udf.requests.Response(
        status_code=403,
        json_data={
            "error": {
                "status": "PERMISSION_DENIED",
                "message": "Permission denied on resource",
            }
        },
    )
    error = self.udf._extract_response_error(resp, 403)
    self.assertEqual(error, "PERMISSION_DENIED: Permission denied on resource")

  def test_extract_response_error_resolves_code_and_fallback_status(self):
    resp = self.udf.requests.Response(
        status_code=429,
        json_data={
            "error": {
                "code": 429,
                "message": "Rate limit exceeded",
            }
        },
    )
    error = self.udf._extract_response_error(resp, 429)
    self.assertEqual(error, "RESOURCE_EXHAUSTED: Rate limit exceeded")

  def test_geap_evaluate_auth_error_returns_status(self):
    single_turn = _load_fixture("single_turn_trace.json")
    result = self._evaluate_with_mocks(
        trace_str=json.dumps(single_turn),
        creds_return=(None, "UNAUTHENTICATED: missing credentials"),
    )
    self.assertIsNone(result["score"])
    self.assertIsNone(result["explanation"])
    self.assertEqual(result["rubric_verdicts"], [])
    self.assertEqual(result["status"], "UNAUTHENTICATED: missing credentials")

  def test_geap_evaluate_retry_error_returns_status(self):
    single_turn = _load_fixture("single_turn_trace.json")
    result = self._evaluate_with_mocks(
        trace_str=json.dumps(single_turn),
        creds_return=("test-token", None),
        retry_return=(None, 429, "rate limit exceeded"),
    )
    self.assertIsNone(result["score"])
    self.assertIsNone(result["explanation"])
    self.assertEqual(result["rubric_verdicts"], [])
    self.assertEqual(result["status"], "rate limit exceeded")

  def test_status_mapping_includes_retryable_codes_408_and_502(self):
    self.assertEqual(
        self.udf._HTTP_STATUS_TO_CANONICAL_NAME.get(408), "DEADLINE_EXCEEDED"
    )
    self.assertEqual(
        self.udf._HTTP_STATUS_TO_CANONICAL_NAME.get(502), "UNAVAILABLE"
    )

  def test_evaluate_with_retries_prefixes_connection_error_with_unavailable(
      self,
  ):
    resp, http_code, error, attempts, _, _ = self._run_retry_test(
        [
            self.udf.requests.exceptions.ConnectionError(
                "Connection reset by peer"
            ),
        ]
        * self.udf._MAX_RETRY_ATTEMPTS
    )
    self.assertIsNone(resp)
    self.assertEqual(http_code, 503)
    self.assertIsNotNone(error)
    self.assertTrue(
        error.startswith("UNAVAILABLE:"),
        f"Expected UNAVAILABLE prefix, got {error}",
    )

  def test_evaluate_with_retries_prefixes_request_exception_with_internal(self):
    resp, http_code, error, attempts, _, _ = self._run_retry_test(
        [
            self.udf.requests.exceptions.RequestException("Stream failure"),
        ]
        * self.udf._MAX_RETRY_ATTEMPTS
    )
    self.assertIsNone(resp)
    self.assertEqual(http_code, 500)
    self.assertIsNotNone(error)
    self.assertTrue(
        error.startswith("INTERNAL:"),
        f"Expected INTERNAL prefix, got {error}",
    )

  def test_get_http_session_caches_session_instance(self):
    session1 = self.udf._get_http_session()
    session2 = self.udf._get_http_session()
    self.assertIs(session1, session2)

  def test_geap_evaluate_unexpected_exception_returns_internal_status(self):
    single_turn = _load_fixture("single_turn_trace.json")
    with mock.patch.object(
        self.udf,
        "get_credentials",
        side_effect=RuntimeError("unexpected runtime failure"),
    ):
      result = self.udf.geap_evaluate(
          trace_str=json.dumps(single_turn),
          managed_metric="tool_use_quality_v1",
          prompt_template=None,
      )
      self.assertIsNone(result["score"])
      self.assertIsNone(result["explanation"])
      self.assertEqual(result["rubric_verdicts"], [])
      self.assertEqual(result["status"], "INTERNAL: unexpected runtime failure")

  def test_geap_evaluate_sql_matches_generator_output(self):
    geap_dir = _find_geap_dir()
    sql_path = geap_dir / "geap_evaluate.sql"
    udf_path = geap_dir / "geap_evaluate_udf.py"
    self.assertTrue(sql_path.is_file(), f"Missing file: {sql_path}")
    self.assertTrue(udf_path.is_file(), f"Missing file: {udf_path}")

    generator_module = _load_generator_module()
    expected_sql = generator_module.generate_sql(
        udf_path.read_text(encoding="utf-8")
    )
    actual_sql = sql_path.read_text(encoding="utf-8")
    self.assertEqual(
        actual_sql,
        expected_sql,
        "geap_evaluate.sql does not match generate_udf_sql.py output. Run"
        " python3 sql_scripts/agent_evaluation/geap/tools/generate_udf_sql.py"
        " to regenerate.",
    )

  def test_generate_udf_sql_refuses_triple_single_quotes(self):
    generator_module = _load_generator_module()
    with self.assertRaisesRegex(ValueError, "triple single quotes"):
      generator_module.generate_sql("bad_code = '''premature end'''")

  def test_generate_udf_sql_body_size_limit(self):
    generator_module = _load_generator_module()
    # Content exactly at limit succeeds.
    sql_at_limit = generator_module.generate_sql(
        "#" * generator_module.MAX_UDF_BODY_BYTES
    )
    self.assertIn("CREATE OR REPLACE FUNCTION", sql_at_limit)

    # Content exceeding limit by 1 byte raises ValueError.
    with self.assertRaisesRegex(
        ValueError, "exceeds the BigQuery inline limit"
    ):
      generator_module.generate_sql(
          "#" * (generator_module.MAX_UDF_BODY_BYTES + 1)
      )

  def test_geap_evaluate_python_udf_body_size_within_limit(self):
    geap_dir = _find_geap_dir()
    sql_text = (geap_dir / "geap_evaluate.sql").read_text(encoding="utf-8")
    py_bodies = _extract_udf_python_bodies(sql_text)
    self.assertEqual(len(py_bodies), 1)
    body_bytes_len = len(py_bodies[0].encode("utf-8"))
    generator_module = _load_generator_module()
    self.assertLessEqual(
        body_bytes_len,
        generator_module.MAX_UDF_BODY_BYTES,
        f"Python UDF body ({body_bytes_len} bytes) exceeds BigQuery limit of"
        f" {generator_module.MAX_UDF_BODY_BYTES} bytes.",
    )

  def test_geap_evaluate_sql_placeholders_and_signature_contract(self):
    geap_dir = _find_geap_dir()
    sql_text = (geap_dir / "geap_evaluate.sql").read_text(encoding="utf-8")
    expected_placeholders = {
        "YOUR_PROJECT_ID",
        "YOUR_DATASET_ID",
        "YOUR_CONNECTION_LOCATION",
        "YOUR_CONNECTION_ID",
        "YOUR_GEAP_REGION",
    }
    found_placeholders = set(re.findall(r"\bYOUR_[A-Z_]+\b", sql_text))
    self.assertEqual(
        found_placeholders,
        expected_placeholders,
        "Unexpected placeholders in SQL:"
        f" {found_placeholders ^ expected_placeholders}",
    )

    # Verify that the SQL RETURNS STRUCT schema fields match the dictionary
    # keys produced by geap_evaluate.
    expected_top_level_fields = {
        "score",
        "explanation",
        "rubric_verdicts",
        "status",
    }
    expected_rubric_fields = {
        "rubric_id",
        "rubric_content",
        "type",
        "importance",
        "verdict",
        "reasoning",
    }

    # Extract field names from both SQL routine definitions:
    # geap_evaluate_impl and geap_evaluate.
    struct_matches = re.findall(
        r"RETURNS\s+STRUCT<(?P<fields>.*?)>\s*(?:LANGUAGE|AS)",
        sql_text,
        flags=re.DOTALL,
    )
    self.assertEqual(
        len(struct_matches),
        2,
        f"Expected 2 RETURNS STRUCT blocks in SQL, found {len(struct_matches)}",
    )

    for idx, fields_block in enumerate(struct_matches, 1):
      # Match top-level fields (lines with 2 leading spaces)
      top_level_names = set(
          re.findall(r"^  ([a-z_]+)\s+", fields_block, flags=re.MULTILINE)
      )
      self.assertEqual(
          top_level_names,
          expected_top_level_fields,
          f"SQL routine {idx} top-level return struct fields mismatch:"
          f" {top_level_names ^ expected_top_level_fields}",
      )

      # Match inner rubric fields (lines with 4 leading spaces inside
      # rubric_verdicts).
      rubric_match = re.search(
          r"rubric_verdicts\s+ARRAY<STRUCT<(?P<rubric_fields>.*?)>>",
          fields_block,
          flags=re.DOTALL,
      )
      self.assertIsNotNone(
          rubric_match,
          f"Could not find rubric_verdicts STRUCT in SQL routine {idx}.",
      )
      rubric_fields_block = rubric_match.group("rubric_fields")
      rubric_field_names = set(
          re.findall(
              r"^    ([a-z_]+)\s+", rubric_fields_block, flags=re.MULTILINE
          )
      )
      self.assertEqual(
          rubric_field_names,
          expected_rubric_fields,
          f"SQL routine {idx} rubric verdict struct fields mismatch:"
          f" {rubric_field_names ^ expected_rubric_fields}",
      )

    # Verify that the Python UDF returns dictionaries with matching keys.
    dummy_result = self.udf.geap_evaluate(
        trace_str=None, managed_metric=None, prompt_template=None
    )
    self.assertEqual(set(dummy_result.keys()), expected_top_level_fields)

    inline_rubric_payload = {
        "metricResults": [{
            "rubricVerdicts": [{
                "evaluatedRubric": {
                    "rubricId": "politeness",
                    "content": {
                        "property": {
                            "description": "The agent must respond politely."
                        }
                    },
                    "type": "POLITENESS",
                    "importance": "HIGH",
                },
                "verdict": True,
                "reasoning": "Agent used respectful language.",
            }]
        }]
    }
    parsed_response = self.udf.parse_evaluation_response(inline_rubric_payload)
    self.assertTrue(parsed_response["rubric_verdicts"])
    self.assertEqual(
        set(parsed_response["rubric_verdicts"][0].keys()),
        expected_rubric_fields,
    )

  def _make_fake_cli_env(
      self,
      tmp_path: pathlib.Path,
      cli_log_file: Optional[pathlib.Path] = None,
      captured_sql_file: Optional[pathlib.Path] = None,
  ) -> dict[str, str]:
    """Creates hermetic fake gcloud and bq CLIs in tmp_path and returns env.

    The fake dataset exists in FAKE_EXISTING_DS_LOC (default us-central1).
    FAKE_DS_MISSING=1 makes it not exist, and FAKE_DS_SHOW_FAIL=1 makes the
    lookup fail with a non-"not found" error.
    """
    log_target = str(cli_log_file) if cli_log_file else "/dev/null"
    sql_target = str(captured_sql_file) if captured_sql_file else "/dev/null"

    fake_gcloud = tmp_path / "gcloud"
    fake_gcloud.write_text(
        f"""#!/usr/bin/env bash
echo "gcloud $*" >> "{log_target}"
if [[ "$1" == "config" && "$2" == "get-value" ]]; then
  echo "(unset)"
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)

    fake_bq = tmp_path / "bq"
    fake_bq.write_text(
        f"""#!/usr/bin/env bash
echo "bq $*" >> "{log_target}"
if [[ " $* " == *" --format=json "* && " $* " == *" --dataset "* ]]; then
  if [[ "${{FAKE_DS_SHOW_FAIL:-}}" == "1" ]]; then
    echo "BigQuery error in show operation: Access Denied" >&2
    exit 1
  fi
  if [[ "${{FAKE_DS_MISSING:-}}" == "1" ]]; then
    echo "BigQuery error in show operation: Not found: Dataset" >&2
    exit 1
  fi
  echo "{{\\"location\\": \\"${{FAKE_EXISTING_DS_LOC:-us-central1}}\\"}}"
  exit 0
fi
if [[ " $* " == *" --format=json "* ]]; then
  echo '{{"cloudResource": {{"serviceAccountId": "bqcx-111222333-test@gcp-sa-bigquery-condel.iam.gserviceaccount.com"}}}}'
  exit 0
fi
if [[ " $* " == *" show "* ]]; then
  exit 1
fi
if [[ " $* " == *" query "* ]]; then
  if [[ " $* " == *"smoke_test_metric"* ]]; then
    if [[ "${{FAKE_SMOKE_QUERY_FAIL:-}}" == "1" ]]; then
      echo "Error: INVALID_ARGUMENT in bq query flags" >&2
      exit 1
    fi
    if [[ "${{FAKE_SMOKE_FAIL:-}}" == "1" ]]; then
      echo "status"
      echo "INTERNAL: service unavailable"
      exit 0
    fi
    echo "status"
    echo "INVALID_ARGUMENT: Managed metric smoke_test_metric is not supported."
    exit 0
  fi
  cat > "{sql_target}"
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_bq.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    for var in ("PROJECT_ID", "DATASET_ID", "CONNECTION_ID"):
      env.pop(var, None)
    return env

  def _run_install(
      self, env: dict[str, str], *args: str
  ) -> subprocess.CompletedProcess[str]:
    """Runs install.sh with the given positional arguments and environment."""
    install_sh = _find_geap_dir() / "install.sh"
    return subprocess.run(
        ["bash", str(install_sh), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_geap_install_sh_requires_project_id(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      env = self._make_fake_cli_env(pathlib.Path(tmp_dir))
      proc = self._run_install(env)
      self.assertNotEqual(proc.returncode, 0)
      self.assertIn("Error: PROJECT_ID is not set", proc.stderr)

  def test_geap_install_sh_help(self):
    proc = self._run_install(os.environ.copy(), "--help")
    self.assertEqual(proc.returncode, 0)
    self.assertIn("Usage:", proc.stdout)
    self.assertIn("[PROJECT_ID] [DATASET_ID] [CONNECTION_ID]", proc.stdout)
    self.assertNotIn("LOCATION", proc.stdout)

  def test_geap_install_sh_validates_identifiers(self):
    bad_cases = [
        (
            "PROJECT_ID",
            ["bad project", "test_ds", "conn"],
            (
                "Error: PROJECT_ID must start with a lowercase letter and end"
                " with a lowercase letter or number, and contain only"
                " lowercase letters, numbers, and hyphens."
            ),
        ),
        (
            "DATASET_ID",
            ["test-project-123", "bad|ds", "conn"],
            (
                "Error: DATASET_ID must contain only letters, numbers, and"
                " underscores."
            ),
        ),
        (
            "CONNECTION_ID",
            ["test-project-123", "test_ds", "bad&conn"],
            (
                "Error: CONNECTION_ID must contain only letters, numbers, and"
                " underscores."
            ),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
      env = self._make_fake_cli_env(pathlib.Path(tmp_dir))
      for var_name, args, expected_error in bad_cases:
        with self.subTest(var=var_name):
          proc = self._run_install(env, *args)
          self.assertNotEqual(proc.returncode, 0)
          self.assertIn(expected_error, proc.stderr)

  def test_geap_install_sh_rejects_unsupported_dataset_region(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      env = self._make_fake_cli_env(pathlib.Path(tmp_dir))
      env["FAKE_EXISTING_DS_LOC"] = "asia-east1"
      proc = self._run_install(
          env, "test-project-123", "agent_analytics", "conn"
      )
      self.assertNotEqual(proc.returncode, 0)
      self.assertIn(
          "Error: Unsupported location 'asia-east1' for GEAP evaluation"
          " service.",
          proc.stderr,
      )

  def test_geap_install_sh_stops_on_missing_dataset_without_creating_it(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      cli_log_file = tmp_path / "cli_calls.log"
      env = self._make_fake_cli_env(tmp_path, cli_log_file=cli_log_file)
      env["FAKE_DS_MISSING"] = "1"
      proc = self._run_install(env, "customer-gcp-proj", "new_ds", "conn")
      self.assertNotEqual(proc.returncode, 0)
      self.assertIn(
          "Error: Dataset customer-gcp-proj:new_ds does not exist.",
          proc.stderr,
      )
      self.assertNotIn("mk --dataset", cli_log_file.read_text("utf-8"))

  def test_geap_install_sh_stops_on_dataset_lookup_failure(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      env = self._make_fake_cli_env(pathlib.Path(tmp_dir))
      env["FAKE_DS_SHOW_FAIL"] = "1"
      proc = self._run_install(env, "customer-gcp-proj", "ds", "conn")
      self.assertNotEqual(proc.returncode, 0)
      self.assertIn(
          "Error: Failed to inspect dataset customer-gcp-proj:ds", proc.stderr
      )
      self.assertIn("Access Denied", proc.stderr)

  def test_geap_install_sh_end_to_end_with_fake_clis(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      captured_sql_file = tmp_path / "captured_query.sql"
      cli_log_file = tmp_path / "cli_calls.log"
      env = self._make_fake_cli_env(
          tmp_path,
          cli_log_file=cli_log_file,
          captured_sql_file=captured_sql_file,
      )

      proc = self._run_install(
          env, "customer-gcp-proj", "custom_dataset", "custom_ge_conn"
      )
      self.assertEqual(
          proc.returncode,
          0,
          f"install.sh failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}",
      )

      # APIs are enabled before the dataset lookup, and IAM is never changed.
      cli_calls = cli_log_file.read_text(encoding="utf-8")
      self.assertLess(
          cli_calls.index("services enable bigquery.googleapis.com"),
          cli_calls.index("show --format=json --dataset"),
      )
      self.assertNotIn("iam-policy-binding", cli_calls)
      service_account = "serviceAccount:bqcx-111222333-test@gcp-sa-bigquery-condel.iam.gserviceaccount.com"
      self.assertIn(
          "Grant  : gcloud projects add-iam-policy-binding customer-gcp-proj"
          f" --member={service_account} --role=roles/aiplatform.user",
          proc.stdout,
      )
      self.assertIn(
          "Revoke : gcloud projects remove-iam-policy-binding"
          f" customer-gcp-proj --member={service_account}"
          " --role=roles/aiplatform.user",
          proc.stdout,
      )

      rendered_sql = captured_sql_file.read_text(encoding="utf-8")
      self.assertIn(
          "CREATE OR REPLACE FUNCTION"
          " `customer-gcp-proj.custom_dataset.geap_evaluate`",
          rendered_sql,
      )
      self.assertIn(
          "WITH CONNECTION `customer-gcp-proj.us-central1.custom_ge_conn`",
          rendered_sql,
      )
      self.assertIn('_DEFAULT_GEAP_REGION = "us-central1"', rendered_sql)
      # The global replace fills the project but leaves the "never filled in"
      # sentinel intact, so the installed UDF calls the install project.
      self.assertIn('_DEFAULT_PROJECT_ID = "customer-gcp-proj"', rendered_sql)
      self.assertIn('"YOUR_" + "PROJECT_ID"', rendered_sql)
      self.assertNotIn('!= "customer-gcp-proj"', rendered_sql)
      for placeholder in (
          "YOUR_PROJECT_ID",
          "YOUR_DATASET_ID",
          "YOUR_CONNECTION_LOCATION",
          "YOUR_CONNECTION_ID",
          "YOUR_GEAP_REGION",
      ):
        self.assertNotIn(placeholder, rendered_sql)

  def test_geap_install_sh_maps_eu_dataset_to_europe_west4(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      captured_sql_file = tmp_path / "captured_query.sql"
      env = self._make_fake_cli_env(
          tmp_path, captured_sql_file=captured_sql_file
      )
      env["FAKE_EXISTING_DS_LOC"] = "EU"
      proc = self._run_install(env, "customer-gcp-proj", "eu_dataset", "conn")
      self.assertEqual(proc.returncode, 0, proc.stderr)
      eu_sql = captured_sql_file.read_text(encoding="utf-8")
      self.assertIn('_DEFAULT_GEAP_REGION = "europe-west4"', eu_sql)
      self.assertIn("WITH CONNECTION `customer-gcp-proj.EU.conn`", eu_sql)

  def test_geap_install_sh_stops_when_smoke_call_fails(self):
    cases = [
        (
            "FAKE_SMOKE_FAIL",
            "Smoke test call failed to return expected INVALID_ARGUMENT status",
        ),
        # A non-zero bq exit fails even if stderr mentions INVALID_ARGUMENT.
        ("FAKE_SMOKE_QUERY_FAIL", "Smoke test query failed"),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
      base_env = self._make_fake_cli_env(pathlib.Path(tmp_dir))
      for flag, expected_error in cases:
        with self.subTest(flag=flag):
          env = dict(base_env, **{flag: "1"})
          proc = self._run_install(env, "customer-gcp-proj", "ds", "conn")
          self.assertNotEqual(proc.returncode, 0)
          self.assertIn(expected_error, proc.stderr)

  def test_geap_files_exist_and_cleanliness(self):
    geap_dir = _find_geap_dir()
    install_sh = geap_dir / "install.sh"
    readme_md = geap_dir / "README.md"
    parent_readme = geap_dir.parent / "README.md"
    geap_sql = geap_dir / "geap_evaluate.sql"

    self.assertTrue(install_sh.is_file())
    self.assertTrue(readme_md.is_file())
    self.assertTrue(parent_readme.is_file())

    # Check executable permission
    self.assertNotEqual(
        install_sh.stat().st_mode & stat.S_IXUSR,
        0,
        "install.sh must be executable (chmod +x)",
    )

    # Check license headers
    for path in (install_sh, geap_sql):
      content = path.read_text(encoding="utf-8")
      self.assertIn("Copyright 2026 Google LLC", content)
      self.assertIn("Licensed under the Apache License, Version 2.0", content)

    # Check parent README contains GEAP integration row
    parent_readme_content = parent_readme.read_text(encoding="utf-8")
    self.assertIn("geap_evaluate", parent_readme_content)
    self.assertIn("[`geap/`](geap/)", parent_readme_content)


if __name__ == "__main__":
  unittest.main()
