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
import pathlib
from typing import Any
import unittest


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


def _load_udf_module() -> Any:
  """Dynamically loads the geap_evaluate_udf module from disk.

  Returns:
    Loaded geap_evaluate_udf module object.

  Raises:
    ImportError: If the module spec or loader fails to load.
  """
  geap_dir = _find_geap_dir()
  module_path = geap_dir / "geap_evaluate_udf.py"
  spec = importlib.util.spec_from_file_location(
      "geap_evaluate_udf", module_path
  )
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {module_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


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
    cls.udf = _load_udf_module()

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
    self.udf.validate_inputs(
        trace_str=single_turn,
        managed_metric="tool_use_quality_v1",
        prompt_template=None,
    )
    # Valid multi-turn managed metric
    self.udf.validate_inputs(
        trace_str=multi_turn,
        managed_metric="multi_turn_task_success_v1",
        prompt_template=None,
    )
    # Valid custom prompt template
    self.udf.validate_inputs(
        trace_str=single_turn,
        managed_metric=None,
        prompt_template="Check {agent_data}",
    )
    # Valid None trace (only validates metrics)
    self.udf.validate_inputs(
        trace_str=None,
        managed_metric="tool_use_quality_v1",
        prompt_template=None,
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


if __name__ == "__main__":
  unittest.main()
