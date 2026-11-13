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

"""Unit tests for the DeepEval BigQuery Python UDF and installer script.

These tests use lightweight fakes for hermetic CI (no network calls or external
packages required). When bumping the pinned `deepeval` or `google-genai` package
versions in `deepeval_udfs.sql`, run a manual end-to-end smoke test in BigQuery
in addition to this unit test suite.

Run standalone with standard Python or pytest:
  python3 -m unittest discover -s sql_scripts/agent_evaluation/tests -p
  "*_test.py"
  pytest sql_scripts/agent_evaluation/tests/
"""

import asyncio
import dataclasses
import enum
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any
import unittest
from unittest import mock

_UDF_BODY_LIMIT_BYTES = 32768


def _find_script_dir() -> pathlib.Path:
  """Locates sql_scripts/agent_evaluation/deepeval relative to this test file."""
  for base in (
      pathlib.Path(__file__).parent.parent,
      pathlib.Path(__file__).resolve().parent.parent,
  ):
    candidate = base / 'deepeval'
    if (candidate / 'deepeval_udfs.sql').is_file():
      return candidate
  raise FileNotFoundError(
      'Could not locate sql_scripts/agent_evaluation/deepeval directory.'
  )


def _extract_udf_python_bodies(sql_text: str) -> list[str]:
  """Extracts all Python UDF bodies from SQL text."""
  bodies = re.findall(r"AS\s+r'''\n(.*?)\n''';", sql_text, flags=re.DOTALL)
  if not bodies:
    raise ValueError('Could not extract Python UDF body from deepeval_udfs.sql')
  return bodies


@dataclasses.dataclass
class FakeToolCall:
  name: str
  description: str | None = None
  input_parameters: dict[str, Any] | None = None
  output: Any = None


@dataclasses.dataclass
class FakeTurn:
  role: str
  content: str
  tools_called: list[FakeToolCall] | None = None
  retrieval_context: list[str] | None = None


@dataclasses.dataclass
class FakeLLMTestCase:
  input: str
  actual_output: str
  expected_output: str | None = None
  retrieval_context: list[str] | None = None
  context: list[str] | None = None
  tools_called: list[FakeToolCall] | None = None
  expected_tools: list[FakeToolCall] | None = None


@dataclasses.dataclass
class FakeConversationalTestCase:
  turns: list[FakeTurn]
  chatbot_role: str | None = None
  scenario: str | None = None
  expected_outcome: str | None = None
  user_description: str | None = None
  context: list[str] | None = None


class FakeLLMTestCaseParams(enum.Enum):
  INPUT = 'input'
  ACTUAL_OUTPUT = 'actual_output'
  EXPECTED_OUTPUT = 'expected_output'
  CONTEXT = 'context'
  RETRIEVAL_CONTEXT = 'retrieval_context'
  TOOLS_CALLED = 'tools_called'


class FakeTurnParams(enum.Enum):
  ROLE = 'role'
  CONTENT = 'content'
  RETRIEVAL_CONTEXT = 'retrieval_context'
  TOOLS_CALLED = 'tools_called'


class FakeGeminiModel:
  """Fake base GeminiModel recording initialization args and generate calls."""

  genai_models_mod: Any = None

  def __init__(
      self, model: str = '', project: str = '', location: str = '', **kwargs
  ):
    self.model = model
    self.project = project
    self.location = location
    self.kwargs = kwargs
    self.generate_calls: list[tuple[str, Any]] = []

  def generate(self, prompt: str, schema: Any = None) -> tuple[Any, float]:
    self.generate_calls.append((prompt, schema))
    if schema is not None and getattr(schema, 'REJECT_ADDITIONAL_PROPS', False):
      raise ValueError('Invalid schema: additional_properties is not supported')
    if schema is not None and getattr(
        schema, 'REJECT_CAMEL_ADDITIONAL_PROPS', False
    ):
      raise ValueError(
          '1 validation error for Schema\nadditionalProperties\n  Extra inputs'
          ' are not permitted [type=extra_forbidden, input_value=False,'
          ' input_type=bool]'
      )
    if self.genai_models_mod is not None:
      self.genai_models_mod.Models().generate_content()
    if schema is None:
      return '```json\n{"data": {"project": "demo-proj"}}\n```', 0.0
    return schema.model_validate({'data': {'project': 'demo-proj'}}), 0.0

  async def a_generate(
      self, prompt: str, schema: Any = None
  ) -> tuple[Any, float]:
    self.generate_calls.append((prompt, schema))
    if schema is not None and getattr(schema, 'REJECT_ADDITIONAL_PROPS', False):
      raise ValueError('Invalid schema: additional_properties is not supported')
    if schema is not None and getattr(
        schema, 'REJECT_CAMEL_ADDITIONAL_PROPS', False
    ):
      raise ValueError(
          '1 validation error for Schema\nadditionalProperties\n  Extra inputs'
          ' are not permitted [type=extra_forbidden, input_value=False,'
          ' input_type=bool]'
      )
    if self.genai_models_mod is not None:
      await self.genai_models_mod.AsyncModels().generate_content()
    if schema is None:
      return '```json\n{"data": {"project": "demo-proj"}}\n```', 0.0
    return schema.model_validate({'data': {'project': 'demo-proj'}}), 0.0


class FakeBaseMetric:
  """Fake DeepEval metric for testing evaluate_test_case without network calls."""

  fail_models: set[str] = set()
  fail_error_cls: type[Exception] = RuntimeError
  fail_error_msg: str = 'Transient error'
  score_override: float = 0.9
  last_instances: list['FakeBaseMetric'] = []

  def __init__(self, threshold: float = 0.7, model: Any = None, **kwargs):
    self.threshold = threshold
    self.model = model
    self.kwargs = kwargs
    self.score: float | None = None
    self.reason: str | None = None
    self.measured_test_case: Any = None
    self.measured_retrieval_context: list[str] = []
    FakeBaseMetric.last_instances.append(self)

  def measure(self, test_case: Any) -> float:
    if self.model is not None and self.model.model in self.fail_models:
      raise self.fail_error_cls(
          f'{self.fail_error_msg} from {self.model.model}'
      )
    if self.model is not None:
      self.model.generate('eval_prompt')
    self.measured_test_case = test_case
    self.measured_retrieval_context = list(
        getattr(test_case, 'retrieval_context', None) or []
    )
    self.score = self.score_override
    model_name = getattr(self.model, 'model', 'unknown')
    self.reason = f'Passed via {self.__class__.__name__} on {model_name}'
    return self.score

  def is_successful(self) -> bool:
    return self.score is not None and self.score >= self.threshold


def _load_udf_module(
    sa_email: str = 'bqcx-123456789012-abcd@gcp-sa-bigquery-condel.iam.gserviceaccount.com',
    default_project: str | None = 'fallback-project',
    default_location: str = 'us-central1',
    default_model: str = 'YOUR_MODEL_ID',
) -> dict[str, Any]:
  """Loads the Python UDF bodies from deepeval_udfs.sql with fake deepeval/google.auth modules."""
  script_dir = _find_script_dir()
  sql_text = (script_dir / 'deepeval_udfs.sql').read_text(encoding='utf-8')
  py_bodies = _extract_udf_python_bodies(sql_text)

  fake_deepeval: Any = types.ModuleType('deepeval')
  fake_metrics: Any = types.ModuleType('deepeval.metrics')
  fake_test_case: Any = types.ModuleType('deepeval.test_case')
  fake_models: Any = types.ModuleType('deepeval.models')

  for metric_cls_name in [
      'AnswerRelevancyMetric',
      'FaithfulnessMetric',
      'ContextualRelevancyMetric',
      'ContextualPrecisionMetric',
      'ContextualRecallMetric',
      'HallucinationMetric',
      'ToolCorrectnessMetric',
      'TaskCompletionMetric',
      'GEval',
      'TurnRelevancyMetric',
      'KnowledgeRetentionMetric',
      'ConversationCompletenessMetric',
      'RoleAdherenceMetric',
      'ConversationalGEval',
  ]:
    setattr(
        fake_metrics,
        metric_cls_name,
        type(metric_cls_name, (FakeBaseMetric,), {}),
    )

  fake_test_case.LLMTestCase = FakeLLMTestCase
  fake_test_case.LLMTestCaseParams = FakeLLMTestCaseParams
  fake_test_case.ToolCall = FakeToolCall
  fake_test_case.ConversationalTestCase = FakeConversationalTestCase
  fake_test_case.Turn = FakeTurn
  fake_test_case.TurnParams = FakeTurnParams
  fake_models.GeminiModel = FakeGeminiModel
  fake_deepeval.metrics = fake_metrics
  fake_deepeval.test_case = fake_test_case
  fake_deepeval.models = fake_models

  fake_google: Any = types.ModuleType('google')
  fake_google_auth: Any = types.ModuleType('google.auth')
  fake_google_genai: Any = types.ModuleType('google.genai')
  fake_google_genai_models: Any = types.ModuleType('google.genai.models')

  class FakeGenaiModels:

    def generate_content(self, *args, **kwargs):
      del args, kwargs
      return types.SimpleNamespace(
          usage_metadata=types.SimpleNamespace(
              prompt_token_count=40,
              tool_use_prompt_token_count=10,
              candidates_token_count=15,
              thoughts_token_count=5,
          )
      )

  class FakeGenaiAsyncModels:

    async def generate_content(self, *args, **kwargs):
      del args, kwargs
      return types.SimpleNamespace(
          usage_metadata=types.SimpleNamespace(
              prompt_token_count=60,
              tool_use_prompt_token_count=0,
              candidates_token_count=12,
              thoughts_token_count=8,
          )
      )

  fake_google_genai_models.Models = FakeGenaiModels
  fake_google_genai_models.AsyncModels = FakeGenaiAsyncModels
  fake_google_genai.models = fake_google_genai_models
  FakeGeminiModel.genai_models_mod = fake_google_genai_models

  class FakeCreds:

    def __init__(self, email: str):
      self.service_account_email = email

  fake_google_auth.default = lambda: (FakeCreds(sa_email), default_project)
  fake_google.auth = fake_google_auth
  fake_google.genai = fake_google_genai

  patched_deepeval_modules = {
      'deepeval': fake_deepeval,
      'deepeval.metrics': fake_metrics,
      'deepeval.test_case': fake_test_case,
      'deepeval.models': fake_models,
  }
  patched_exec_modules = {
      **patched_deepeval_modules,
      'google': fake_google,
      'google.auth': fake_google_auth,
      'google.genai': fake_google_genai,
      'google.genai.models': fake_google_genai_models,
  }

  namespace: dict[str, Any] = {'__name__': 'deepeval_udf_module'}
  deepeval_patcher = mock.patch.dict(sys.modules, patched_deepeval_modules)
  deepeval_patcher.start()
  _ACTIVE_PATCHERS.append(deepeval_patcher)
  with mock.patch.dict(sys.modules, patched_exec_modules):
    for idx, py_code in enumerate(py_bodies):
      if default_location != 'YOUR_LOCATION':
        py_code = py_code.replace(
            '"YOUR_LOCATION"', json.dumps(default_location)
        )
      if default_model != 'YOUR_MODEL_ID':
        # Deliberately mirrors the unquoted global `s/YOUR_MODEL_ID/.../g` in
        # install.sh to catch regressions if the sentinel uses uppercase.
        py_code = py_code.replace('YOUR_MODEL_ID', default_model)
      compiled = compile(py_code, f'<deepeval_udfs.sql:{idx}>', 'exec')
      exec(compiled, namespace)  # pylint: disable=exec-used
  return namespace


_ACTIVE_PATCHERS: list[Any] = []


class AgentEvaluationDeepevalTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    FakeBaseMetric.fail_models = set()
    FakeBaseMetric.fail_error_cls = RuntimeError
    FakeBaseMetric.fail_error_msg = 'Transient error'
    FakeBaseMetric.score_override = 0.9
    FakeBaseMetric.last_instances = []
    self.script_dir = _find_script_dir()

  def tearDown(self):
    while _ACTIVE_PATCHERS:
      _ACTIVE_PATCHERS.pop().stop()
    super().tearDown()

  def test_files_exist_and_have_license_headers(self):
    expected_files = [
        'install.sh',
        'deepeval_udfs.sql',
        'agent_events_v2_eval_queries.sql',
        'README.md',
        '../README.md',
    ]
    for filename in expected_files:
      path = self.script_dir / filename
      self.assertTrue(path.is_file(), f'Missing expected file: {path}')
      content = path.read_text(encoding='utf-8')
      self.assertTrue(content)
      if not filename.endswith('README.md'):
        self.assertIn('Copyright 2026 Google LLC', content)
        self.assertIn('Licensed under the Apache License, Version 2.0', content)

    install_mode = (self.script_dir / 'install.sh').stat().st_mode
    self.assertNotEqual(
        install_mode & stat.S_IXUSR,
        0,
        'install.sh must be executable (chmod +x)',
    )

  def test_gcp_project_discovery_from_bq_connection_sa(self):
    ns = _load_udf_module(
        sa_email='bqcx-987654321098-wxyz@gcp-sa-bigquery-condel.iam.gserviceaccount.com',
        default_project='ignored-project',
    )
    self.assertEqual(ns['GCP_PROJECT'], '987654321098')

    ns_fallback = _load_udf_module(
        sa_email='custom-sa@my-gcp-project.iam.gserviceaccount.com',
        default_project='my-gcp-project',
    )
    self.assertEqual(ns_fallback['GCP_PROJECT'], 'my-gcp-project')

    with mock.patch.dict(os.environ, {'GOOGLE_CLOUD_LOCATION': 'us'}):
      ns_us = _load_udf_module()
    self.assertEqual(ns_us['GCP_LOCATION'], 'us-central1')
    parsed_tools = ns_us['_parse_tools']([
        {'tool_name': 'run_sql', 'args': '{"sql": "SELECT 1"}'},
    ])
    self.assertEqual(len(parsed_tools), 1)
    self.assertEqual(parsed_tools[0].name, 'run_sql')
    self.assertEqual(parsed_tools[0].input_parameters, {'sql': 'SELECT 1'})

    with mock.patch.dict(os.environ, {'GOOGLE_CLOUD_LOCATION': 'eu'}):
      ns_eu = _load_udf_module()
    self.assertEqual(ns_eu['GCP_LOCATION'], 'europe-west4')

    # Gemini 3 models (such as the default gemini-3.8-flash) route to the
    # global endpoint by default, while regional models (e.g. gemini-2.5-flash,
    # including retry fallback), non-Gemini-3 model IDs, and explicit per-call
    # location overrides preserve the regional endpoint.
    with mock.patch.dict(
        os.environ, {'GOOGLE_CLOUD_LOCATION': 'asia-northeast1'}
    ):
      ns_regional = _load_udf_module()
      self.assertEqual(ns_regional['GCP_LOCATION'], 'asia-northeast1')
      default_res = ns_regional['evaluate_test_case'](
          json.dumps({'input': 'Q', 'actual_output': 'A'}),
          ['answer_relevancy'],
      )
      self.assertTrue(default_res[0]['passed'])
      self.assertEqual(
          FakeBaseMetric.last_instances[-1].model.location, 'global'
      )
      regional_res = ns_regional['evaluate_test_case'](
          json.dumps({
              'input': 'Q',
              'actual_output': 'A',
              'judge_model': 'gemini-2.5-flash',
          }),
          ['answer_relevancy'],
      )
      self.assertTrue(regional_res[0]['passed'])
      self.assertEqual(
          FakeBaseMetric.last_instances[-1].model.location, 'asia-northeast1'
      )
      explicit_loc_res = ns_regional['evaluate_test_case'](
          json.dumps({
              'input': 'Q',
              'actual_output': 'A',
              'location': 'asia-northeast1',
          }),
          ['answer_relevancy'],
      )
      self.assertTrue(explicit_loc_res[0]['passed'])
      self.assertEqual(
          FakeBaseMetric.last_instances[-1].model.model, 'gemini-3.8-flash'
      )
      self.assertEqual(
          FakeBaseMetric.last_instances[-1].model.location, 'asia-northeast1'
      )
      substring_model_res = ns_regional['evaluate_test_case'](
          json.dumps({
              'input': 'Q',
              'actual_output': 'A',
              'judge_model': 'custom-gemini-3-adapter',
          }),
          ['answer_relevancy'],
      )
      self.assertTrue(substring_model_res[0]['passed'])
      self.assertEqual(
          FakeBaseMetric.last_instances[-1].model.location, 'asia-northeast1'
      )
      FakeBaseMetric.fail_models = {'gemini-3.8-flash'}
      with mock.patch.object(
          ns_regional['time'], 'sleep', autospec=True
      ) as mock_sleep_regional:
        fallback_loc_res = ns_regional['evaluate_test_case'](
            json.dumps({'input': 'Q', 'actual_output': 'A'}),
            ['answer_relevancy'],
        )
      FakeBaseMetric.fail_models = set()
    self.assertEqual(mock_sleep_regional.call_count, 2)
    self.assertTrue(fallback_loc_res[0]['passed'])
    self.assertIn('[Judge: gemini-2.5-flash]', fallback_loc_res[0]['reason'])
    self.assertEqual(
        FakeBaseMetric.last_instances[-2].model.model, 'gemini-3.8-flash'
    )
    self.assertEqual(
        FakeBaseMetric.last_instances[-2].model.location, 'global'
    )
    self.assertEqual(
        FakeBaseMetric.last_instances[-1].model.model, 'gemini-2.5-flash'
    )
    self.assertEqual(
        FakeBaseMetric.last_instances[-1].model.location, 'asia-northeast1'
    )

    # Unconfigured location raises ValueError rather than silently routing to
    # us-central1.
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('GOOGLE_CLOUD_LOCATION', None)
      ns_unconfigured = _load_udf_module(default_location='YOUR_LOCATION')
    self.assertEqual(ns_unconfigured['GCP_LOCATION'], '')
    unconfigured_res = ns_unconfigured['evaluate_test_case'](
        json.dumps({'input': 'Q', 'actual_output': 'A'}),
        ['answer_relevancy'],
    )
    self.assertIsNone(unconfigured_res[0]['score'])
    self.assertIsNone(unconfigured_res[0]['passed'])
    self.assertIn(
        'ValueError: GCP location is not configured',
        unconfigured_res[0]['reason'],
    )

  def test_single_turn_evaluation_and_missing_context_handling(self):
    ns = _load_udf_module()
    evaluate_test_case = ns['evaluate_test_case']

    tc_json = json.dumps({
        'input': 'Where is order #999?',
        'actual_output': 'Order #999 shipped via UPS.',
        'retrieval_context': ['Order #999 shipped via UPS yesterday.'],
        'tools_called': [
            {'name': 'lookup_order', 'output': {'status': 'shipped'}},
            'audit_log_tool',
        ],
    })
    results = evaluate_test_case(
        tc_json,
        [
            'answer_relevancy',
            'faithfulness',
            'task_completion',
            'geval:Order Accuracy:Verify carrier name.',
        ],
    )
    self.assertEqual(len(results), 4)
    self.assertEqual(
        [r['metric'] for r in results],
        [
            'answer_relevancy',
            'faithfulness',
            'task_completion',
            'geval:Order Accuracy',
        ],
    )
    for res in results:
      self.assertAlmostEqual(res['score'], 0.9)
      self.assertTrue(res['passed'])
      self.assertIn('gemini-3.8-flash', res['reason'])

    # Verify GEval received RETRIEVAL_CONTEXT and TOOLS_CALLED in
    # evaluation_params.
    geval_inst = FakeBaseMetric.last_instances[-1]
    self.assertIn(
        FakeLLMTestCaseParams.RETRIEVAL_CONTEXT,
        geval_inst.kwargs['evaluation_params'],
    )
    self.assertIn(
        FakeLLMTestCaseParams.TOOLS_CALLED,
        geval_inst.kwargs['evaluation_params'],
    )

    # Verify failing score below threshold returns passed=False
    FakeBaseMetric.score_override = 0.3
    low_results = evaluate_test_case(tc_json, ['answer_relevancy'])
    self.assertAlmostEqual(low_results[0]['score'], 0.3)
    self.assertFalse(low_results[0]['passed'])
    FakeBaseMetric.score_override = 0.9

    # Without required fields or turns, context/expected_output/expected_tools
    # and multi-turn metrics return N/A.
    no_ctx_json = json.dumps({
        'input': 'Hello',
        'actual_output': 'Hi there!',
    })
    no_ctx_results = evaluate_test_case(
        no_ctx_json,
        [
            'faithfulness',
            'contextual_relevancy',
            'contextual_precision',
            'contextual_recall',
            'hallucination',
            'tool_correctness',
            'turn_relevancy',
            'knowledge_retention',
            'conversation_completeness',
            'role_adherence',
            'conversational_geval:Flow:Check flow',
        ],
    )
    self.assertEqual(len(no_ctx_results), 11)
    for res in no_ctx_results:
      self.assertIsNone(res['score'])
      self.assertIsNone(res['passed'])
      self.assertIn('N/A', res['reason'])

  def test_multi_turn_conversational_evaluation(self):
    ns = _load_udf_module()
    evaluate_test_case = ns['evaluate_test_case']

    convo_json = json.dumps({
        'chatbot_role': 'BigQuery Data Exploration Assistant',
        'turns': [
            {
                'input_query': 'List datasets in project demo-proj.',
                'actual_output': 'Found dataset sales_ds in demo-proj.',
                'retrieval_context': ['datasets: [sales_ds]'],
                'tools_called': [{'name': 'list_datasets'}],
            },
            {
                'role': 'system',
                'content': 'System instructions should be skipped.',
            },
            {
                'role': 'human',
                'content': 'What tables are in sales_ds?',
            },
            {
                'role': 'model',
                'content': 'sales_ds contains orders and customers.',
            },
        ],
    })
    results = evaluate_test_case(
        convo_json,
        [
            'turn_relevancy',
            'knowledge_retention',
            'conversation_completeness',
            'role_adherence',
            'conversational_geval:Carryover:Verify project context carryover.',
            'answer_relevancy',
            'geval:Single Turn Check:Verify response clarity.',
        ],
    )
    self.assertEqual(len(results), 7)
    self.assertEqual(
        [r['metric'] for r in results],
        [
            'turn_relevancy',
            'knowledge_retention',
            'conversation_completeness',
            'role_adherence',
            'conversational_geval:Carryover',
            'answer_relevancy',
            'geval:Single Turn Check',
        ],
    )
    for res in results:
      self.assertAlmostEqual(res['score'], 0.9)
      self.assertTrue(res['passed'])

    # Verify 'system' role was skipped, 'human' normalized to 'user', and
    # 'model' normalized to 'assistant'.
    convo_tc = FakeBaseMetric.last_instances[0].measured_test_case
    self.assertIsInstance(convo_tc, FakeConversationalTestCase)
    self.assertEqual(len(convo_tc.turns), 4)
    self.assertEqual(convo_tc.turns[-2].role, 'user')
    self.assertEqual(convo_tc.turns[-1].role, 'assistant')

    # Verify single-turn metrics received LLMTestCase when turns was present.
    single_tc = FakeBaseMetric.last_instances[-1].measured_test_case
    self.assertIsInstance(single_tc, FakeLLMTestCase)
    self.assertEqual(single_tc.input, 'What tables are in sales_ds?')
    self.assertEqual(
        single_tc.actual_output, 'sales_ds contains orders and customers.'
    )

  def test_safe_gemini_model_json_fallback_on_knowledge_and_additional_props(
      self,
  ):
    ns = _load_udf_module()
    safe_model_cls = ns['SafeGeminiModel']
    model = safe_model_cls(
        model='gemini-2.5-flash', project='12345', location='us-central1'
    )

    class Knowledge:

      @staticmethod
      def model_json_schema():
        return {'type': 'object', 'properties': {'data': {'type': 'object'}}}

      @staticmethod
      def model_validate(obj):
        return obj

    parsed, cost = model.generate('Extract knowledge facts', schema=Knowledge)
    self.assertEqual(parsed, {'data': {'project': 'demo-proj'}})
    self.assertEqual(cost, 0.0)
    # Schema should have been passed as None on the fallback call
    self.assertIsNone(model.generate_calls[-1][1])

    class StrictSchemaWithAdditionalProps:
      REJECT_ADDITIONAL_PROPS = True

      @staticmethod
      def model_json_schema():
        return {'type': 'object'}

      @staticmethod
      def model_validate(obj):
        return obj

    class StrictSchemaWithCamelAdditionalProps:
      REJECT_CAMEL_ADDITIONAL_PROPS = True

      @staticmethod
      def model_json_schema():
        return {'type': 'object'}

      @staticmethod
      def model_validate(obj):
        return obj

    parsed2, _ = model.generate(
        'Extract dict', schema=StrictSchemaWithAdditionalProps
    )
    self.assertEqual(parsed2, {'data': {'project': 'demo-proj'}})
    parsed3, _ = model.generate(
        'Extract camel additionalProperties',
        schema=StrictSchemaWithCamelAdditionalProps,
    )
    self.assertEqual(parsed3, {'data': {'project': 'demo-proj'}})

    # Verify async a_generate also shortcuts Knowledge and falls back on both
    # additional_properties and camelCase additionalProperties.
    parsed_async, _ = asyncio.run(
        model.a_generate('Extract knowledge async', schema=Knowledge)
    )
    self.assertEqual(parsed_async, {'data': {'project': 'demo-proj'}})
    parsed_async2, _ = asyncio.run(
        model.a_generate(
            'Extract dict async', schema=StrictSchemaWithAdditionalProps
        )
    )
    self.assertEqual(parsed_async2, {'data': {'project': 'demo-proj'}})
    parsed_async3, _ = asyncio.run(
        model.a_generate(
            'Extract camel additionalProperties async',
            schema=StrictSchemaWithCamelAdditionalProps,
        )
    )
    self.assertEqual(parsed_async3, {'data': {'project': 'demo-proj'}})

    # Verify malformed JSON / validation failure in fallback raises RuntimeError
    class RejectingSchema:

      @staticmethod
      def model_json_schema():
        return {'type': 'object'}

      @staticmethod
      def model_validate(obj):
        del obj
        raise ValueError('Schema validation failed on LLM output')

    with self.assertRaises(RuntimeError):
      model._generate_json_fallback('Extract invalid', RejectingSchema)

  def test_primary_model_id_survives_install_sed_substitution(self):
    # Unsubstituted placeholder resolves to the default gemini-3.8-flash model.
    ns_default = _load_udf_module()
    self.assertEqual(ns_default['PRIMARY_MODEL_ID'], 'gemini-3.8-flash')

    # Global sed replacement of YOUR_MODEL_ID by install.sh preserves the
    # user-supplied PRIMARY_MODEL_ID rather than resetting to gemini-3.8-flash.
    ns_custom_model = _load_udf_module(default_model='gemini-2.5-flash')
    self.assertEqual(ns_custom_model['PRIMARY_MODEL_ID'], 'gemini-2.5-flash')

  def test_retry_fallback_to_flash_lite_and_error_handling(self):
    ns = _load_udf_module()
    evaluate_test_case = ns['evaluate_test_case']

    # Fail primary model ('gemini-3.8-flash') so attempt 3 falls back to
    # 'gemini-2.5-flash'.
    FakeBaseMetric.fail_models = {'gemini-3.8-flash'}
    with mock.patch.object(ns['time'], 'sleep', autospec=True) as mock_sleep:
      results = evaluate_test_case(
          json.dumps({'input': 'Q', 'actual_output': 'A'}),
          ['answer_relevancy'],
      )
    self.assertEqual(len(results), 1)
    self.assertTrue(results[0]['passed'])
    self.assertIn('[Judge: gemini-2.5-flash]', results[0]['reason'])
    self.assertEqual(mock_sleep.call_count, 2)

    # Exhausting all 3 attempts (both primary and fallback fail) sleeps twice
    # and emits passed=None.
    FakeBaseMetric.fail_models = {'gemini-3.8-flash', 'gemini-2.5-flash'}
    with mock.patch.object(
        ns['time'], 'sleep', autospec=True
    ) as mock_sleep_all:
      err_results = evaluate_test_case(
          json.dumps({'input': 'Q', 'actual_output': 'A'}),
          ['answer_relevancy'],
      )
    self.assertEqual(mock_sleep_all.call_count, 2)
    self.assertIsNone(err_results[0]['passed'])
    self.assertIn('Error: RuntimeError:', err_results[0]['reason'])
    FakeBaseMetric.fail_models = set()

    # Verify DeepEval invalid-JSON ValueError on primary model is retried and
    # reaches fallback model (including custom judge_model /
    # fallback_judge_model).
    FakeBaseMetric.fail_models = {'custom-primary-model'}
    FakeBaseMetric.fail_error_cls = ValueError
    FakeBaseMetric.fail_error_msg = (
        'Evaluation LLM outputted an invalid JSON. Please use a better'
        ' evaluation model.'
    )
    with mock.patch.object(
        ns['time'], 'sleep', autospec=True
    ) as mock_sleep_json:
      json_retry_res = evaluate_test_case(
          json.dumps({
              'input': 'Q',
              'actual_output': 'A',
              'judge_model': 'custom-primary-model',
              'fallback_judge_model': 'custom-fallback-model',
          }),
          ['answer_relevancy'],
      )
    self.assertEqual(mock_sleep_json.call_count, 2)
    self.assertTrue(json_retry_res[0]['passed'])
    self.assertIn('[Judge: custom-fallback-model]', json_retry_res[0]['reason'])
    FakeBaseMetric.fail_models = set()
    FakeBaseMetric.fail_error_cls = RuntimeError
    FakeBaseMetric.fail_error_msg = 'Transient error'

    # Verify _is_non_retryable_error classification
    is_non_retryable = ns['_is_non_retryable_error']
    self.assertTrue(is_non_retryable(RuntimeError('HTTP 400: Bad Request')))
    self.assertTrue(is_non_retryable(RuntimeError('status_code: 403')))
    self.assertTrue(is_non_retryable(ValueError('Unsupported parameter type')))
    self.assertFalse(
        is_non_retryable(ValueError('Evaluation LLM outputted an invalid JSON'))
    )
    self.assertFalse(is_non_retryable(RuntimeError('HTTP 429: Rate limit')))
    self.assertFalse(
        is_non_retryable(json.JSONDecodeError('Expecting value', 'bad', 0))
    )

    # Invalid JSON, non-dict JSON, empty turns, and unsupported metric handling
    for bad_input in (
        '{not-valid-json',
        '[]',
        'null',
        json.dumps({'turns': []}),
    ):
      bad_json_res = evaluate_test_case(bad_input, ['answer_relevancy'])
      self.assertEqual(bad_json_res[0]['metric'], 'parse_error')
      self.assertFalse(bad_json_res[0]['passed'])

    unsupported_res = evaluate_test_case(
        json.dumps({'input': 'Q', 'actual_output': 'A'}),
        ['non_existent_metric', 'geval_typo_metric'],
    )
    self.assertEqual(len(unsupported_res), 2)
    for res in unsupported_res:
      self.assertFalse(res['passed'])
      self.assertIn('Unsupported metric name', res['reason'])

  def test_bqaa_event_rows_and_agent_data_input(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    evaluate_test_case = ns['evaluate_test_case']

    self.assertIsNone(parse_trace(''))
    self.assertIsNone(parse_trace(None))

    # 1. Single-turn ShopBot BQAA event array
    #    (TO_JSON_STRING(ARRAY_AGG(t ORDER BY timestamp)))
    shopbot_events = [
        {
            'timestamp': '2025-11-13T00:57:09.498843Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'ShopBot',
            'session_id': 'sess-shop-1',
            'invocation_id': 'inv-shop-1',
            'content': "User Content: text: 'Where is my order #999?'",
            'status': 'OK',
        },
        {
            'timestamp': '2025-11-13T00:57:09.499092Z',
            'event_type': 'TOOL_STARTING',
            'agent': 'ShopBot',
            'session_id': 'sess-shop-1',
            'invocation_id': 'inv-shop-1',
            'content': (
                'Tool Name: OrderTool, Description: Retrieves order status,'
                ' Arguments: {"order_id": "999"}'
            ),
            'status': 'OK',
        },
        {
            'timestamp': '2025-11-13T00:57:09.499130Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'ShopBot',
            'session_id': 'sess-shop-1',
            'invocation_id': 'inv-shop-1',
            'content': (
                'Tool Name: OrderTool, Result: {"status": "Shipped",'
                ' "tracking_id": "TRK12345"}'
            ),
            'status': 'OK',
        },
        {
            'timestamp': '2025-11-13T00:57:09.499181Z',
            'event_type': 'LLM_RESPONSE',
            'agent': 'ShopBot',
            'session_id': 'sess-shop-1',
            'invocation_id': 'inv-shop-1',
            'content': (
                "Tool Name: text_response, text: 'Your order #999 has been"
                " shipped. Tracking ID is TRK12345.'"
            ),
            'status': 'OK',
        },
    ]
    parsed_shop_trace = parse_trace(json.dumps(shopbot_events))
    res_shop = evaluate_test_case(
        parsed_shop_trace,
        [
            'answer_relevancy',
            'faithfulness',
            'task_completion',
            'geval:Technical Precision:Check tracking ID.',
        ],
    )
    self.assertEqual(len(res_shop), 4)
    for r in res_shop:
      self.assertTrue(r['passed'])
    measured_single = FakeBaseMetric.last_instances[0].measured_test_case
    self.assertIsInstance(measured_single, FakeLLMTestCase)
    self.assertEqual(measured_single.input, 'Where is my order #999?')
    self.assertEqual(
        measured_single.actual_output,
        'Your order #999 has been shipped. Tracking ID is TRK12345.',
    )
    self.assertIsNotNone(measured_single.tools_called)
    self.assertEqual(len(measured_single.tools_called), 1)
    self.assertEqual(measured_single.tools_called[0].name, 'OrderTool')
    self.assertEqual(
        measured_single.tools_called[0].input_parameters, {'order_id': '999'}
    )

    # 2. Multi-turn my_bq_agent BQAA session
    bq_agent_session_events = [
        {
            'timestamp': '2026-01-21T21:08:38.881890Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'tell me what is in my project p1'},
        },
        {
            'timestamp': '2026-01-21T21:08:38.909024Z',
            'event_type': 'AGENT_STARTING',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-1',
            'content': (
                'You are a helpful assistant with access to BigQuery tools.'
            ),
        },
        {
            'timestamp': '2026-01-21T21:09:00.994247Z',
            'event_type': 'TOOL_STARTING',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-1',
            'content': {
                'args': {'project_id': 'p1'},
                'tool': 'list_dataset_ids',
            },
        },
        {
            'timestamp': '2026-01-21T21:09:01.362022Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-1',
            'content': {'result': ['ds1', 'ds2'], 'tool': 'list_dataset_ids'},
        },
        {
            'timestamp': '2026-01-21T21:09:04.148653Z',
            'event_type': 'LLM_RESPONSE',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-1',
            'content': {'response': "text: 'Project p1 has ds1 and ds2.'"},
        },
        {
            'timestamp': '2026-01-21T21:09:12.266570Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'list tables in ds1'},
        },
        {
            'timestamp': '2026-01-21T21:09:16.099240Z',
            'event_type': 'LLM_RESPONSE',
            'agent': 'my_bq_agent',
            'session_id': 'sess-bq-1',
            'invocation_id': 'inv-2',
            'content': {'response': "text: 'ds1 contains cards and sets.'"},
        },
    ]
    parsed_multi_trace = parse_trace(json.dumps(bq_agent_session_events))
    res_multi = evaluate_test_case(
        parsed_multi_trace,
        [
            'turn_relevancy',
            'knowledge_retention',
            'conversation_completeness',
            'role_adherence',
            'conversational_geval:Carryover:Verify project p1 carryover.',
        ],
    )
    self.assertEqual(len(res_multi), 5)
    for r in res_multi:
      self.assertTrue(r['passed'])
    measured_convo = FakeBaseMetric.last_instances[-1].measured_test_case
    self.assertIsInstance(measured_convo, FakeConversationalTestCase)
    self.assertEqual(
        measured_convo.chatbot_role,
        'You are a helpful assistant with access to BigQuery tools.',
    )
    self.assertEqual(len(measured_convo.turns), 4)

    # 3. Direct canonical AgentData JSON input
    canonical_agent_data = {
        'agents': {
            'my_bq_agent': {
                'agent_id': 'my_bq_agent',
                'instruction': 'Helpful BQ Agent',
            }
        },
        'turns': [{
            'turn_index': 0,
            'turn_id': 't-0',
            'events': [
                {
                    'author': 'user',
                    'content': {
                        'role': 'user',
                        'parts': [{'text': 'What datasets exist?'}],
                    },
                },
                {
                    'author': 'my_bq_agent',
                    'content': {
                        'role': 'model',
                        'parts': [{
                            'function_call': {
                                'name': 'list_dataset_ids',
                                'args': {'project_id': 'p1'},
                            }
                        }],
                    },
                },
                {
                    'author': 'my_bq_agent',
                    'content': {
                        'role': 'user',
                        'parts': [{
                            'function_response': {
                                'name': 'list_dataset_ids',
                                'response': {'result': ['ds1']},
                            }
                        }],
                    },
                },
                {
                    'author': 'my_bq_agent',
                    'content': {
                        'role': 'model',
                        'parts': [{'text': 'Dataset ds1 exists.'}],
                    },
                },
            ],
        }],
    }
    parsed_ad_trace = parse_trace(json.dumps(canonical_agent_data))
    res_ad = evaluate_test_case(
        parsed_ad_trace,
        ['answer_relevancy', 'faithfulness', 'task_completion'],
    )
    self.assertEqual(len(res_ad), 3)
    for r in res_ad:
      self.assertTrue(r['passed'])

    # 4. Verify agent_events_v2_eval_queries.sql uses CTE-first grouping +
    #    deepeval_parse_trace + deepeval_evaluate
    queries_sql = (
        self.script_dir / 'agent_events_v2_eval_queries.sql'
    ).read_text(encoding='utf-8')
    self.assertIn('WITH grouped_invocations AS', queries_sql)
    self.assertIn('WITH grouped_sessions AS', queries_sql)
    self.assertIn('TO_JSON_STRING(events)', queries_sql)
    self.assertIn('deepeval_parse_trace', queries_sql)
    self.assertIn('deepeval_evaluate', queries_sql)

    # 5. Verify same-timestamp BQAA benchmark events:
    #    - AGENT_RESPONSE with response.parts[*].markdown and
    #      clarifying_question
    #    - TOOL_ERROR with content=None matched to TOOL_STARTING via span_id
    #    - Error-terminated invocation without AGENT_RESPONSE
    same_ts = '2026-10-03 00:32:43+00'
    markdown_and_clarify_events = [
        {
            'timestamp': same_ts,
            'event_type': 'AGENT_RESPONSE',
            'invocation_id': 'inv_md',
            'content': {
                'response': {
                    'parts': [{'markdown': 'Found **5** active schools.'}]
                }
            },
        },
        {
            'timestamp': same_ts,
            'event_type': 'USER_MESSAGE_RECEIVED',
            'invocation_id': 'inv_md',
            'content_parts': [
                {'mime_type': 'text/plain', 'text': 'How many active schools?'}
            ],
        },
        {
            'timestamp': same_ts,
            'event_type': 'USER_MESSAGE_RECEIVED',
            'invocation_id': 'inv_clarify',
            'content': {'text_summary': 'What is the best school?'},
        },
        {
            'timestamp': same_ts,
            'event_type': 'AGENT_RESPONSE',
            'invocation_id': 'inv_clarify',
            'content': {
                'response': {
                    'clarifying_question': 'Do you mean by SAT or FRPM?'
                }
            },
        },
    ]
    parsed_mc = json.loads(parse_trace(json.dumps(markdown_and_clarify_events)))
    self.assertEqual(len(parsed_mc['turns']), 2)
    self.assertEqual(parsed_mc['turns'][0]['input'], 'How many active schools?')
    self.assertEqual(
        parsed_mc['turns'][0]['actual_output'], 'Found **5** active schools.'
    )
    self.assertEqual(
        parsed_mc['turns'][1]['actual_output'], 'Do you mean by SAT or FRPM?'
    )

    error_span_events = [
        {
            'timestamp': same_ts,
            'event_type': 'TOOL_ERROR',
            'invocation_id': 'inv_err',
            'span_id': 'sp_sql_1',
            'content': None,
            'error_message': 'Query exceeded quota',
            'status': 'ERROR',
        },
        {
            'timestamp': same_ts,
            'event_type': 'TOOL_STARTING',
            'invocation_id': 'inv_err',
            'span_id': 'sp_sql_1',
            'content': {
                'tool_name': 'execute_sql',
                'tool_input': {'sql': 'SELECT 1'},
                'text_summary': 'execute_sql: SELECT 1',
            },
        },
        {
            'timestamp': same_ts,
            'event_type': 'USER_MESSAGE_RECEIVED',
            'invocation_id': 'inv_err',
            'content': {'text_summary': 'Run heavy query'},
        },
        {
            'timestamp': same_ts,
            'event_type': 'AGENT_ERROR',
            'invocation_id': 'inv_err',
            'error_message': 'Agent aborted on quota error',
            'status': 'ERROR',
        },
    ]
    parsed_err = json.loads(parse_trace(json.dumps(error_span_events)))
    self.assertEqual(parsed_err['input'], 'Run heavy query')
    self.assertEqual(
        parsed_err['actual_output'], 'Error: Agent aborted on quota error'
    )
    self.assertEqual(len(parsed_err['tools_called']), 1)
    self.assertEqual(parsed_err['tools_called'][0]['name'], 'execute_sql')
    self.assertEqual(
        parsed_err['tools_called'][0]['input_parameters'], {'sql': 'SELECT 1'}
    )

    # 6. Verify additional BQAA parser edge cases:
    #    - Wrapper retrieval_context preserved when ext_instr is present and no
    #      tools are called
    #    - AGENT_STARTING with JSON-encoded "{}" falls back to
    #      attributes.static_instruction
    #    - LLM_RESPONSE with "text: '...' | call: <tool>" is ignored and
    #      terminal error resets intermediate llm_response_text
    #    - Completed tools are popped from pending_tools so subsequent
    #      content=None TOOL_ERROR without span_id matches the active pending
    #      tool
    #    - Same-timestamp recovered TOOL_ERROR without AGENT_RESPONSE preserves
    #      recovered LLM_RESPONSE instead of overwriting actual_output with
    #      Error: ...
    wrapper_rc_payload = {
        'events': [
            {
                'timestamp': same_ts,
                'event_type': 'AGENT_STARTING',
                'invocation_id': 'inv_w',
                'content': '{}',
                'attributes': {
                    'static_instruction': 'Follow strict groundedness rules.'
                },
            },
            {
                'timestamp': same_ts,
                'event_type': 'USER_MESSAGE_RECEIVED',
                'invocation_id': 'inv_w',
                'content': {'text_summary': 'Summarize doc'},
            },
            {
                'timestamp': same_ts,
                'event_type': 'AGENT_RESPONSE',
                'invocation_id': 'inv_w',
                'content': {'text_summary': 'Doc summary.'},
            },
        ],
        'retrieval_context': ['Caller supplied chunk 1'],
    }
    parsed_w = json.loads(parse_trace(json.dumps(wrapper_rc_payload)))
    self.assertEqual(
        parsed_w['chatbot_role'], 'Follow strict groundedness rules.'
    )
    self.assertEqual(parsed_w['retrieval_context'], ['Caller supplied chunk 1'])
    self.assertNotIn('context', parsed_w)

    multi_tool_and_call_events = [
        {
            'timestamp': '2026-10-03 00:32:40+00',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'invocation_id': 'inv_mt',
            'content': {'text_summary': 'Check schema and run query'},
        },
        {
            'timestamp': '2026-10-03 00:32:41+00',
            'event_type': 'LLM_RESPONSE',
            'invocation_id': 'inv_mt',
            'content': "text: 'Let me check' | call: get_table_info",
        },
        {
            'timestamp': '2026-10-03 00:32:42+00',
            'event_type': 'TOOL_STARTING',
            'invocation_id': 'inv_mt',
            'content': {
                'tool_name': 'get_table_info',
                'tool_input': {'table': 'schools'},
            },
        },
        {
            'timestamp': '2026-10-03 00:32:43+00',
            'event_type': 'TOOL_COMPLETED',
            'invocation_id': 'inv_mt',
            'content': {
                'tool_name': 'get_table_info',
                'tool_output': {'columns': ['id']},
            },
        },
        {
            'timestamp': '2026-10-03 00:32:44+00',
            'event_type': 'TOOL_STARTING',
            'invocation_id': 'inv_mt',
            'content': {
                'tool_name': 'execute_sql',
                'tool_input': {'sql': 'SELECT id FROM schools'},
            },
        },
        {
            'timestamp': '2026-10-03 00:32:45+00',
            'event_type': 'TOOL_ERROR',
            'invocation_id': 'inv_mt',
            'content': None,
            'error_message': 'Table not found',
        },
        {
            'timestamp': '2026-10-03 00:32:46+00',
            'event_type': 'INVOCATION_ERROR',
            'invocation_id': 'inv_mt',
            'error_message': 'Invocation failed after SQL error',
        },
    ]
    parsed_mt = json.loads(parse_trace(json.dumps(multi_tool_and_call_events)))
    self.assertEqual(
        parsed_mt['actual_output'], 'Error: Invocation failed after SQL error'
    )
    self.assertEqual(len(parsed_mt['tools_called']), 2)
    self.assertEqual(parsed_mt['tools_called'][0]['name'], 'get_table_info')
    self.assertEqual(parsed_mt['tools_called'][1]['name'], 'execute_sql')
    self.assertEqual(
        parsed_mt['tools_called'][1]['input_parameters'],
        {'sql': 'SELECT id FROM schools'},
    )

    recovered_same_ts_events = [
        {
            'timestamp': same_ts,
            'event_type': 'USER_MESSAGE_RECEIVED',
            'invocation_id': 'inv_rec',
            'content': {'text_summary': 'Count active schools'},
        },
        {
            'timestamp': same_ts,
            'event_type': 'TOOL_STARTING',
            'invocation_id': 'inv_rec',
            'span_id': 'sp_rec_1',
            'content': {
                'tool_name': 'execute_sql',
                'tool_input': {'sql': 'SELECT COUNT(*) FROM bad_table'},
            },
        },
        {
            'timestamp': same_ts,
            'event_type': 'TOOL_ERROR',
            'invocation_id': 'inv_rec',
            'span_id': 'sp_rec_1',
            'content': None,
            'error_message': 'Table bad_table not found',
        },
        {
            'timestamp': same_ts,
            'event_type': 'LLM_RESPONSE',
            'invocation_id': 'inv_rec',
            'content': {'response': "text: 'Recovered: there are 42 schools.'"},
        },
    ]
    parsed_rec = json.loads(parse_trace(json.dumps(recovered_same_ts_events)))
    self.assertEqual(
        parsed_rec['actual_output'], 'Recovered: there are 42 schools.'
    )

    # 7. Verify terminal error resets, transfer tool filtering, and AgentData
    #    context fallback:
    #    - Terminal error (INVOCATION_ERROR / AGENT_ERROR) resets
    #      agent_response_text as well as llm_response_text
    #    - TOOL_ERROR on transfer_to_agent / adk_request_* is filtered out
    #    - _parse_agent_data_dict populates context=[primary_instr] when
    #      retrieval_context is empty
    ar_reset_and_transfer_err_events = [
        {
            'timestamp': '2026-10-03 00:32:40+00',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'invocation_id': 'inv_ar_err',
            'content': {'text_summary': 'Route and answer'},
        },
        {
            'timestamp': '2026-10-03 00:32:41+00',
            'event_type': 'TOOL_ERROR',
            'invocation_id': 'inv_ar_err',
            'content': {'tool_name': 'transfer_to_agent'},
            'error_message': 'Subagent unavailable',
        },
        {
            'timestamp': '2026-10-03 00:32:42+00',
            'event_type': 'AGENT_RESPONSE',
            'invocation_id': 'inv_ar_err',
            'content': {'text_summary': 'Partial response before fatal error'},
        },
        {
            'timestamp': '2026-10-03 00:32:43+00',
            'event_type': 'INVOCATION_ERROR',
            'invocation_id': 'inv_ar_err',
            'error_message': 'Fatal invocation error',
        },
    ]
    parsed_ar_err = json.loads(
        parse_trace(json.dumps(ar_reset_and_transfer_err_events))
    )
    self.assertEqual(
        parsed_ar_err['actual_output'], 'Error: Fatal invocation error'
    )
    self.assertEqual(parsed_ar_err['tools_called'], [])

    ad_no_tools = {
        'agents': {
            'my_bq_agent': {
                'agent_id': 'my_bq_agent',
                'instruction': 'Strict AgentData Instruction',
            }
        },
        'turns': [{
            'turn_index': 0,
            'turn_id': 't-0',
            'events': [
                {
                    'author': 'user',
                    'content': {
                        'role': 'user',
                        'parts': [{'text': 'Hello'}],
                    },
                },
                {
                    'author': 'my_bq_agent',
                    'content': {
                        'role': 'model',
                        'parts': [{'text': 'Hi!'}],
                    },
                },
            ],
        }],
    }
    parsed_ad_no_tools = json.loads(parse_trace(json.dumps(ad_no_tools)))
    self.assertEqual(
        parsed_ad_no_tools['context'], ['Strict AgentData Instruction']
    )

  def test_customer_feedback_enhancements_and_token_usage(self):
    ns = _load_udf_module()
    evaluate_test_case = ns['evaluate_test_case']

    # 1. Token usage fields (judge_input_tokens, judge_output_tokens) on LLM and
    #    skipped metrics (exercised end-to-end through google.genai.models.Models).
    tok_tc = json.dumps({
        'input': 'Order status?',
        'actual_output': 'Shipped via UPS.',
        'expected_output': 'Shipped via UPS.',
    })
    tok_res = evaluate_test_case(
        tok_tc,
        ['answer_relevancy', 'faithfulness', 'unknown_metric'],
    )
    self.assertEqual(len(tok_res), 3)
    # LLM metric records FakeGenaiModels.generate_content (50 in, 20 out)
    self.assertEqual(tok_res[0]['judge_input_tokens'], 50)
    self.assertEqual(tok_res[0]['judge_output_tokens'], 20)
    # Skipped metric (faithfulness without retrieval_context) -> None tokens
    self.assertIsNone(tok_res[1]['judge_input_tokens'])
    self.assertIsNone(tok_res[1]['judge_output_tokens'])
    # Unsupported metric -> None tokens
    self.assertIsNone(tok_res[2]['judge_input_tokens'])
    self.assertIsNone(tok_res[2]['judge_output_tokens'])

    # Verify AsyncModels.generate_content via SafeGeminiModel.a_generate
    judge = ns['_get_judge_model']('gemini-3.8-flash')
    ns['_TOKEN_USAGE'][:] = [0, 0]
    asyncio.run(judge.a_generate('async prompt'))
    self.assertEqual(ns['_TOKEN_USAGE'], [60, 20])

    # 2. GEval deduplication when retrieval_context consists solely of
    #    "Tool <name> returned: ..." and tools_called has output, plus
    #    single-pass evaluation_steps ("steps:...") support.
    FakeBaseMetric.last_instances.clear()
    dedup_tc = json.dumps({
        'input': 'Lookup table schema',
        'actual_output': 'Table sales has column id.',
        'retrieval_context': ['Tool get_schema returned: {"columns": ["id"]}'],
        'tools_called': [{'name': 'get_schema', 'output': {'columns': ['id']}}],
    })
    dedup_res = evaluate_test_case(
        dedup_tc,
        ['geval:Grounding:steps:Check column names;Verify no hallucinations'],
    )
    self.assertTrue(dedup_res[0]['passed'])
    geval_dedup_inst = FakeBaseMetric.last_instances[-1]
    self.assertEqual(
        geval_dedup_inst.kwargs['evaluation_steps'],
        ['Check column names', 'Verify no hallucinations'],
    )
    self.assertNotIn('criteria', geval_dedup_inst.kwargs)
    self.assertIn(
        FakeLLMTestCaseParams.TOOLS_CALLED,
        geval_dedup_inst.kwargs['evaluation_params'],
    )
    self.assertNotIn(
        FakeLLMTestCaseParams.RETRIEVAL_CONTEXT,
        geval_dedup_inst.kwargs['evaluation_params'],
    )
    self.assertNotIn(
        FakeLLMTestCaseParams.CONTEXT,
        geval_dedup_inst.kwargs['evaluation_params'],
    )

    # 3. Preserve RETRIEVAL_CONTEXT when tools_called lacks output, and preserve
    #    CONTEXT when context differs from retrieval_context; also test
    #    tc_data["evaluation_steps"] and empty "steps:" fallback.
    FakeBaseMetric.last_instances.clear()
    no_out_tc = json.dumps({
        'input': 'Lookup table schema',
        'actual_output': 'Table sales has column id.',
        'retrieval_context': ['Tool get_schema returned: {"columns": ["id"]}'],
        'context': ['Static schema policy'],
        'tools_called': ['get_schema'],
        'evaluation_steps': 'Step 1;Step 2',
    })
    evaluate_test_case(no_out_tc, ['geval:Grounding:steps:'])
    geval_no_out = FakeBaseMetric.last_instances[-1]
    self.assertEqual(
        geval_no_out.kwargs['evaluation_steps'], ['Step 1', 'Step 2']
    )
    self.assertIn(
        FakeLLMTestCaseParams.RETRIEVAL_CONTEXT,
        geval_no_out.kwargs['evaluation_params'],
    )
    self.assertIn(
        FakeLLMTestCaseParams.CONTEXT,
        geval_no_out.kwargs['evaluation_params'],
    )

    # 4. ConversationalGEval with inline steps: and turn-level RETRIEVAL_CONTEXT
    #    deduplication when turn tools_called has output.
    FakeBaseMetric.last_instances.clear()
    convo_dedup_tc = json.dumps({
        'turns': [
            {'role': 'user', 'content': 'Find order'},
            {
                'role': 'assistant',
                'content': 'Order 123 shipped.',
                'retrieval_context': ['Tool OrderTool returned: {"id": 123}'],
                'tools_called': [{'name': 'OrderTool', 'output': {'id': 123}}],
            },
        ]
    })
    evaluate_test_case(
        convo_dedup_tc,
        [
            'conversational_geval:Flow:steps:Verify turn flow;Check tool'
            ' grounding'
        ],
    )
    c_inst = FakeBaseMetric.last_instances[-1]
    self.assertEqual(
        c_inst.kwargs['evaluation_steps'],
        ['Verify turn flow', 'Check tool grounding'],
    )
    self.assertIn(
        FakeTurnParams.TOOLS_CALLED, c_inst.kwargs['evaluation_params']
    )
    self.assertNotIn(
        FakeTurnParams.RETRIEVAL_CONTEXT, c_inst.kwargs['evaluation_params']
    )

  def test_prior_turn_dialogue_carryover_in_retrieval_context(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    evaluate_test_case = ns['evaluate_test_case']
    # 1. 2-turn tool-less BQAA trace with AGENT_STARTING instruction:
    #    - retrieval_context has Prior turn 1
    #    - context falls back to [extracted_instruction] (not polluted by Prior turn 1)
    #    - per-turn turns[i]['retrieval_context'] stays empty (no duplication across turns)
    multi_turn_events = [
        {
            'timestamp': '2026-02-01T10:00:00Z',
            'event_type': 'AGENT_STARTING',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'You are SpaceAgent.'},
        },
        {
            'timestamp': '2026-02-01T10:00:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Who owns the sales domain?'},
        },
        {
            'timestamp': '2026-02-01T10:00:03Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Sales PO is Alice.'},
        },
        {
            'timestamp': '2026-02-01T10:01:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'And what table should I query?'},
        },
        {
            'timestamp': '2026-02-01T10:01:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'Query proj.sales.orders (PO: Alice).'},
        },
    ]
    parsed_mt = json.loads(parse_trace(json.dumps(multi_turn_events)))
    self.assertEqual(
        parsed_mt['retrieval_context'],
        [
            'Prior turn 1: User: Who owns the sales domain? | Assistant: Sales'
            ' PO is Alice.'
        ],
    )
    self.assertEqual(parsed_mt['context'], ['You are SpaceAgent.'])
    self.assertEqual(parsed_mt['turns'][0]['retrieval_context'], [])
    self.assertEqual(parsed_mt['turns'][1]['retrieval_context'], [])

    # Verify evaluate_test_case on 2-turn tool-less trace:
    # hallucination uses context=['You are SpaceAgent.'], single-turn geval
    # includes both CONTEXT and RETRIEVAL_CONTEXT, and contextual_relevancy
    # returns N/A because only Prior turn 1 dialogue is in retrieval_context.
    FakeBaseMetric.last_instances.clear()
    res_toolless = evaluate_test_case(
        json.dumps(parsed_mt),
        [
            'hallucination',
            'geval:Grounded:Check accuracy',
            'contextual_relevancy',
        ],
    )
    self.assertEqual(len(res_toolless), 3)
    self.assertIsNone(res_toolless[2]['score'])
    self.assertIn('N/A', res_toolless[2]['reason'])
    hall_inst, geval_toolless_inst = FakeBaseMetric.last_instances[-2:]
    self.assertEqual(
        hall_inst.measured_test_case.context, ['You are SpaceAgent.']
    )
    self.assertIn(
        FakeLLMTestCaseParams.CONTEXT,
        geval_toolless_inst.kwargs['evaluation_params'],
    )
    self.assertIn(
        FakeLLMTestCaseParams.RETRIEVAL_CONTEXT,
        geval_toolless_inst.kwargs['evaluation_params'],
    )

    # 2. 2-turn tool-using BQAA trace:
    #    - tool_rc precedes prior_rc in top-level retrieval_context
    #    - evaluate_test_case default context excludes 'Prior turn 1: ...'
    #    - single-turn geval strips 'Tool ... returned:' from retrieval_context
    #      when tools_called has outputs, leaving only 'Prior turn 1: ...'
    tool_mt_events = [
        {
            'timestamp': '2026-02-01T10:00:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Who owns the sales domain?'},
        },
        {
            'timestamp': '2026-02-01T10:00:02Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {
                'tool': 'lookup_owner',
                'result': {'owner': 'Alice'},
            },
        },
        {
            'timestamp': '2026-02-01T10:00:03Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Sales PO is Alice.'},
        },
        {
            'timestamp': '2026-02-01T10:01:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'And what table should I query?'},
        },
        {
            'timestamp': '2026-02-01T10:01:01Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {
                'tool': 'lookup_table',
                'result': {'table': 'proj.sales.orders'},
            },
        },
        {
            'timestamp': '2026-02-01T10:01:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'Query proj.sales.orders (PO: Alice).'},
        },
    ]
    parsed_tool_mt = json.loads(parse_trace(json.dumps(tool_mt_events)))
    self.assertEqual(
        parsed_tool_mt['retrieval_context'],
        [
            'Tool lookup_owner returned: {"owner": "Alice"}',
            'Tool lookup_table returned: {"table": "proj.sales.orders"}',
            (
                'Prior turn 1: User: Who owns the sales domain? | Assistant:'
                ' Sales PO is Alice.'
            ),
        ],
    )
    self.assertNotIn('context', parsed_tool_mt)

    FakeBaseMetric.last_instances.clear()
    evaluate_test_case(
        json.dumps(parsed_tool_mt),
        [
            'hallucination',
            'geval:DedupCheck:Check grounding',
            'contextual_relevancy',
            'faithfulness',
        ],
    )

    hall_tool_inst, geval_tool_inst, ctx_rel_inst, faith_tool_inst = (
        FakeBaseMetric.last_instances[-4:]
    )
    # Default context excludes 'Prior turn 1: ...'
    self.assertEqual(
        hall_tool_inst.measured_test_case.context,
        [
            'Tool lookup_owner returned: {"owner": "Alice"}',
            'Tool lookup_table returned: {"table": "proj.sales.orders"}',
        ],
    )
    # Single-turn geval keeps TOOLS_CALLED + RETRIEVAL_CONTEXT, but strips
    # duplicate 'Tool ... returned:' strings from retrieval_context during geval
    self.assertIn(
        FakeLLMTestCaseParams.TOOLS_CALLED,
        geval_tool_inst.kwargs['evaluation_params'],
    )
    self.assertIn(
        FakeLLMTestCaseParams.RETRIEVAL_CONTEXT,
        geval_tool_inst.kwargs['evaluation_params'],
    )
    self.assertEqual(
        geval_tool_inst.measured_retrieval_context,
        [
            'Prior turn 1: User: Who owns the sales domain? | Assistant: Sales'
            ' PO is Alice.'
        ],
    )
    # contextual_* filters out 'Prior turn 1: ...' from retrieval_context
    self.assertEqual(
        ctx_rel_inst.measured_retrieval_context,
        [
            'Tool lookup_owner returned: {"owner": "Alice"}',
            'Tool lookup_table returned: {"table": "proj.sales.orders"}',
        ],
    )
    # Subsequent faithfulness metric still sees full retrieval_context
    self.assertEqual(
        faith_tool_inst.measured_retrieval_context,
        parsed_tool_mt['retrieval_context'],
    )

    # 3. Multi-turn AgentData case (with tools and tool-less):
    agent_data_mt = {
        'agents': {'root_agent': {'instruction': 'You are RootAgent.'}},
        'turns': [
            {
                'turn_index': 0,
                'events': [
                    {
                        'author': 'user',
                        'content': {
                            'role': 'user',
                            'parts': [{'text': 'Find order 42.'}],
                        },
                    },
                    {
                        'author': 'root_agent',
                        'content': {
                            'role': 'model',
                            'parts': [{
                                'function_call': {
                                    'name': 'get_order',
                                    'args': {'id': 42},
                                }
                            }],
                        },
                    },
                    {
                        'author': 'root_agent',
                        'content': {
                            'role': 'user',
                            'parts': [{
                                'function_response': {
                                    'name': 'get_order',
                                    'response': {'status': 'shipped'},
                                }
                            }],
                        },
                    },
                    {
                        'author': 'root_agent',
                        'content': {
                            'role': 'model',
                            'parts': [{'text': 'Order 42 is shipped.'}],
                        },
                    },
                ],
            },
            {
                'turn_index': 1,
                'events': [
                    {
                        'author': 'user',
                        'content': {
                            'role': 'user',
                            'parts': [{'text': 'Summarize its status.'}],
                        },
                    },
                    {
                        'author': 'root_agent',
                        'content': {
                            'role': 'model',
                            'parts': [{'text': 'Order 42 has shipped.'}],
                        },
                    },
                ],
            },
        ],
    }
    parsed_ad = json.loads(parse_trace(json.dumps(agent_data_mt)))
    self.assertEqual(
        parsed_ad['retrieval_context'],
        [
            'Tool get_order returned: {"status": "shipped"}',
            (
                'Prior turn 1: User: Find order 42. | Assistant: Order 42 is'
                ' shipped.'
            ),
        ],
    )
    self.assertEqual(
        parsed_ad['turns'][0]['retrieval_context'],
        ['Tool get_order returned: {"status": "shipped"}'],
    )
    self.assertEqual(parsed_ad['turns'][1]['retrieval_context'], [])

    # Tool-less 2-turn AgentData preserves context=[primary_instr]
    agent_data_toolless = {
        'agents': {'root_agent': {'instruction': 'You are RootAgent.'}},
        'turns': [
            {
                'turn_index': 0,
                'events': [
                    {
                        'author': 'user',
                        'content': {'role': 'user', 'parts': [{'text': 'Hi'}]},
                    },
                    {
                        'author': 'root_agent',
                        'content': {
                            'role': 'model',
                            'parts': [{'text': 'Hello!'}],
                        },
                    },
                ],
            },
            {
                'turn_index': 1,
                'events': [
                    {
                        'author': 'user',
                        'content': {
                            'role': 'user',
                            'parts': [{'text': 'Who are you?'}],
                        },
                    },
                    {
                        'author': 'root_agent',
                        'content': {
                            'role': 'model',
                            'parts': [{'text': 'I am RootAgent.'}],
                        },
                    },
                ],
            },
        ],
    }
    parsed_ad_tl = json.loads(parse_trace(json.dumps(agent_data_toolless)))
    self.assertEqual(parsed_ad_tl['context'], ['You are RootAgent.'])
    self.assertEqual(
        parsed_ad_tl['retrieval_context'],
        ['Prior turn 1: User: Hi | Assistant: Hello!'],
    )

    # 4. Single-turn no-op check: 1-turn trace has no 'Prior turn' entries
    single_turn_events = multi_turn_events[:3]
    parsed_st = json.loads(parse_trace(json.dumps(single_turn_events)))
    self.assertEqual(parsed_st['retrieval_context'], [])
    self.assertEqual(parsed_st['context'], ['You are SpaceAgent.'])

  def test_root_agent_filtering_excludes_agent_tool_sub_invocations(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    evaluate_test_case = ns['evaluate_test_case']
    adk_multi_events = [
        {
            'timestamp': '2026-02-01T10:00:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Who owns the sales domain?'},
        },
        {
            'timestamp': '2026-02-01T10:00:01.5Z',
            'event_type': 'TOOL_STARTING',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {
                'tool_name': 'search_documentation_agent',
                'tool_input': {'request': 'Find sales domain owner'},
            },
        },
        {
            'timestamp': '2026-02-01T10:00:02Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'search_documentation_agent',
            'invocation_id': 'sub-inv-1',
            'attributes': {'root_agent_name': 'search_documentation_agent'},
            'content': {'text_summary': 'internal subagent rag query'},
        },
        {
            'timestamp': '2026-02-01T10:00:02.1Z',
            'event_type': 'TOOL_STARTING',
            'agent': 'search_documentation_agent',
            'invocation_id': 'sub-inv-1',
            'attributes': {'root_agent_name': 'search_documentation_agent'},
            'content': {
                'tool_name': 'internal_rag_search',
                'tool_input': {'q': 'sales'},
            },
        },
        {
            'timestamp': '2026-02-01T10:00:02.2Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'search_documentation_agent',
            'invocation_id': 'sub-inv-1',
            'attributes': {'root_agent_name': 'search_documentation_agent'},
            'content': {
                'tool_name': 'internal_rag_search',
                'tool_output': {'doc': 'Alice'},
            },
        },
        {
            'timestamp': '2026-02-01T10:00:02.3Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': None,
            'invocation_id': 'sub-inv-1',
            'attributes': {'root_agent_name': 'search_documentation_agent'},
            'content': {},
        },
        {
            'timestamp': '2026-02-01T10:00:02.5Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {
                'tool_name': 'search_documentation_agent',
                'tool_output': {'owner': 'Alice'},
            },
        },
        {
            'timestamp': '2026-02-01T10:00:03Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Sales PO is Alice.'},
        },
        {
            'timestamp': '2026-02-01T10:01:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'And what table should I query?'},
        },
        {
            'timestamp': '2026-02-01T10:01:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'Query proj.sales.orders (PO: Alice).'},
        },
    ]
    # 1. Baseline without root_agent includes sub-inv-1 and internal_rag_search
    parsed_no_filter = json.loads(
        parse_trace(json.dumps({'events': adk_multi_events}))
    )
    self.assertEqual(len(parsed_no_filter['turns']), 3)
    self.assertEqual(
        [t['name'] for t in parsed_no_filter['tools_called']],
        ['search_documentation_agent', 'internal_rag_search'],
    )

    # 2. With root_agent='space_agent', sub-inv-1 (including its agent=None
    #    callback and internal_rag_search tool calls) is excluded
    parsed_adk = json.loads(
        parse_trace(
            json.dumps({
                'events': adk_multi_events,
                'root_agent': 'space_agent',
            })
        )
    )
    self.assertEqual(parsed_adk['agent_id'], 'space_agent')
    self.assertEqual(len(parsed_adk['turns']), 2)
    self.assertEqual(
        [t['name'] for t in parsed_adk['tools_called']],
        ['search_documentation_agent'],
    )
    self.assertEqual(
        parsed_adk['turns'][0]['actual_output'], 'Sales PO is Alice.'
    )
    self.assertEqual(
        parsed_adk['actual_output'], 'Query proj.sales.orders (PO: Alice).'
    )

    # 3. Sub-agent-only group (even with an agent=None callback row) returns
    #    None from parse_trace so evaluate_test_case safely returns []
    sub_only_events = [
        ev for ev in adk_multi_events if ev['invocation_id'] == 'sub-inv-1'
    ]
    self.assertIsNone(
        parse_trace(
            json.dumps({
                'events': sub_only_events,
                'root_agent': 'space_agent',
            })
        )
    )
    self.assertEqual(evaluate_test_case(None, ['answer_relevancy']), [])

  def test_root_agent_filtering_preserves_transfer_to_agent_and_followups(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    transfer_events = [
        {
            'timestamp': '2026-02-01T10:02:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-transfer-1',
            'content': {'text_summary': 'Run billing SQL'},
        },
        {
            'timestamp': '2026-02-01T10:02:01Z',
            'event_type': 'TOOL_STARTING',
            'agent': 'space_agent',
            'invocation_id': 'inv-transfer-1',
            'content': {
                'tool_name': 'transfer_to_agent',
                'tool_input': {'agent_name': 'billing_sub_agent'},
            },
        },
        {
            'timestamp': '2026-02-01T10:02:02Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'space_agent',
            'invocation_id': 'inv-transfer-1',
            'content': {'tool_name': 'transfer_to_agent', 'tool_output': 'ok'},
        },
        {
            'timestamp': '2026-02-01T10:02:03Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-1',
            'content': {
                'tool_name': 'execute_sql',
                'tool_output': {'rows': 5},
            },
        },
        {
            'timestamp': '2026-02-01T10:02:04Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-1',
            'content': {'text_summary': 'Found 5 billing rows.'},
        },
        # Follow-up Turn 2 (all rows logged under billing_sub_agent with
        # attributes.root_agent_name stamped by TraceManager.init_trace)
        {
            'timestamp': '2026-02-01T10:03:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-2',
            'attributes': {'root_agent_name': 'space_agent'},
            'content': {'text_summary': 'Filter to Q1 only'},
        },
        {
            'timestamp': '2026-02-01T10:03:01Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-2',
            'attributes': {'root_agent_name': 'space_agent'},
            'content': {
                'tool_name': 'execute_q1_sql',
                'tool_output': {'rows': 2},
            },
        },
        {
            'timestamp': '2026-02-01T10:03:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-2',
            'attributes': {'root_agent_name': 'space_agent'},
            'content': {'text_summary': 'Found 2 Q1 billing rows.'},
        },
        # Follow-up Turn 3 (USER_MESSAGE_RECEIVED logged under root_agent while
        # TOOL_* and AGENT_RESPONSE run under billing_sub_agent)
        {
            'timestamp': '2026-02-01T10:04:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-transfer-3',
            'content': {'text_summary': 'Export those 2 rows'},
        },
        {
            'timestamp': '2026-02-01T10:04:01Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-3',
            'content': {
                'tool_name': 'export_csv',
                'tool_output': {'status': 'exported'},
            },
        },
        {
            'timestamp': '2026-02-01T10:04:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'billing_sub_agent',
            'invocation_id': 'inv-transfer-3',
            'content': {'text_summary': 'Exported 2 rows to CSV.'},
        },
    ]
    parsed_transfer = json.loads(
        parse_trace(
            json.dumps({
                'events': transfer_events,
                'root_agent': 'space_agent',
            })
        )
    )
    self.assertEqual(len(parsed_transfer['turns']), 3)
    self.assertEqual(
        parsed_transfer['turns'][0]['actual_output'], 'Found 5 billing rows.'
    )
    self.assertEqual(
        parsed_transfer['turns'][1]['actual_output'], 'Found 2 Q1 billing rows.'
    )
    self.assertEqual(
        parsed_transfer['actual_output'], 'Exported 2 rows to CSV.'
    )
    self.assertEqual(
        [t['name'] for t in parsed_transfer['tools_called']],
        ['execute_sql', 'execute_q1_sql', 'export_csv'],
    )
    # Single-turn GROUP BY session_id, invocation_id on inv-transfer-2 alone
    # succeeds via attributes.root_agent_name, and returns None if stripped
    inv_transfer_2_only = [
        ev for ev in transfer_events if ev['invocation_id'] == 'inv-transfer-2'
    ]
    parsed_t2_alone = json.loads(
        parse_trace(
            json.dumps({
                'events': inv_transfer_2_only,
                'root_agent': 'space_agent',
            })
        )
    )
    self.assertEqual(
        parsed_t2_alone['actual_output'], 'Found 2 Q1 billing rows.'
    )
    inv_transfer_2_no_attr = [
        {k: v for k, v in ev.items() if k != 'attributes'}
        for ev in inv_transfer_2_only
    ]
    self.assertIsNone(
        parse_trace(
            json.dumps({
                'events': inv_transfer_2_no_attr,
                'root_agent': 'space_agent',
            })
        )
    )

  def test_root_agent_filtering_preserves_programmatic_agent_transfer(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    agent_transfer_events = [
        {
            'timestamp': '2026-02-01T10:05:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-at-1',
            'content': {'text_summary': 'Check refund status'},
        },
        {
            'timestamp': '2026-02-01T10:05:01Z',
            'event_type': 'AGENT_TRANSFER',
            'agent': 'space_agent',
            'invocation_id': 'inv-at-1',
            'content': {
                'from_agent': 'space_agent',
                'to_agent': 'refund_sub_agent',
            },
        },
        {
            'timestamp': '2026-02-01T10:05:02Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'refund_sub_agent',
            'invocation_id': 'inv-at-1',
            'content': {
                'tool_name': 'lookup_refund',
                'tool_output': {'status': 'approved'},
            },
        },
        {
            'timestamp': '2026-02-01T10:05:03Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'refund_sub_agent',
            'invocation_id': 'inv-at-1',
            'content': {'text_summary': 'Refund is approved.'},
        },
        {
            'timestamp': '2026-02-01T10:06:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'refund_sub_agent',
            'invocation_id': 'inv-at-2',
            'content': {'text_summary': 'When will it arrive?'},
        },
        {
            'timestamp': '2026-02-01T10:06:01Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'refund_sub_agent',
            'invocation_id': 'inv-at-2',
            'content': {'text_summary': 'In 3-5 business days.'},
        },
    ]
    parsed_at = json.loads(
        parse_trace(
            json.dumps({
                'events': agent_transfer_events,
                'root_agent': 'space_agent',
            })
        )
    )
    self.assertEqual(len(parsed_at['turns']), 2)
    self.assertEqual(
        parsed_at['turns'][0]['actual_output'], 'Refund is approved.'
    )
    self.assertEqual(parsed_at['actual_output'], 'In 3-5 business days.')
    self.assertEqual(
        [t['name'] for t in parsed_at['tools_called']], ['lookup_refund']
    )

  def test_root_agent_filtering_preserves_sequential_workflow_children(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    sequential_events = [
        {
            'timestamp': '2026-02-01T10:07:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'seq_pipeline_root',
            'invocation_id': 'inv-seq-1',
            'content': {'text_summary': 'Fetch and summarize report'},
        },
        {
            'timestamp': '2026-02-01T10:07:01Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'fetcher_child_agent',
            'invocation_id': 'inv-seq-1',
            'content': {
                'tool_name': 'fetch_report_data',
                'tool_output': {'items': 10},
            },
        },
        {
            'timestamp': '2026-02-01T10:07:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'summarizer_child_agent',
            'invocation_id': 'inv-seq-1',
            'content': {'text_summary': 'Summary of 10 report items.'},
        },
    ]
    parsed_seq = json.loads(
        parse_trace(
            json.dumps({
                'events': sequential_events,
                'root_agent': 'seq_pipeline_root',
            })
        )
    )
    self.assertEqual(parsed_seq['actual_output'], 'Summary of 10 report items.')
    self.assertEqual(
        [t['name'] for t in parsed_seq['tools_called']], ['fetch_report_data']
    )

  def test_root_agent_filtering_ignores_transfer_text_in_non_transfer_prompts(
      self,
  ):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    non_transfer_events = [
        {
            'timestamp': '2026-02-01T10:08:00Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-nt-1',
            'content': {
                'text_summary': 'Explain how transfer_to_agent works in ADK'
            },
        },
        {
            'timestamp': '2026-02-01T10:08:01Z',
            'event_type': 'LLM_REQUEST',
            'agent': 'space_agent',
            'invocation_id': 'inv-nt-1',
            'content': {
                'system_prompt': (
                    'You have other agents: billing_sub_agent. Call'
                    ' transfer_to_agent to hand off.'
                )
            },
        },
        {
            'timestamp': '2026-02-01T10:08:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-nt-1',
            'content': {'text_summary': 'transfer_to_agent delegates control.'},
        },
        {
            'timestamp': '2026-02-01T10:08:03Z',
            'event_type': 'LLM_REQUEST',
            'agent': 'billing_sub_agent',
            'invocation_id': 'sub-inv-nt-1',
            'content': {
                'system_prompt': 'Sub-agent mentioning transfer_to_agent.'
            },
        },
        {
            'timestamp': '2026-02-01T10:08:04Z',
            'event_type': 'TOOL_COMPLETED',
            'agent': 'billing_sub_agent',
            'invocation_id': 'sub-inv-nt-1',
            'content': {
                'tool_name': 'sub_agent_only_tool',
                'tool_output': 'ignored',
            },
        },
        {
            'timestamp': '2026-02-01T10:08:05Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'billing_sub_agent',
            'invocation_id': 'sub-inv-nt-1',
            'content': {'text_summary': 'Untransferred sub-agent output.'},
        },
    ]
    parsed_nt = json.loads(
        parse_trace(
            json.dumps({
                'events': non_transfer_events,
                'root_agent': 'space_agent',
            })
        )
    )
    self.assertEqual(len(parsed_nt['turns']), 1)
    self.assertEqual(
        parsed_nt['actual_output'], 'transfer_to_agent delegates control.'
    )
    self.assertEqual(parsed_nt['tools_called'], [])
    self.assertIsNone(
        parse_trace(
            json.dumps({
                'events': non_transfer_events[3:],
                'root_agent': 'space_agent',
            })
        )
    )

  def test_unwrap_output_schema_and_chatbot_role_stripping(self):
    ns = _load_udf_module()
    parse_trace = ns['parse_trace']
    adk_events = [
        {
            'timestamp': '2026-02-01T10:00:00Z',
            'event_type': 'AGENT_STARTING',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': (
                'You are Current Phoenix, the enterprise data assistant.\n1.'
                ' Strict Output Format: You MUST output your final response'
                ' strictly as a JSON object matching the AgentResponse schema.'
            ),
        },
        {
            'timestamp': '2026-02-01T10:00:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {'text_summary': 'Who owns the sales domain?'},
        },
        {
            'timestamp': '2026-02-01T10:00:03Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-1',
            'content': {
                'text_summary': json.dumps({
                    'response': 'Sales PO is Alice.',
                    'usefulness': 0.95,
                })
            },
        },
        {
            'timestamp': '2026-02-01T10:01:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {'text_summary': 'And who owns finance?'},
        },
        {
            'timestamp': '2026-02-01T10:01:03Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-2',
            'content': {
                'text_summary': json.dumps({
                    'response': 'Finance PO is Bob.',
                    'usefulness': 0.98,
                })
            },
        },
    ]
    parsed_adk = json.loads(parse_trace(json.dumps(adk_events)))
    self.assertEqual(parsed_adk['actual_output'], 'Finance PO is Bob.')
    self.assertEqual(
        [t['actual_output'] for t in parsed_adk['turns']],
        ['Sales PO is Alice.', 'Finance PO is Bob.'],
    )
    self.assertEqual(
        parsed_adk['retrieval_context'],
        [
            'Prior turn 1: User: Who owns the sales domain? | Assistant: Sales'
            ' PO is Alice.'
        ],
    )
    self.assertEqual(
        parsed_adk['chatbot_role'],
        'You are Current Phoenix, the enterprise data assistant.',
    )

    # Explicit output_response_field unwraps a custom field name (even on a
    # single-key JSON object without schema keywords in AGENT_STARTING) and
    # strips schema-mandate lines from a wrapper-supplied chatbot_role
    custom_events = [
        {
            'timestamp': '2026-02-01T10:00:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-c1',
            'content': {'text_summary': 'Summarize status'},
        },
        {
            'timestamp': '2026-02-01T10:00:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-c1',
            'content': {
                'text_summary': json.dumps({'markdown_answer': 'All green.'})
            },
        },
    ]
    parsed_custom = json.loads(
        parse_trace(
            json.dumps({
                'events': custom_events,
                'output_response_field': 'markdown_answer',
                'chatbot_role': (
                    'You are a status monitor.\nOutput strictly as a JSON'
                    ' object matching the AgentResponse schema.'
                ),
            })
        )
    )
    self.assertEqual(parsed_custom['actual_output'], 'All green.')
    self.assertEqual(parsed_custom['chatbot_role'], 'You are a status monitor.')

    # Single-line schema-only chatbot_role falls back to 'Helpful AI Assistant'
    # instead of restoring the unstripped JSON-schema mandate; also verify
    # AgentData (_parse_agent_data_dict) unwrapping
    agent_data_schema = {
        'agents': {
            'root_agent': {
                'instruction': (
                    'You MUST output your final response strictly as a JSON'
                    ' object matching the AgentResponse schema.'
                )
            }
        },
        'turns': [{
            'turn_index': 0,
            'events': [
                {
                    'author': 'user',
                    'content': {'role': 'user', 'parts': [{'text': 'Status?'}]},
                },
                {
                    'author': 'root_agent',
                    'content': {
                        'role': 'model',
                        'parts': [{
                            'text': json.dumps(
                                {'response': 'Healthy.', 'score': 1.0}
                            )
                        }],
                    },
                },
            ],
        }],
    }
    parsed_ad_schema = json.loads(parse_trace(json.dumps(agent_data_schema)))
    self.assertEqual(parsed_ad_schema['actual_output'], 'Healthy.')
    self.assertEqual(parsed_ad_schema['chatbot_role'], 'Helpful AI Assistant')

    # Without output_response_field or ADK output_schema instruction, a JSON
    # response and AGENT_STARTING instruction mentioning 'JSON schema' are
    # preserved untouched
    caller_role = 'You help users query tables and explain their JSON schema.'
    plain_json_events = [
        {
            'timestamp': '2026-02-01T10:00:00Z',
            'event_type': 'AGENT_STARTING',
            'agent': 'space_agent',
            'invocation_id': 'inv-p1',
            'content': caller_role,
        },
        {
            'timestamp': '2026-02-01T10:00:01Z',
            'event_type': 'USER_MESSAGE_RECEIVED',
            'agent': 'space_agent',
            'invocation_id': 'inv-p1',
            'content': {'text_summary': 'Return a JSON payload'},
        },
        {
            'timestamp': '2026-02-01T10:00:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-p1',
            'content': {
                'text_summary': json.dumps(
                    {'response': 'raw payload', 'code': 200}
                )
            },
        },
    ]
    parsed_plain = json.loads(parse_trace(json.dumps(plain_json_events)))
    self.assertEqual(
        json.loads(parsed_plain['actual_output']),
        {'response': 'raw payload', 'code': 200},
    )
    self.assertEqual(parsed_plain['chatbot_role'], caller_role)

    empty_resp_events = [
        *plain_json_events[:2],
        {
            'timestamp': '2026-02-01T10:00:02Z',
            'event_type': 'AGENT_RESPONSE',
            'agent': 'space_agent',
            'invocation_id': 'inv-p1',
            'content': {
                'text_summary': json.dumps(
                    {'response': '   ', 'error': 'timeout'}
                )
            },
        },
    ]
    parsed_empty_resp = json.loads(
        parse_trace(
            json.dumps({
                'events': empty_resp_events,
                'output_response_field': 'response',
                'chatbot_role': caller_role,
            })
        )
    )
    self.assertEqual(
        json.loads(parsed_empty_resp['actual_output']),
        {'response': '   ', 'error': 'timeout'},
    )
    self.assertEqual(parsed_empty_resp['chatbot_role'], caller_role)

  def test_install_sh_help_and_end_to_end_with_fake_clis(self):
    install_sh = self.script_dir / 'install.sh'

    # 1. Test --help
    help_proc = subprocess.run(
        ['bash', str(install_sh), '--help'],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(help_proc.returncode, 0)
    self.assertIn('Usage:', help_proc.stdout)

    # 2. Test end-to-end execution with fake gcloud and bq in a temp bin dir
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      captured_sql_file = tmp_path / 'captured_query.sql'
      cli_log_file = tmp_path / 'cli_calls.log'

      fake_gcloud = tmp_path / 'gcloud'
      fake_gcloud.write_text(
          f"""#!/usr/bin/env bash
echo "gcloud $*" >> "{cli_log_file}"
if [[ "$1" == "config" && "$2" == "get-value" ]]; then
  echo "(unset)"
  exit 0
fi
exit 0
""",
          encoding='utf-8',
      )
      fake_gcloud.chmod(0o755)

      fake_bq = tmp_path / 'bq'
      fake_bq.write_text(
          f"""#!/usr/bin/env bash
echo "bq $*" >> "{cli_log_file}"
if [[ " $* " == *" --format=json "* && " $* " == *" --dataset "* ]]; then
  if [[ "${{FAKE_EXISTING_DS_LOC:-}}" != "" ]]; then
    echo "{{\\"location\\": \\"${{FAKE_EXISTING_DS_LOC}}\\"}}"
    exit 0
  fi
  exit 1
fi
if [[ " $* " == *" --format=json "* ]]; then
  echo '{{"cloudResource": {{"serviceAccountId": "bqcx-111222333-test@gcp-sa-bigquery-condel.iam.gserviceaccount.com"}}}}'
  exit 0
fi
if [[ " $* " == *" show "* ]]; then
  exit 1
fi
if [[ " $* " == *" query "* ]]; then
  cat > "{captured_sql_file}"
  exit 0
fi
exit 0
""",
          encoding='utf-8',
      )
      fake_bq.chmod(0o755)

      env = os.environ.copy()
      env['PATH'] = f"{tmp_path}:{env.get('PATH', '')}"
      env.pop('PROJECT_ID', None)

      # Running without PROJECT_ID when gcloud returns (unset) fails cleanly
      fail_proc = subprocess.run(
          ['bash', str(install_sh)],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(fail_proc.returncode, 0)
      self.assertIn('Error: PROJECT_ID is not set', fail_proc.stderr)

      # Running with explicit args succeeds and substitutes all placeholders
      ok_proc = subprocess.run(
          [
              'bash',
              str(install_sh),
              'customer-gcp-proj',
              'custom_dataset',
              'us-central1',
              'custom_ge_conn',
          ],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(
          ok_proc.returncode,
          0,
          f'install.sh failed:\nstdout: {ok_proc.stdout}\nstderr:'
          f' {ok_proc.stderr}',
      )
      cli_calls = cli_log_file.read_text(encoding='utf-8')
      self.assertIn(
          '--member=serviceAccount:bqcx-111222333-test@gcp-sa-bigquery-condel.iam.gserviceaccount.com',
          cli_calls,
      )
      rendered_sql = captured_sql_file.read_text(encoding='utf-8')
      self.assertIn(
          'CREATE OR REPLACE FUNCTION'
          ' `customer-gcp-proj.custom_dataset.deepeval_parse_trace`',
          rendered_sql,
      )
      self.assertIn(
          'CREATE OR REPLACE FUNCTION'
          ' `customer-gcp-proj.custom_dataset.deepeval_evaluate`',
          rendered_sql,
      )
      self.assertIn(
          'WITH CONNECTION `customer-gcp-proj.us-central1.custom_ge_conn`',
          rendered_sql,
      )
      for placeholder in (
          'YOUR_PROJECT_ID',
          'YOUR_DATASET_ID',
          'YOUR_LOCATION',
          'YOUR_CONNECTION_ID',
          'YOUR_MODEL_ID',
      ):
        self.assertNotIn(placeholder, rendered_sql)
      self.assertIn('"gemini-3.8-flash"', rendered_sql)
      py_bodies = _extract_udf_python_bodies(rendered_sql)
      self.assertEqual(len(py_bodies), 2)
      for py_body in py_bodies:
        self.assertLessEqual(
            len(py_body.encode('utf-8')),
            _UDF_BODY_LIMIT_BYTES,
            'Python UDF definition body must be <= 32,768 bytes for BigQuery.',
        )

      # Running with explicit 5th arg MODEL_ID substitutes custom model
      model_proc = subprocess.run(
          [
              'bash',
              str(install_sh),
              'customer-gcp-proj',
              'custom_dataset',
              'us-central1',
              'custom_ge_conn',
              'gemini-2.5-pro',
          ],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(model_proc.returncode, 0)
      custom_rendered_sql = captured_sql_file.read_text(encoding='utf-8')
      self.assertIn('"gemini-2.5-pro"', custom_rendered_sql)
      custom_py_bodies = _extract_udf_python_bodies(custom_rendered_sql)
      self.assertEqual(len(custom_py_bodies), 2)
      for custom_py_body in custom_py_bodies:
        self.assertLessEqual(
            len(custom_py_body.encode('utf-8')),
            _UDF_BODY_LIMIT_BYTES,
            'Custom-model Python UDF definition body must be <= 32,768 bytes.',
        )

      # Invalid identifier with sed delimiter or whitespace is rejected upfront
      bad_id_proc = subprocess.run(
          [
              'bash',
              str(install_sh),
              'customer-gcp-proj',
              'bad|dataset',
              'us-central1',
              'custom_ge_conn',
          ],
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(bad_id_proc.returncode, 0)
      self.assertIn('Error: Invalid DATASET_ID', bad_id_proc.stderr)

      # Existing dataset with case-insensitive matching location (e.g. 'US' vs
      # 'us') succeeds on macOS Bash 3.2 (no Bash 4+ ${VAR,,} syntax)
      install_sh_text = install_sh.read_text(encoding='utf-8')
      self.assertIsNone(
          re.search(r'\$\{[^}]*(?:,,|\^\^)\}', install_sh_text),
          'install.sh must not use Bash 4+ case expansion (${var,,}) so it '
          'runs on macOS /bin/bash 3.2.',
      )
      same_loc_env = env.copy()
      same_loc_env['FAKE_EXISTING_DS_LOC'] = 'US'
      same_loc_proc = subprocess.run(
          [
              'bash',
              str(install_sh),
              'customer-gcp-proj',
              'custom_dataset',
              'us',
              'custom_ge_conn',
          ],
          env=same_loc_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(same_loc_proc.returncode, 0, same_loc_proc.stderr)

      # Existing dataset in a different location fails early with clear error
      mismatch_env = env.copy()
      mismatch_env['FAKE_EXISTING_DS_LOC'] = 'EU'
      mismatch_proc = subprocess.run(
          [
              'bash',
              str(install_sh),
              'customer-gcp-proj',
              'custom_dataset',
              'us-central1',
              'custom_ge_conn',
          ],
          env=mismatch_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(mismatch_proc.returncode, 0)
      self.assertIn(
          "already exists in location 'EU'",
          mismatch_proc.stderr,
      )


if __name__ == '__main__':
  unittest.main()
