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

"""Evaluation logic for BigQuery GEAP agent evaluation Python UDF."""

import json
import re
from typing import Any, Mapping, Optional

# Supported Vertex AI evaluation managed metrics.
VALID_MANAGED_METRICS = frozenset({
    "tool_use_quality_v2",
    "final_response_quality_v2",
    "hallucination_v2",
    "safety_v2",
    "safety_v3",
    "tool_use_quality_v1",
    "final_response_quality_v1",
    "hallucination_v1",
    "safety_v1",
    "multi_turn_task_success_v1",
    "multi_turn_tool_use_quality_v1",
    "multi_turn_trajectory_quality_v1",
})

# Metrics strictly limited to single-turn traces.
SINGLE_TURN_MANAGED_METRICS = frozenset({
    "tool_use_quality_v2",
    "final_response_quality_v2",
    "hallucination_v2",
    "safety_v2",
    "safety_v3",
    "tool_use_quality_v1",
    "final_response_quality_v1",
    "hallucination_v1",
    "safety_v1",
})

# Matches prompt template variable root identifiers (e.g. {agent_data},
# { agent_data }, {agent_data.turns}, {agent_data | trim}).
_TEMPLATE_VAR_PATTERN = re.compile(r"\{\s*([_a-zA-Z]\w*)[^}]*\}")

# Mapping of standard google.rpc.Code integer values to canonical status strings.
_CANONICAL_RPC_STATUS_BY_CODE = {
    1: "CANCELLED",
    2: "UNKNOWN",
    3: "INVALID_ARGUMENT",
    4: "DEADLINE_EXCEEDED",
    5: "NOT_FOUND",
    6: "ALREADY_EXISTS",
    7: "PERMISSION_DENIED",
    8: "RESOURCE_EXHAUSTED",
    9: "FAILED_PRECONDITION",
    10: "ABORTED",
    11: "OUT_OF_RANGE",
    12: "UNIMPLEMENTED",
    13: "INTERNAL",
    14: "UNAVAILABLE",
    15: "DATA_LOSS",
    16: "UNAUTHENTICATED",
}

# Mapping of standard HTTP status codes to canonical error names.
_HTTP_STATUS_TO_CANONICAL_NAME = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    409: "ALREADY_EXISTS",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    500: "INTERNAL",
    501: "UNIMPLEMENTED",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}


def _resolve_error_tag(
    status_str: Optional[str] = None,
    code: Optional[int] = None,
    fallback_http_status: Optional[int] = None,
    default_tag: str = "UNKNOWN",
) -> str:
  """Resolves a canonical error status string from status name, code, or HTTP status.

  Args:
    status_str: Optional string status name (e.g. 'INVALID_ARGUMENT').
    code: Optional integer error code from google.rpc.Code or HTTP status.
    fallback_http_status: Optional fallback HTTP status code if code is absent.
    default_tag: Fallback tag to return if no status can be resolved.

  Returns:
    Canonical uppercase status string (e.g. 'RESOURCE_EXHAUSTED', 'INTERNAL').
    An unrecognized code resolves to 'UNKNOWN' so every status keeps a
    google.rpc.Code name as its prefix.
  """
  if status_str:
    return status_str
  if code is not None:
    if code in _CANONICAL_RPC_STATUS_BY_CODE:
      return _CANONICAL_RPC_STATUS_BY_CODE[code]
    if code in _HTTP_STATUS_TO_CANONICAL_NAME:
      return _HTTP_STATUS_TO_CANONICAL_NAME[code]
    return "UNKNOWN"
  if (
      fallback_http_status is not None
      and fallback_http_status in _HTTP_STATUS_TO_CANONICAL_NAME
  ):
    return _HTTP_STATUS_TO_CANONICAL_NAME[fallback_http_status]
  return default_tag


def validate_inputs(
    trace_str: Optional[str],
    managed_metric: Optional[str],
    prompt_template: Optional[str],
) -> None:
  """Validates inputs locally before spending GEAP quota.

  Args:
    trace_str: Serialized JSON trace containing conversation turns and events.
    managed_metric: Name of a versioned GEAP managed metric.
    prompt_template: Custom prompt template string containing placeholders.

  Raises:
    ValueError: If any input fails local contract validation.
  """
  # Exactly one metric specification must be provided.
  has_managed = managed_metric is not None
  has_prompt = prompt_template is not None
  if has_managed == has_prompt:
    raise ValueError(
        "Exactly one of managed_metric or prompt_template must be provided."
    )

  if managed_metric is not None and managed_metric not in VALID_MANAGED_METRICS:
    raise ValueError(
        f"Managed metric {managed_metric} is not supported. "
        f"Supported metrics: {VALID_MANAGED_METRICS}"
    )

  # Custom prompt templates must contain at least one placeholder
  # (e.g. {agent_data}, {events}) to be populated with trace content by GEAP.
  # {question_block} is the rubric injection variable, not a trace placeholder.
  template_vars: set[str] = set()
  if prompt_template is not None:
    template_vars = set(_TEMPLATE_VAR_PATTERN.findall(prompt_template))
    if not (template_vars - {"question_block"}):
      raise ValueError(
          "Custom prompt_template must contain at least one placeholder"
          " enclosed in single braces, e.g. {agent_data}."
      )

  if trace_str is None:
    return

  # Trace must be a valid JSON object with at least one turn.
  try:
    trace_dict = json.loads(trace_str)
  except json.JSONDecodeError as e:
    raise ValueError(f"Failed to parse trace: {e}") from e

  if not isinstance(trace_dict, dict):
    raise ValueError("Trace must be a valid JSON object.")

  turns = trace_dict.get("turns")
  if not isinstance(turns, list) or not turns:
    raise ValueError("Trace must contain at least one turn.")

  if managed_metric in SINGLE_TURN_MANAGED_METRICS and len(turns) != 1:
    raise ValueError(
        f"Single-turn metric '{managed_metric}' requires a trace with"
        f" exactly 1 turn, but found {len(turns)} turns."
    )

  # Multi-turn traces cannot use single-turn placeholders
  # ({prompt}, {response}), as GEAP only populates them when evaluating
  # single-turn interactions.
  if len(turns) > 1 and {"prompt", "response"} & template_vars:
    raise ValueError(
        "Multi-turn metrics cannot be used with {prompt} or {response}"
        " placeholders."
    )


def build_evaluation_request(
    project_id: str,
    region: str,
    trace_dict: Mapping[str, Any],
    managed_metric: Optional[str],
    prompt_template: Optional[str],
) -> dict[str, Any]:
  """Builds an EvaluateInstancesRequest dictionary.

  Args:
    project_id: GCP project ID hosting the evaluation.
    region: Regional Vertex AI endpoint location (e.g. us-central1).
    trace_dict: Parsed conversation trace containing turns and events.
    managed_metric: Optional GEAP managed metric name.
    prompt_template: Optional custom prompt template string.

  Returns:
    Dictionary payload matching the EvaluateInstancesRequest proto schema.
  """
  request: dict[str, Any] = {
      "location": f"projects/{project_id}/locations/{region}",
      "instance": {
          "agent_eval_data": trace_dict,
      },
  }
  # GEAP evaluates only the first metric in the metrics array.
  if managed_metric is not None:
    request["metrics"] = [
        {
            "predefined_metric_spec": {
                "metric_spec_name": managed_metric,
            }
        }
    ]
  elif prompt_template is not None:
    request["metrics"] = [
        {
            "llm_based_metric_spec": {
                "metric_prompt_template": prompt_template,
            }
        }
    ]
  return request


def _error_result(status: str) -> dict[str, Any]:
  """Builds a BigQuery UDF error return struct with null score and explanation.

  Args:
    status: Canonical error message to record in the return struct.

  Returns:
    Dictionary matching BigQuery evaluation return struct contract with null
    score and explanation, empty rubric verdicts, and the error status.
  """
  return {
      "score": None,
      "explanation": None,
      "rubric_verdicts": [],
      "status": status,
  }


def _parse_rubric_verdicts(raw_verdicts: Any) -> list[dict[str, Any]]:
  """Parses EvaluateInstancesResponse rubricVerdicts into BigQuery structs.

  Args:
    raw_verdicts: Raw list of rubric verdict mappings from the GEAP response.

  Returns:
    List of parsed rubric verdict dictionaries matching BigQuery struct fields:
    rubric_id, rubric_content, type, importance, verdict, and reasoning.
  """
  if not isinstance(raw_verdicts, list):
    return []

  verdicts = []
  for item in raw_verdicts:
    if not isinstance(item, Mapping):
      continue
    rubric = item.get("evaluatedRubric") or {}
    if not isinstance(rubric, Mapping):
      rubric = {}

    importance = rubric.get("importance")
    if importance == "IMPORTANCE_UNSPECIFIED":
      importance = None

    content_obj = rubric.get("content")
    prop = (
        content_obj.get("property")
        if isinstance(content_obj, Mapping)
        else None
    )
    rubric_content = (
        prop.get("description") if isinstance(prop, Mapping) else None
    )

    # Proto3 JSON mapping omits boolean fields when False. Explicitly default
    # missing 'verdict' to False to avoid hiding failed rubrics as NULL.
    verdicts.append({
        "rubric_id": rubric.get("rubricId") or None,
        "rubric_content": rubric_content or None,
        "type": rubric.get("type") or None,
        "importance": importance or None,
        "verdict": bool(item.get("verdict", False)),
        "reasoning": item.get("reasoning") or None,
    })
  return verdicts


def parse_evaluation_response(
    response_data: Mapping[str, Any],
    http_status: int = 200,
) -> dict[str, Any]:
  """Parses EvaluateInstancesResponse into the BigQuery UDF return struct.

  Args:
    response_data: Decoded JSON response payload from the GEAP endpoint.
    http_status: HTTP status code received with the response.

  Returns:
    Dictionary containing score, explanation, rubric_verdicts, and status.
  """
  results = response_data.get("metricResults")
  if (
      not isinstance(results, list)
      or not results
      or not isinstance(results[0], Mapping)
  ):
    # Extract error from top-level API error response or fallback to
    # HTTP status.
    raw_error = response_data.get("error")
    error = raw_error if isinstance(raw_error, Mapping) else {}
    status_str = error.get("status")
    code = error.get("code")
    message = error.get("message")
    if status_str or code is not None or message:
      tag = _resolve_error_tag(
          status_str=status_str,
          code=code,
          fallback_http_status=http_status,
          default_tag="INTERNAL",
      )
      status = f"{tag}: {message}" if message else tag
    else:
      status = (
          f"INTERNAL: empty metric results in response (HTTP {http_status})"
      )
    return _error_result(status)

  first_result = results[0]

  # GEAP can report an error inside an HTTP 200 response. Suppress score and
  # explanation to None on error to avoid skewing averages with 0.0.
  raw_error = first_result.get("error")
  error = raw_error if isinstance(raw_error, Mapping) else {}
  if error and (error.get("code", 0) != 0 or error.get("message")):
    code = error.get("code")
    message = error.get("message")
    tag = _resolve_error_tag(
        status_str=error.get("status"),
        code=code if code != 0 else None,
    )
    return _error_result(f"{tag}: {message}" if message else tag)

  score = first_result.get("score")
  explanation = first_result.get("explanation")
  return {
      "score": float(score) if score is not None else None,
      "explanation": explanation or None,
      "rubric_verdicts": _parse_rubric_verdicts(
          first_result.get("rubricVerdicts")
      ),
      "status": None,
  }
