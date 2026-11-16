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
import random
import re
import threading
import time
from typing import Any, Callable, Mapping, Optional

import google.auth
import google.auth.exceptions
from google.auth.transport import requests as google_auth_requests
import requests
import requests.adapters
import requests.exceptions

# Regional endpoint template and defaults.
# YOUR_GEAP_REGION and YOUR_PROJECT_ID are replaced during installation.
_DEFAULT_GEAP_REGION = "YOUR_GEAP_REGION"
_DEFAULT_PROJECT_ID = "YOUR_PROJECT_ID"
# Built from two parts so install.sh's global placeholder replace can't rewrite
# it. Matching _DEFAULT_PROJECT_ID means the install step never ran.
_UNFILLED_PROJECT_ID = "YOUR_" + "PROJECT_ID"
_GEAP_ENDPOINT_TEMPLATE = (
    "https://{region}-aiplatform.googleapis.com/v1beta1/"
    "projects/{project}/locations/{region}:evaluateInstances"
)
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Retry budget limits.
_MAX_RETRY_ATTEMPTS = 5
_MAX_CUMULATIVE_WAIT_SECONDS = 60.0
_INITIAL_BACKOFF_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 60.0
_HTTP_POOL_SIZE = 100
_MAX_RAW_ERROR_CHARS = 512

# HTTP status codes eligible for retry.
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

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
    408: "DEADLINE_EXCEEDED",
    409: "ALREADY_EXISTS",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    500: "INTERNAL",
    501: "UNIMPLEMENTED",
    502: "UNAVAILABLE",
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


# Module-level cached credentials across row evaluations.
_CACHED_CREDENTIALS: Any = None


def validate_inputs(
    trace_str: Optional[str],
    managed_metric: Optional[str],
    prompt_template: Optional[str],
) -> Optional[dict[str, Any]]:
  """Validates inputs locally before spending GEAP quota.

  Args:
    trace_str: Serialized JSON trace containing conversation turns and events.
    managed_metric: Name of a versioned GEAP managed metric.
    prompt_template: Custom prompt template string containing placeholders.

  Returns:
    The parsed trace, or None if trace_str is None.

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
    return None

  # Trace must be a valid JSON object with at least one turn.
  try:
    trace_dict = json.loads(trace_str)
    # Reject non-finite numbers (NaN, Inf) that are not compliant with standard JSON.
    json.dumps(trace_dict, allow_nan=False)
  except (json.JSONDecodeError, ValueError, RecursionError) as e:
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

  return trace_dict


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


_CREDENTIALS_LOCK = threading.Lock()


def get_credentials(
    scope: str = _CLOUD_PLATFORM_SCOPE,
) -> tuple[Optional[str], Optional[str]]:
  """Obtains an authorization token using google.auth.default.

  The project to call never comes from google-auth: inside a connection-backed
  UDF it can be the connection's runtime project. geap_evaluate always uses the
  install-time _DEFAULT_PROJECT_ID instead.

  Args:
    scope: OAuth scope for Google Cloud Platform.

  Returns:
    (token, status_error) where token is the Bearer token and status_error is
    non-None if authentication failed.
  """
  global _CACHED_CREDENTIALS
  with _CREDENTIALS_LOCK:
    try:
      if hasattr(google.auth, "default"):
        if _CACHED_CREDENTIALS is None:
          creds, _ = google.auth.default(scopes=[scope])
        else:
          creds = _CACHED_CREDENTIALS

        if hasattr(creds, "valid") and not creds.valid:
          if hasattr(creds, "refresh"):
            creds.refresh(google_auth_requests.Request())

        token = getattr(creds, "token", None)
        if not token:
          _CACHED_CREDENTIALS = None
          return (
              None,
              (
                  "UNAUTHENTICATED: Failed to obtain authentication token from"
                  " credentials."
              ),
          )
        _CACHED_CREDENTIALS = creds
        return (token, None)
      return (None, "UNAUTHENTICATED: google.auth library not available.")
    except (
        google.auth.exceptions.GoogleAuthError,
        requests.exceptions.RequestException,
    ) as e:
      _CACHED_CREDENTIALS = None
      return (None, f"UNAUTHENTICATED: {e}")


_SESSION_LOCK = threading.Lock()
# Module-level cached requests.Session for connection pooling across rows.
_HTTP_SESSION: Optional[requests.Session] = None


def _get_http_session() -> requests.Session:
  """Returns a cached module-level requests.Session for connection reuse."""
  global _HTTP_SESSION
  if _HTTP_SESSION is None:
    with _SESSION_LOCK:
      if _HTTP_SESSION is None:
        session = requests.Session()
        if hasattr(requests, "adapters") and hasattr(
            requests.adapters, "HTTPAdapter"
        ):
          adapter = requests.adapters.HTTPAdapter(
              pool_connections=_HTTP_POOL_SIZE, pool_maxsize=_HTTP_POOL_SIZE
          )
          session.mount("https://", adapter)
          session.mount("http://", adapter)
        _HTTP_SESSION = session
  return _HTTP_SESSION


def execute_http_request(
    url: str,
    payload: Mapping[str, Any],
    token: str,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
) -> Any:
  """Executes an HTTP POST request to the GEAP endpoint.

  Args:
    url: Full target URL for the GEAP evaluateInstances endpoint.
    payload: JSON-serializable request payload dictionary.
    token: Bearer authorization token.
    timeout: Request timeout duration in seconds.
    session: Optional requests.Session instance for connection pooling.

  Returns:
    The response object returned by the HTTP session's ``post`` call.

  Raises:
    requests.exceptions.Timeout: If the request exceeds the timeout deadline.
    requests.exceptions.RequestException: If the underlying HTTP transport
    fails.
  """
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
  }
  http_client = session or _get_http_session()
  try:
    response = http_client.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response
  except requests.exceptions.Timeout as e:
    raise requests.exceptions.Timeout(
        f"Request timed out after {timeout} seconds: {e}"
    ) from e


def _extract_response_error(response: Any, default_status: int) -> str:
  """Extracts error message from response JSON, fallback text, or status code.

  Args:
    response: HTTP response object, or None if the request failed at transport.
    default_status: Fallback HTTP status code if no message is present.

  Returns:
    Extracted error message string or formatted HTTP status code.
  """
  if response is None:
    return _resolve_error_tag(
        fallback_http_status=default_status,
        default_tag=f"HTTP {default_status}",
    )
  try:
    data = (
        response.json() if callable(getattr(response, "json", None)) else None
    )
    if isinstance(data, Mapping):
      raw_error = data.get("error")
      if isinstance(raw_error, Mapping):
        code = raw_error.get("code")
        status_str = raw_error.get("status")
        msg = raw_error.get("message")
        tag = _resolve_error_tag(
            status_str=status_str,
            code=code if code != 0 else None,
            fallback_http_status=default_status,
            default_tag=f"HTTP {default_status}",
        )
        if msg:
          return f"{tag}: {msg}".strip(": ")
        return tag
  except (ValueError, TypeError, AttributeError):
    pass
  tag = _resolve_error_tag(
      fallback_http_status=default_status,
      default_tag=f"HTTP {default_status}",
  )
  raw_text = (getattr(response, "text", "") or "").strip()
  if raw_text:
    return f"{tag}: {raw_text[:_MAX_RAW_ERROR_CHARS]}"
  return tag


def evaluate_with_retries(
    send_fn: Callable[[], Any],
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    random_fn: Callable[[float, float], float] = random.uniform,
) -> tuple[Optional[requests.Response], int, Optional[str]]:
  """Executes request within the retry budget using backoff and jitter.

  Args:
    send_fn: Callable executing a single HTTP request attempt.
    sleep_fn: Function to sleep for a specified duration in seconds.
    clock_fn: Function returning current monotonic or epoch time in seconds.
    random_fn: Function returning uniform random float in [a, b] for jitter.

  Returns:
    A 3-tuple (response, http_status, error_status):
      - response: Successful response object, or None if all attempts failed.
      - http_status: HTTP status code of the final attempt (e.g. 200, 429).
      - error_status: Human-readable error message, or None on success.
  """
  attempts = 0
  cumulative_wait_seconds = 0.0
  start_time = clock_fn()
  last_status_code = 504
  last_error_msg: Optional[str] = (
      "DEADLINE_EXCEEDED: Exceeded retry wait budget"
  )
  while attempts < _MAX_RETRY_ATTEMPTS:
    if clock_fn() - start_time >= _MAX_CUMULATIVE_WAIT_SECONDS:
      return (None, last_status_code, last_error_msg)
    try:
      response = send_fn()
      status_code = getattr(response, "status_code", 500)
      client_error = None
    except (
        requests.exceptions.InvalidURL,
        requests.exceptions.InvalidHeader,
        requests.exceptions.InvalidJSONError,
    ) as e:
      tag = _resolve_error_tag(fallback_http_status=400)
      return (None, 400, f"{tag}: {e}")
    except requests.exceptions.Timeout as e:
      # Map client timeouts directly to HTTP 408 for standard backoff.
      tag = _resolve_error_tag(fallback_http_status=408)
      response, status_code, client_error = None, 408, f"{tag}: {e}"
    except requests.exceptions.ConnectionError as e:
      # Map dropped connections directly to HTTP 503.
      tag = _resolve_error_tag(fallback_http_status=503)
      response, status_code, client_error = None, 503, f"{tag}: {e}"
    except requests.exceptions.RequestException as e:
      # Map generic transport exceptions directly to HTTP 500.
      tag = _resolve_error_tag(fallback_http_status=500)
      response, status_code, client_error = None, 500, f"{tag}: {e}"

    # Success: return HTTP 200 response immediately.
    if status_code == 200:
      return (response, 200, None)

    error_msg = client_error or _extract_response_error(response, status_code)
    last_status_code = status_code
    last_error_msg = error_msg

    # Non-retryable error (e.g. 400 Bad Request, 404 Not Found): abort
    # immediately.
    if status_code not in _RETRYABLE_HTTP_STATUS_CODES:
      return (None, status_code, error_msg)

    attempts += 1
    if attempts >= _MAX_RETRY_ATTEMPTS:
      return (None, status_code, error_msg)

    # Respect explicit Retry-After backoff duration if requested by GEAP.
    retry_after_str = (getattr(response, "headers", None) or {}).get(
        "Retry-After"
    )
    try:
      retry_after = float(retry_after_str) if retry_after_str else 0.0
    except ValueError:
      retry_after = 0.0

    backoff = _INITIAL_BACKOFF_SECONDS * (2 ** (attempts - 1)) + random_fn(
        0, _INITIAL_BACKOFF_SECONDS
    )
    wait_seconds = max(backoff, retry_after)
    cumulative_wait_seconds += wait_seconds

    # Abort without sleeping if the backoff wait duration would exceed the
    # cumulative retry wait budget or the elapsed wall-clock deadline.
    elapsed = clock_fn() - start_time
    if (
        elapsed + wait_seconds >= _MAX_CUMULATIVE_WAIT_SECONDS
        or cumulative_wait_seconds > _MAX_CUMULATIVE_WAIT_SECONDS
    ):
      return (None, status_code, error_msg)

    sleep_fn(wait_seconds)
  return (None, last_status_code, last_error_msg)


def geap_evaluate(
    trace_str: Optional[str],
    managed_metric: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> dict[str, Any]:
  """BigQuery Python UDF entry point.

  Validates inputs, obtains authorization, constructs the request payload,
  and dispatches the evaluation HTTP call within the retry budget. Under
  the BigQuery Python UDF execution model, this entry point catches all
  exceptions and maps them into the structured return dict so row-level
  failures never terminate the enclosing query.

  Args:
    trace_str: Serialized JSON conversation trace containing turns and events.
    managed_metric: Optional name of a supported GEAP managed metric.
    prompt_template: Optional custom prompt template containing placeholders.

  Returns:
    A dictionary matching the BigQuery evaluation struct:
      - score: Optional float rating score, or None on failure or error.
      - explanation: Optional textual rationale, or None on failure or error.
      - rubric_verdicts: List of individual rubric evaluation dictionaries.
      - status: Non-None error status string if validation or execution failed.
  """
  try:
    trace_dict = validate_inputs(trace_str, managed_metric, prompt_template)
  except ValueError as e:
    return _error_result(f"INVALID_ARGUMENT: {e}")

  if trace_dict is None:
    return _error_result("INVALID_ARGUMENT: trace cannot be None.")

  if _DEFAULT_PROJECT_ID == _UNFILLED_PROJECT_ID:
    return _error_result(
        "INTERNAL: install project ID was never filled in; rerun install.sh."
    )

  try:
    token, status = get_credentials()
    if status or not token:
      return _error_result(
          status or "UNAUTHENTICATED: missing credentials token"
      )
    project_id = _DEFAULT_PROJECT_ID
    region = _DEFAULT_GEAP_REGION
    url = _GEAP_ENDPOINT_TEMPLATE.format(project=project_id, region=region)
    payload = build_evaluation_request(
        project_id=project_id,
        region=region,
        trace_dict=trace_dict,
        managed_metric=managed_metric,
        prompt_template=prompt_template,
    )
    resp, http_status, error_status = evaluate_with_retries(
        lambda: execute_http_request(url, payload, token)
    )
    if error_status or resp is None:
      return _error_result(
          error_status or "INTERNAL: empty response from service"
      )
    resp_data = resp.json()
    if not isinstance(resp_data, Mapping):
      resp_data = {}
    return parse_evaluation_response(resp_data, http_status)
  except Exception as e:  # pylint: disable=broad-except
    # Top-level boundary safety: BigQuery row evaluation must return a struct
    # with an error status rather than raising and aborting the entire query.
    return _error_result(f"INTERNAL: {e}")


# Alias for BigQuery routine mapping.
geap_evaluate_impl = geap_evaluate
