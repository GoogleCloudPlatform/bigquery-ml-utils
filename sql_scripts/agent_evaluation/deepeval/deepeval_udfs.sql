-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- =============================================================================
-- DeepEval BigQuery Managed Python UDF (`deepeval_evaluate`)
--
-- Placeholders (substituted by ./install.sh):
--   YOUR_PROJECT_ID, YOUR_DATASET_ID (default: agent_analytics),
--   YOUR_LOCATION (default: us), YOUR_CONNECTION_ID (default: gemini_enterprise_connection),
--   YOUR_MODEL_ID (default: gemini-2.5-flash)
--
-- Authentication:
--   Uses Google Cloud ADC via google.auth.default() (requires `roles/aiplatform.user`
--   on the BigQuery Cloud Resource Connection SA; no API key needed).
-- =============================================================================

CREATE OR REPLACE FUNCTION `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_evaluate`(
  test_case_json STRING,
  metrics ARRAY<STRING>
)
RETURNS ARRAY<STRUCT<metric STRING, score FLOAT64, passed BOOL, reason STRING>>
LANGUAGE python
WITH CONNECTION `YOUR_PROJECT_ID.YOUR_LOCATION.YOUR_CONNECTION_ID`
OPTIONS (
  runtime_version = 'python-3.11',
  entry_point = 'evaluate_test_case',
  packages = ['deepeval==3.8.0', 'google-genai==1.15.0', 'google-auth==2.40.0'],
  max_batching_rows = 1
)
AS r'''
from dataclasses import dataclass
import importlib
import json
import os
import re
import time
import urllib.request
import google.auth

# Configure DeepEval for BigQuery's read-only container filesystem and disable telemetry.
os.environ["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["HOME"] = "/tmp"

from deepeval.models import GeminiModel
from deepeval.test_case import (
    ConversationalTestCase, LLMTestCase, LLMTestCaseParams, ToolCall, Turn, TurnParams,
)

# -----------------------------------------------------------------------------
# 1. GCP Project Auto-Discovery & Judge Model Setup
# -----------------------------------------------------------------------------
def _get_gcp_project() -> str:
    """Discovers the host GCP project from the BigQuery Cloud Resource Connection SA."""
    creds, project_id = google.auth.default()
    sa_email = getattr(creds, "service_account_email", None) or ""
    if not sa_email or sa_email == "default":
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                sa_email = resp.read().decode("utf-8").strip()
        except Exception:
            pass
    # Connection SAs follow: bqcx-<project_number>-<id>@gcp-sa-bigquery-condel...
    m = re.match(r"^bqcx-(\d+)-", sa_email)
    return m.group(1) if m else (project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", ""))


_MULTI_REGION_LOCATIONS = {"us": "us-central1", "eu": "europe-west4"}


def _resolve_gcp_location(override_loc: str = "") -> str:
    """Resolves the Gemini endpoint location without cross-region fallback (preserving DRZ)."""
    raw_loc = (
        override_loc.strip().lower()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip().lower()
        or "YOUR_LOCATION".strip().lower()
    )
    if not raw_loc or raw_loc == "your_location":
        raise ValueError(
            "GCP location is not configured. Set YOUR_LOCATION via install.sh, "
            "GOOGLE_CLOUD_LOCATION, or 'location' in test_case_json."
        )
    mapped = _MULTI_REGION_LOCATIONS.get(raw_loc, raw_loc)
    if not re.match(r"^[a-z0-9-]+$", mapped):
        raise ValueError(f"Invalid GCP location '{raw_loc}'.")
    return mapped


GCP_PROJECT = _get_gcp_project()
try:
    GCP_LOCATION = _resolve_gcp_location()
except ValueError:
    GCP_LOCATION = ""
_RAW_MODEL_ID = "YOUR_MODEL_ID".strip()
# Default to gemini-2.5-flash (latest GA model available across regional endpoints for DRZ).
# Preview models (e.g. gemini-3-flash-preview) can be configured via install.sh MODEL_ID or judge_model.
PRIMARY_MODEL_ID = (
    "gemini-2.5-flash"
    if not _RAW_MODEL_ID or _RAW_MODEL_ID == "YOUR_MODEL_ID"
    else _RAW_MODEL_ID
)
FALLBACK_MODEL_ID = "gemini-2.5-flash-lite"


class SafeGeminiModel(GeminiModel):
    """GeminiModel wrapper with JSON-prompt fallback for open Dict schemas.

    Metrics like KnowledgeRetentionMetric use Pydantic Dict fields (`additional_properties`),
    which Gemini's structured output API rejects with HTTP 400. When that occurs, this
    wrapper falls back to requesting raw JSON in the prompt and validating client-side.
    """

    def _clean_json(self, text: str) -> dict:
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(s)

    def _generate_json_fallback(self, prompt: str, schema):
        schema_json = json.dumps(schema.model_json_schema())
        raw = super().generate(
            f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{schema_json}",
            schema=None,
        )
        raw_text = raw[0] if isinstance(raw, tuple) else raw
        try:
            return schema.model_validate(self._clean_json(raw_text)), 0.0
        except Exception as e:
            raise RuntimeError(f"Fallback JSON validation failed: {e}") from e

    def generate(self, prompt: str, schema=None):
        if schema is not None and getattr(schema, "__name__", "") == "Knowledge":
            return self._generate_json_fallback(prompt, schema)
        try:
            return super().generate(prompt, schema=schema)
        except Exception as e:
            if schema is not None and "additional_properties" in str(e):
                return self._generate_json_fallback(prompt, schema)
            raise

    async def _a_generate_json_fallback(self, prompt: str, schema):
        schema_json = json.dumps(schema.model_json_schema())
        raw = await super().a_generate(
            f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{schema_json}",
            schema=None,
        )
        raw_text = raw[0] if isinstance(raw, tuple) else raw
        try:
            return schema.model_validate(self._clean_json(raw_text)), 0.0
        except Exception as e:
            raise RuntimeError(f"Fallback JSON validation failed: {e}") from e

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None and getattr(schema, "__name__", "") == "Knowledge":
            return await self._a_generate_json_fallback(prompt, schema)
        try:
            return await super().a_generate(prompt, schema=schema)
        except Exception as e:
            if schema is not None and "additional_properties" in str(e):
                return await self._a_generate_json_fallback(prompt, schema)
            raise


_MODEL_CACHE: dict[tuple[str, str, str], SafeGeminiModel] = {}


def _get_judge_model(model_id: str, location: str = "") -> SafeGeminiModel:
    """Returns a module-cached SafeGeminiModel instance to avoid per-row re-initialization."""
    resolved_loc = _resolve_gcp_location(location) if location else (GCP_LOCATION or _resolve_gcp_location())
    key = (model_id, GCP_PROJECT, resolved_loc)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SafeGeminiModel(
            model=model_id, project=GCP_PROJECT, location=resolved_loc
        )
    return _MODEL_CACHE[key]


def _is_non_retryable_error(err: Exception) -> bool:
    """Returns True for deterministic caller or HTTP 4xx errors (excluding 408/429)."""
    if isinstance(err, json.JSONDecodeError) or "ValidationError" in type(err).__name__:
        return False
    if isinstance(err, ValueError) and re.search(
        r"invalid json|json|schema|parse|outputted", str(err), re.IGNORECASE
    ):
        return False
    if isinstance(err, (TypeError, ValueError, KeyError)):
        return True
    return bool(
        re.search(
            r"(?:\bHTTP\s+|status[_ ]?code[=:\s]+|\bcode[=:\s]+)4(?!08|29)\d\d\b",
            str(err),
            re.IGNORECASE,
        )
    )


# -----------------------------------------------------------------------------
# 2. BQAA Event, AgentData & Input Normalization Helpers
# -----------------------------------------------------------------------------
def _parse_json_field(val, default):
    """Parses val as JSON if it is a string, otherwise returns val or default if None."""
    if val is None or val == "":
        return default
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return default if parsed is None else parsed
        except Exception:
            return default
    return val


def _parse_str_list(val) -> list[str]:
    """Normalizes a string or list of values into a list of non-null strings."""
    if not val:
        return []
    if isinstance(val, str):
        return [val]
    return [str(x) for x in val if x is not None] if isinstance(val, list) else [str(val)]


def _extract_bqaa_user_text(ev: dict, content_dict: dict, content_str: str) -> str:
    """Extracts user message text from a BQAA USER_MESSAGE_RECEIVED event."""
    raw = str(content_dict.get("text_summary") or content_dict.get("message") or "").strip()
    if not raw and isinstance(ev.get("content_parts"), list):
        parts = []
        for p in ev["content_parts"]:
            if not isinstance(p, dict):
                continue
            p_text = p.get("text")
            p_attr = str(p.get("part_attributes") or "")
            if (
                p.get("mime_type") == "text/plain"
                and isinstance(p_text, str)
                and p_text not in ("[BINARY DATA]", "[UPLOAD FAILED]")
                and "function_response" not in p_attr
            ):
                parts.append(p_text)
        raw = "\n".join(parts).strip()
    if not raw and content_str:
        raw = content_str.strip()
    if raw.startswith("Function response:"):
        return ""
    m = re.match(r"^(?:User Content:\s*)?text:\s*'(.*)'$", raw, re.DOTALL)
    return m.group(1) if m else raw


def _extract_bqaa_response_text(content_dict: dict, content_str: str) -> str:
    """Extracts assistant response text from a BQAA AGENT_RESPONSE or LLM_RESPONSE event."""
    resp_obj = content_dict.get("response")
    if isinstance(resp_obj, dict):
        candidates = resp_obj.get("candidates")
        if isinstance(candidates, list) and candidates:
            c0 = candidates[0] if isinstance(candidates[0], dict) else {}
            parts = (c0.get("content") or {}).get("parts") if isinstance(c0.get("content"), dict) else []
            texts = [str(p["text"]) for p in (parts if isinstance(parts, list) else []) if isinstance(p, dict) and p.get("text")]
            if texts:
                return "\n".join(texts)
    if content_dict.get("text_summary"):
        return str(content_dict["text_summary"])
    raw = resp_obj if isinstance(resp_obj, str) else content_str
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = raw.strip()
    if s.startswith("call:") or (s.startswith("Tool Name:") and "text:" not in s):
        return ""
    m_tool_text = re.match(r"^Tool Name:\s*text_response,\s*text:\s*'(.*)'$", s, re.DOTALL)
    if m_tool_text:
        return m_tool_text.group(1)
    m_text = re.match(r"^text:\s*'(.*)'$", s, re.DOTALL)
    if m_text:
        return m_text.group(1).replace("' | text: '", "\n")
    return s if not s.startswith(("Tool Name:", "Model:")) else ""


def _parse_bqaa_events(ev_list: list, wrapper: dict) -> dict:
    """Converts BQAA event rows (from BigQueryAgentAnalyticsPlugin) into DeepEval test_case fields."""
    if not isinstance(ev_list, list) or not ev_list:
        raise ValueError("BQAA events list cannot be empty")
    valid_events = [e for e in ev_list if isinstance(e, dict) and e.get("event_type")]
    if not valid_events:
        raise ValueError("BQAA events list must contain event objects with 'event_type'")
    sorted_events = [
        ev for _, ev in sorted(
            enumerate(valid_events),
            key=lambda pair: (str(pair[1].get("timestamp") or ""), pair[0]),
        )
    ]
    invocations: dict[str, list[dict]] = {}
    for ev in sorted_events:
        inv_id = str(ev.get("invocation_id") or "turn_0")
        invocations.setdefault(inv_id, []).append(ev)

    primary_agent_id = str(wrapper.get("agent_id") or wrapper.get("agent") or "")
    extracted_instruction = ""
    normalized_turns = []

    for inv_events in invocations.values():
        turn_tools = []
        turn_rc = []
        pending_tools: dict[str, dict] = {}
        user_text = ""
        fallback_user_text = ""
        final_response_text = ""

        has_agent_response = any(
            str(e.get("event_type") or "").upper() == "AGENT_RESPONSE"
            and _extract_bqaa_response_text(
                _parse_json_field(e.get("content"), {}) if isinstance(_parse_json_field(e.get("content"), {}), dict) else {},
                e.get("content") if isinstance(e.get("content"), str) else "",
            )
            for e in inv_events
        )

        for ev in inv_events:
            ev_type = str(ev.get("event_type") or "").upper()
            attrs = _parse_json_field(ev.get("attributes"), {})
            if not isinstance(attrs, dict):
                attrs = {}
            raw_content = ev.get("content")
            parsed_content = _parse_json_field(raw_content, {})
            content_dict = parsed_content if isinstance(parsed_content, dict) else {}
            content_str = raw_content if isinstance(raw_content, str) else ""

            if not primary_agent_id and (ev.get("agent") or attrs.get("root_agent_name")):
                primary_agent_id = str(ev.get("agent") or attrs.get("root_agent_name"))

            if ev_type == "AGENT_STARTING":
                instr = str(content_dict.get("text_summary") or content_str or attrs.get("static_instruction") or "").strip()
                if instr and not instr.startswith(("<function", "<bound method", "functools.partial")):
                    extracted_instruction = instr

            elif ev_type == "LLM_REQUEST":
                sys_prompt = str(content_dict.get("system_prompt") or "").strip()
                if sys_prompt and not extracted_instruction:
                    extracted_instruction = sys_prompt
                if not fallback_user_text:
                    req_contents = (content_dict.get("request") or {}).get("contents") if isinstance(content_dict.get("request"), dict) else None
                    if isinstance(req_contents, list) and req_contents and isinstance(req_contents[0], dict):
                        parts = req_contents[0].get("parts")
                        if isinstance(parts, list) and parts and isinstance(parts[0], dict) and parts[0].get("text"):
                            fallback_user_text = str(parts[0]["text"])
                    elif isinstance(content_dict.get("prompt"), list):
                        for p_item in reversed(content_dict["prompt"]):
                            if isinstance(p_item, dict) and p_item.get("role") == "user" and p_item.get("content"):
                                fallback_user_text = str(p_item["content"])
                                break
                    elif content_str:
                        m_p = re.search(r"Prompt:\s*user:\s*text:\s*'([^']*)'", content_str)
                        if m_p:
                            fallback_user_text = m_p.group(1)

            elif ev_type == "USER_MESSAGE_RECEIVED":
                u_msg = _extract_bqaa_user_text(ev, content_dict, content_str)
                if u_msg:
                    user_text = u_msg

            elif ev_type in ("TOOL_STARTING", "HITL_CONFIRMATION_REQUEST", "HITL_INPUT_REQUEST"):
                t_name = content_dict.get("tool") or content_dict.get("tool_name")
                t_desc = content_dict.get("description")
                t_args = content_dict.get("args") if content_dict.get("args") is not None else content_dict.get("tool_input")
                if not t_name and content_str:
                    m_n = re.search(r"Tool Name:\s*([^,\n]+)", content_str)
                    if m_n:
                        t_name = m_n.group(1).strip()
                    m_d = re.search(r"Description:\s*([^,\n]+?)(?:,\s*Arguments:|$)", content_str)
                    if m_d:
                        t_desc = m_d.group(1).strip()
                    m_a = re.search(r"Arguments:\s*(\{.*\})\s*$", content_str, re.DOTALL)
                    if m_a:
                        t_args = _parse_json_field(m_a.group(1), {})
                t_name = str(t_name or "unknown_tool")
                if ev_type == "TOOL_STARTING" and (t_name == "transfer_to_agent" or t_name.startswith("adk_request_")):
                    continue
                parsed_args = _parse_json_field(t_args, {})
                args_dict = parsed_args if isinstance(parsed_args, dict) else ({"value": parsed_args} if parsed_args is not None else {})
                pending_tools[t_name] = {"args": args_dict, "description": str(t_desc or f"Tool {t_name}")}

            elif ev_type in ("TOOL_COMPLETED", "TOOL_ERROR", "HITL_CONFIRMATION_REQUEST_COMPLETED", "HITL_INPUT_REQUEST_COMPLETED"):
                t_name = content_dict.get("tool") or content_dict.get("tool_name")
                if not t_name and content_str:
                    m_n = re.search(r"Tool Name:\s*([^,\n]+)", content_str)
                    if m_n:
                        t_name = m_n.group(1).strip()
                t_name = str(t_name or "unknown_tool")
                if ev_type == "TOOL_COMPLETED" and (t_name == "transfer_to_agent" or t_name.startswith("adk_request_")):
                    continue
                pending = pending_tools.get(t_name, {})
                inline_args = content_dict.get("args") if content_dict.get("args") is not None else content_dict.get("tool_input")
                if inline_args is None and content_str:
                    m_a = re.search(r"Arguments:\s*(\{.*\})\s*$", content_str, re.DOTALL)
                    if m_a:
                        inline_args = _parse_json_field(m_a.group(1), {})
                parsed_args = _parse_json_field(inline_args, None) if inline_args is not None else pending.get("args", {})
                args_dict = parsed_args if isinstance(parsed_args, dict) else ({"value": parsed_args} if parsed_args is not None else {})
                desc_str = str(content_dict.get("description") or pending.get("description") or f"Tool {t_name}")

                to_obj = content_dict.get("tool_output")
                raw_res = (
                    to_obj.get("result")
                    if isinstance(to_obj, dict) and "result" in to_obj
                    else (to_obj if to_obj is not None else content_dict.get("result"))
                )
                if raw_res is None and content_str:
                    m_r = re.search(r"Result:\s*(.+)$", content_str, re.DOTALL)
                    if m_r:
                        raw_res = _parse_json_field(m_r.group(1).strip(), m_r.group(1).strip())
                if raw_res is None and ev_type == "TOOL_ERROR":
                    raw_res = {"error": str(ev.get("error_message") or "Tool execution failed")}
                parsed_res = _parse_json_field(raw_res, raw_res)
                resp_dict = parsed_res if isinstance(parsed_res, dict) else {"result": parsed_res if parsed_res is not None else "ok"}

                turn_tools.append({
                    "name": t_name,
                    "description": desc_str,
                    "input_parameters": args_dict,
                    "output": resp_dict,
                })
                if (
                    isinstance(parsed_res, dict)
                    and isinstance(parsed_res.get("content"), list)
                    and parsed_res["content"]
                    and isinstance(parsed_res["content"][0], dict)
                    and parsed_res["content"][0].get("text")
                ):
                    rc_val = str(parsed_res["content"][0]["text"])
                elif isinstance(parsed_res, (dict, list)):
                    rc_val = json.dumps(parsed_res)
                else:
                    rc_val = str(parsed_res if parsed_res is not None else "null")
                turn_rc.append(f"Tool {t_name} returned: {rc_val}")

            elif ev_type == "AGENT_RESPONSE" or (ev_type == "LLM_RESPONSE" and not has_agent_response):
                resp_text = _extract_bqaa_response_text(content_dict, content_str)
                if resp_text:
                    final_response_text = resp_text

        if not user_text and fallback_user_text:
            user_text = fallback_user_text

        normalized_turns.append({
            "input": user_text,
            "actual_output": final_response_text,
            "tools_called": turn_tools,
            "retrieval_context": turn_rc,
        })

    primary_instr = str(
        wrapper.get("chatbot_role")
        or wrapper.get("instruction")
        or extracted_instruction
        or "Helpful AI Assistant"
    )
    last_turn = normalized_turns[-1]
    result = dict(wrapper)
    result.setdefault("agent_id", primary_agent_id or "root_agent")
    result.setdefault("chatbot_role", primary_instr)
    result.setdefault("input", last_turn["input"])
    result.setdefault("actual_output", last_turn["actual_output"])
    result.setdefault("tools_called", [t for turn in normalized_turns for t in turn["tools_called"]])
    result.setdefault("retrieval_context", [c for turn in normalized_turns for c in turn["retrieval_context"]])
    result["turns"] = normalized_turns
    return result


def _parse_agent_data_dict(raw_dict: dict) -> dict:
    """Normalizes a canonical AgentData JSON dict (or wrapper with agent_eval_data/trace) into tc_data."""
    inner = _parse_json_field(raw_dict.get("agent_eval_data") or raw_dict.get("trace"), None)
    agent_data = inner if isinstance(inner, dict) else raw_dict
    raw_agents = agent_data.get("agents") if isinstance(agent_data.get("agents"), dict) else {}
    raw_turns = agent_data.get("turns") if isinstance(agent_data.get("turns"), list) else []
    if not raw_turns:
        raise ValueError("AgentData turns list cannot be empty")
    primary_agent_id = str(raw_dict.get("agent_id") or raw_dict.get("agent") or next(iter(raw_agents), "root_agent"))
    primary_instr = str(
        raw_dict.get("chatbot_role")
        or raw_dict.get("instruction")
        or (raw_agents.get(primary_agent_id) or {}).get("instruction")
        or "Helpful AI Assistant"
    )
    normalized_turns = []
    for t_obj in raw_turns:
        if not isinstance(t_obj, dict):
            continue
        u_text, a_text, turn_tools, turn_rc = "", "", [], []
        pending_args: dict[str, dict] = {}
        for ev in (t_obj.get("events") if isinstance(t_obj.get("events"), list) else []):
            if not isinstance(ev, dict):
                continue
            c_obj = ev.get("content") if isinstance(ev.get("content"), dict) else {}
            role = str(c_obj.get("role") or "")
            for part in (c_obj.get("parts") if isinstance(c_obj.get("parts"), list) else []):
                if not isinstance(part, dict):
                    continue
                if part.get("text") is not None:
                    if role == "user" or ev.get("author") == "user":
                        u_text = str(part["text"])
                    else:
                        a_text = str(part["text"])
                if isinstance(part.get("function_call"), dict):
                    fc = part["function_call"]
                    fc_name = str(fc.get("name") or "unknown_tool")
                    if fc_name != "transfer_to_agent":
                        pending_args[fc_name] = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                if isinstance(part.get("function_response"), dict):
                    fr = part["function_response"]
                    fr_name = str(fr.get("name") or "unknown_tool")
                    fr_resp = fr.get("response")
                    resp_dict = fr_resp if isinstance(fr_resp, dict) else {"result": fr_resp}
                    turn_tools.append({
                        "name": fr_name,
                        "description": f"Tool {fr_name}",
                        "input_parameters": pending_args.get(fr_name, {}),
                        "output": resp_dict,
                    })
                    turn_rc.append(f"Tool {fr_name} returned: {json.dumps(fr_resp)}")
        normalized_turns.append({
            "input": u_text,
            "actual_output": a_text,
            "tools_called": turn_tools,
            "retrieval_context": turn_rc,
        })
    if not normalized_turns:
        raise ValueError("AgentData turns list cannot be empty")
    last_turn = normalized_turns[-1]
    result = dict(raw_dict)
    result.setdefault("agent_id", primary_agent_id)
    result.setdefault("chatbot_role", primary_instr)
    result.setdefault("input", last_turn["input"])
    result.setdefault("actual_output", last_turn["actual_output"])
    result.setdefault("tools_called", [t for turn in normalized_turns for t in turn["tools_called"]])
    result.setdefault("retrieval_context", [c for turn in normalized_turns for c in turn["retrieval_context"]])
    result["turns"] = normalized_turns
    return result


def _normalize_input_payload(test_case_json) -> dict:
    """Parses BQAA event arrays, wrapped BQAA events, canonical AgentData, or test_case dicts."""
    decoded = json.loads(test_case_json) if isinstance(test_case_json, str) else test_case_json
    if isinstance(decoded, list):
        return _parse_bqaa_events(decoded, {})
    if isinstance(decoded, dict):
        if "events" in decoded and decoded.get("events") is not None:
            ev_list = _parse_json_field(decoded.get("events"), None)
            return _parse_bqaa_events(ev_list, decoded)
        if "event_type" in decoded:
            return _parse_bqaa_events([decoded], {})
        raw_turns = decoded.get("turns")
        is_agent_data_turns = (
            isinstance(raw_turns, list)
            and bool(raw_turns)
            and isinstance(raw_turns[0], dict)
            and "events" in raw_turns[0]
        )
        if "agent_eval_data" in decoded or "trace" in decoded or is_agent_data_turns:
            return _parse_agent_data_dict(decoded)
        return decoded
    raise ValueError("test_case_json must decode to a BQAA event array or JSON object")


def _parse_tools(raw_tools) -> list[ToolCall]:
    """Converts a JSON string, single tool dict, or list of tool dicts/names into DeepEval ToolCall objects."""
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except Exception:
            return []
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        return []
    tool_calls = []
    for t in raw_tools:
        if isinstance(t, str):
            tool_calls.append(ToolCall(name=t))
        elif isinstance(t, dict):
            raw_params = (
                t.get("input_parameters")
                if t.get("input_parameters") is not None
                else (t.get("args") if t.get("args") is not None else t.get("tool_input"))
            )
            if isinstance(raw_params, str):
                try:
                    raw_params = json.loads(raw_params)
                except Exception:
                    raw_params = None
            tool_calls.append(ToolCall(
                name=str(t.get("name") or t.get("tool_name") or "unknown_tool"),
                description=t.get("description"),
                input_parameters=raw_params if isinstance(raw_params, dict) else None,
                output=t.get("output"),
            ))
    return tool_calls


def _parse_turns(raw_turns) -> list[Turn]:
    """Converts either [{role, content}] or [{input, actual_output}] into DeepEval Turn objects."""
    if isinstance(raw_turns, str):
        try:
            raw_turns = json.loads(raw_turns)
        except Exception:
            return []
    if not isinstance(raw_turns, list):
        return []
    turns = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        rc_list = _parse_str_list(item.get("retrieval_context")) or None
        tools = _parse_tools(item.get("tools_called") or item.get("tools_called_json")) or None
        if "role" in item:
            raw_role = str(item.get("role") or "user").strip().lower()
            if raw_role in ("user", "human"):
                role = "user"
            elif raw_role in ("assistant", "model", "agent"):
                role = "assistant"
            else:
                continue
            turns.append(Turn(
                role=role,
                content=str(item.get("content") or ""),
                tools_called=tools,
                retrieval_context=rc_list,
            ))
        else:
            user_text = item.get("input") or item.get("input_query")
            if user_text:
                turns.append(Turn(role="user", content=str(user_text)))
            if item.get("actual_output") or tools or rc_list:
                turns.append(Turn(
                    role="assistant",
                    content=str(item.get("actual_output") or ""),
                    tools_called=tools,
                    retrieval_context=rc_list,
                ))
    return turns


# -----------------------------------------------------------------------------
# 3. Metric Registry & Builder
# -----------------------------------------------------------------------------
# Maps metric_name -> (metric_class_name, default_threshold, required_fields)
METRIC_REGISTRY = {
    # Single-turn RAG & Agent metrics
    "answer_relevancy": ("AnswerRelevancyMetric", 0.7, ()),
    "faithfulness": ("FaithfulnessMetric", 0.7, ("retrieval_context",)),
    "contextual_relevancy": ("ContextualRelevancyMetric", 0.7, ("retrieval_context",)),
    "contextual_precision": ("ContextualPrecisionMetric", 0.7, ("retrieval_context", "expected_output")),
    "contextual_recall": ("ContextualRecallMetric", 0.7, ("retrieval_context", "expected_output")),
    "hallucination": ("HallucinationMetric", 0.5, ("context",)),
    "tool_correctness": ("ToolCorrectnessMetric", 0.7, ("expected_tools",)),
    "task_completion": ("TaskCompletionMetric", 0.7, ()),
    # Multi-turn conversational metrics
    "turn_relevancy": ("TurnRelevancyMetric", 0.7, ()),
    "knowledge_retention": ("KnowledgeRetentionMetric", 0.7, ()),
    "conversation_completeness": ("ConversationCompletenessMetric", 0.7, ()),
    "role_adherence": ("RoleAdherenceMetric", 0.7, ()),
}

CONVERSATIONAL_METRICS = {
    "turn_relevancy", "knowledge_retention", "conversation_completeness", "role_adherence",
}


@dataclass(frozen=True)
class _ParsedTestCase:
    tc_data: dict
    expected_output: object
    retrieval_context: list[str]
    context: list[str]
    tools_called: list
    expected_tools: list
    turns: list
    is_conversational: bool


def _build_metric(m_name: str, judge_model, parsed: _ParsedTestCase):
    """Lazily imports and instantiates the requested DeepEval metric or custom G-Eval (GEval) rubric."""
    metrics_mod = importlib.import_module("deepeval.metrics")
    tc_data = parsed.tc_data
    expected_output = parsed.expected_output
    retrieval_context = parsed.retrieval_context
    context = parsed.context
    tools_called = parsed.tools_called
    expected_tools = parsed.expected_tools
    turns = parsed.turns
    is_conversational = parsed.is_conversational
    m_lower = m_name.lower()
    if m_lower in METRIC_REGISTRY:
        if m_lower in CONVERSATIONAL_METRICS and not is_conversational:
            return m_lower, "MISSING_TURNS"
        cls_name, threshold, req_fields = METRIC_REGISTRY[m_lower]
        available = {
            "retrieval_context": bool(retrieval_context),
            "context": bool(context),
            "expected_output": bool(expected_output),
            "expected_tools": bool(expected_tools),
        }
        if any(not available.get(f) for f in req_fields):
            return m_lower, None
        metric_cls = getattr(metrics_mod, cls_name)
        return m_lower, metric_cls(threshold=threshold, model=judge_model, async_mode=False)

    # Custom single-turn or multi-turn DeepEval G-Eval rubric ("geval:<Name>:<Criteria>" or
    # "conversational_geval:<Name>:<Criteria>"). Note: GEval and ConversationalGEval are DeepEval's
    # built-in implementations of the academic G-Eval LLM-as-a-Judge framework (unrelated to GEAP).
    if m_lower in ("geval", "conversational_geval") or m_lower.startswith(("conversational_geval:", "geval:")):
        parts = m_name.split(":", 2)
        is_convo_geval = m_lower == "conversational_geval" or m_lower.startswith("conversational_geval:")
        default_name = "Conversational Quality" if is_convo_geval else "Custom Quality"
        default_criteria = (
            "Evaluate whether the multi-turn conversation is coherent, retains context across turns, and fulfills the user request."
            if is_convo_geval
            else "Evaluate whether the response is accurate, helpful, polite, and safe."
        )
        g_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else default_name
        g_criteria = parts[2].strip() if len(parts) > 2 and parts[2].strip() else tc_data.get("criteria", default_criteria)

        if is_convo_geval:
            if not is_conversational:
                return f"conversational_geval:{g_name}", "MISSING_TURNS"
            # Guard optional enum members across DeepEval 3.x minor versions.
            turn_params = [TurnParams.ROLE, TurnParams.CONTENT]
            if any(getattr(t, "retrieval_context", None) for t in turns) and hasattr(TurnParams, "RETRIEVAL_CONTEXT"):
                turn_params.append(TurnParams.RETRIEVAL_CONTEXT)
            if any(getattr(t, "tools_called", None) for t in turns) and hasattr(TurnParams, "TOOLS_CALLED"):
                turn_params.append(TurnParams.TOOLS_CALLED)
            convo_geval_cls = getattr(metrics_mod, "ConversationalGEval")
            return f"conversational_geval:{g_name}", convo_geval_cls(
                name=g_name, criteria=g_criteria, evaluation_params=turn_params,
                model=judge_model, threshold=0.7, async_mode=False,
            )
        # Guard optional enum members across DeepEval 3.x minor versions.
        eval_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        if expected_output:
            eval_params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
        if retrieval_context and hasattr(LLMTestCaseParams, "RETRIEVAL_CONTEXT"):
            eval_params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
        if context:
            eval_params.append(LLMTestCaseParams.CONTEXT)
        if tools_called and hasattr(LLMTestCaseParams, "TOOLS_CALLED"):
            eval_params.append(LLMTestCaseParams.TOOLS_CALLED)
        geval_cls = getattr(metrics_mod, "GEval")
        return f"geval:{g_name}", geval_cls(
            name=g_name, criteria=g_criteria, evaluation_params=eval_params,
            model=judge_model, threshold=0.7, async_mode=False,
        )

    return m_name, "UNSUPPORTED"


# -----------------------------------------------------------------------------
# 4. UDF Entry Point
# -----------------------------------------------------------------------------
def evaluate_test_case(test_case_json: str, metrics: list[str]) -> list[dict]:
    """Evaluates a single-turn or multi-turn agent trace against the requested DeepEval metrics."""
    if not test_case_json or not metrics:
        return []
    try:
        tc_data = _normalize_input_payload(test_case_json)
        has_turns_key = "turns" in tc_data and tc_data.get("turns") is not None
        turns = _parse_turns(tc_data.get("turns")) if has_turns_key else []
        if has_turns_key and not turns:
            raise ValueError("turns list cannot be empty when 'turns' is provided")
        is_conversational = bool(turns)
        expected_output = tc_data.get("expected_output")
        retrieval_context = _parse_str_list(tc_data.get("retrieval_context"))
        tools_called = _parse_tools(tc_data.get("tools_called") or tc_data.get("tools_called_json"))
        has_top_rc = bool(retrieval_context)
        has_top_tools = bool(tools_called)
        last_user, last_assistant = "", ""
        if is_conversational:
            for t in turns:
                if getattr(t, "role", "") == "user" and getattr(t, "content", ""):
                    last_user = t.content
                elif getattr(t, "role", "") == "assistant" and getattr(t, "content", ""):
                    last_assistant = t.content
                if not has_top_rc and getattr(t, "retrieval_context", None):
                    retrieval_context.extend(t.retrieval_context)
                if not has_top_tools and getattr(t, "tools_called", None):
                    tools_called.extend(t.tools_called)
        context = _parse_str_list(tc_data.get("context")) or retrieval_context
        expected_tools = _parse_tools(tc_data.get("expected_tools"))
        parsed_tc = _ParsedTestCase(
            tc_data=tc_data,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
            context=context,
            tools_called=tools_called,
            expected_tools=expected_tools,
            turns=turns,
            is_conversational=is_conversational,
        )

        # Build LLMTestCase for single-turn metrics and ConversationalTestCase when turns are provided
        llm_test_case = LLMTestCase(
            input=str(tc_data.get("input") or tc_data.get("input_query") or last_user or ""),
            actual_output=str(tc_data.get("actual_output") or last_assistant or ""),
            expected_output=str(expected_output) if expected_output is not None else None,
            retrieval_context=retrieval_context or None,
            context=context or None,
            tools_called=tools_called,
            expected_tools=expected_tools,
        )
        convo_test_case = (
            ConversationalTestCase(
                turns=turns,
                chatbot_role=tc_data.get("chatbot_role") or "Helpful AI Assistant",
                scenario=tc_data.get("scenario"),
                expected_outcome=tc_data.get("expected_outcome"),
                user_description=tc_data.get("user_description"),
                context=context or None,
            )
            if is_conversational
            else None
        )
    except Exception as e:
        return [{"metric": "parse_error", "score": None, "passed": False, "reason": f"Invalid test_case_json: {e}"}]

    primary_model_id = str(tc_data.get("judge_model") or PRIMARY_MODEL_ID).strip() or PRIMARY_MODEL_ID
    fallback_model_id = str(tc_data.get("fallback_judge_model") or FALLBACK_MODEL_ID).strip() or FALLBACK_MODEL_ID
    req_location = str(tc_data.get("location") or "").strip()
    results = []

    for m_spec in metrics:
        if not m_spec:
            continue
        m_name = m_spec.strip()
        try:
            primary_model = _get_judge_model(primary_model_id, req_location)
        except Exception as e:
            results.append({
                "metric": m_name.lower(),
                "score": None,
                "passed": None,
                "reason": f"Error: {type(e).__name__}: {str(e)}",
            })
            continue

        label, metric = _build_metric(m_name, primary_model, parsed_tc)
        if metric == "MISSING_TURNS":
            results.append({"metric": label, "score": None, "passed": None, "reason": "N/A (Multi-turn metric requires 'turns' in test_case_json)"})
            continue
        if metric is None:
            results.append({"metric": label, "score": None, "passed": None, "reason": "N/A (Missing required context, expected_output, or expected_tools in test_case)"})
            continue
        if metric == "UNSUPPORTED":
            results.append({"metric": label, "score": None, "passed": False, "reason": f"Unsupported metric name: {m_name}"})
            continue

        use_convo_case = label in CONVERSATIONAL_METRICS or label.startswith("conversational_geval:")
        target_case = convo_test_case if use_convo_case else llm_test_case

        # Retry up to 3 attempts; fall back to fallback_model_id on the final attempt
        last_err = None
        for attempt in range(3):
            try:
                if attempt == 2:
                    _, metric = _build_metric(m_name, _get_judge_model(fallback_model_id, req_location), parsed_tc)
                metric.measure(target_case)
                reason_text = str(metric.reason or "")
                if attempt == 2:
                    reason_text = f"[Judge: {fallback_model_id}] {reason_text}".strip()
                results.append({
                    "metric": label,
                    "score": float(metric.score) if metric.score is not None else None,
                    "passed": bool(metric.is_successful()),
                    "reason": reason_text,
                })
                last_err = None
                break
            except Exception as e:
                last_err = e
                if _is_non_retryable_error(e):
                    break
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

        if last_err is not None:
            results.append({
                "metric": label,
                "score": None,
                "passed": None,
                "reason": f"Error: {type(last_err).__name__}: {str(last_err)}",
            })

    return results
''';
