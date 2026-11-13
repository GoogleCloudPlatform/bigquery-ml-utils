# Evaluate AI Agent Traces in BigQuery Using Python UDFs and DeepEval

This directory provides **BigQuery Managed Python UDFs**
(`deepeval_parse_trace` and `deepeval_evaluate`), a one-command installer
script (`install.sh`), and end-to-end SQL queries for evaluating **Single-Turn**
and **Multi-Turn** AI agent execution traces directly inside BigQuery using
[DeepEval](https://github.com/confident-ai/deepeval) and
[Gemini on Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models)
(`gemini-3.8-flash`).

Because `deepeval_evaluate` authenticates to Gemini Enterprise Agent Platform
using Application Default Credentials (`google.auth.default()`) attached to a
BigQuery Cloud Resource Connection, **no Gemini API keys or Secret Manager
setup are required**.

---

## Prerequisites

1. A Google Cloud project with billing enabled.
2. Google Cloud SDK
   ([`gcloud` and `bq` CLI](https://cloud.google.com/sdk/docs/install))
   installed (pre-installed in
   [Google Cloud Shell](https://shell.cloud.google.com/)).
3. Permissions in your Google Cloud project to enable APIs, create a BigQuery
   dataset and Cloud Resource Connection, and grant the **Agent Platform User**
   IAM role (`roles/aiplatform.user`).

---

## Step 1: Configure Environment Variables & Install (`./install.sh`)

Clone this repository, export your Google Cloud project and BigQuery resource
variables in your Linux terminal, and run `./install.sh`:

```bash
git clone https://github.com/GoogleCloudPlatform/bigquery-ml-utils.git
cd bigquery-ml-utils/sql_scripts/agent_evaluation/deepeval

# 1. Define your environment variables (replace with your GCP project ID):
export PROJECT_ID="your-gcp-project-id"
export DATASET_ID="agent_analytics"
export LOCATION="us"
export CONNECTION_ID="gemini_enterprise_connection"
export MODEL_ID="gemini-3.8-flash"

# 2. Run the one-command installer:
./install.sh "${PROJECT_ID}" "${DATASET_ID}" "${LOCATION}" "${CONNECTION_ID}" "${MODEL_ID}"
```

### What `install.sh` Does

1. Enables the **BigQuery**, **BigQuery Connection**, and **Gemini Enterprise
   Agent Platform** APIs (`bigquery.googleapis.com`,
   `bigqueryconnection.googleapis.com`, `aiplatform.googleapis.com`).
2. Creates the target BigQuery dataset (`${PROJECT_ID}:${DATASET_ID}`) in
   `${LOCATION}` if it does not already exist (or verifies that an existing
   dataset's location matches `${LOCATION}`).
3. Creates the BigQuery Cloud Resource Connection
   (`${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}`) if it does not already exist.
4. Grants the **Agent Platform User** role (`roles/aiplatform.user`) at the
   project level (`gcloud projects add-iam-policy-binding`) to the connection's
   service account so the Python UDF container can invoke Gemini models on
   Gemini Enterprise Agent Platform via `google.auth.default()`.
5. Substitutes `${PROJECT_ID}`, `${DATASET_ID}`, `${LOCATION}`,
   `${CONNECTION_ID}`, and `${MODEL_ID}` into
   [`deepeval_udfs.sql`](deepeval_udfs.sql) and deploys the
   `deepeval_parse_trace` and `deepeval_evaluate` Python UDFs via `bq query`.

> **Note on First Execution, IAM Propagation, Data Residency & Judge Model
> Selection:**
> - **IAM Propagation:** Project-level IAM bindings granted by `install.sh` can
>   take 1–2 minutes to propagate. If a smoke test run immediately after
>   installation returns `403 PERMISSION_DENIED`, wait ~1–2 minutes and retry.
> - **Cold Start:** The very first query you run after `CREATE OR REPLACE`
>   `FUNCTION` takes ~50–75 seconds longer while BigQuery builds and caches the
>   Python container image with pinned packages (`deepeval==3.8.0`,
>   `google-genai==1.15.0`, and `google-auth==2.40.0`). Subsequent queries reuse
>   the cached container image.
> - **Data Residency & Evaluation Regions:** `install.sh` substitutes
>   `${LOCATION}` into the UDF (mapping multi-regions `us` → `us-central1` and
>   `eu` → `europe-west4`, and passing explicit regions such as `us-east4`,
>   `europe-west1`, or `asia-northeast1` directly; if the location is
>   unconfigured or invalid, the UDF fails fast with a `ValueError`). Because
>   Gemini 3.x models such as the default `gemini-3.8-flash` are not served on
>   single-region endpoints (and BigQuery datasets/connections cannot be
>   created in `global`), **Gemini 3.x models route evaluation requests to the
>   `global` endpoint by default** when no per-row `location` is specified,
>   meaning prompt traffic may be processed outside `${LOCATION}`. Regional
>   models such as `gemini-2.5-flash` (including automatic retry fallback on
>   attempt 3) and any explicit per-row `location` override in `test_case_json`
>   always stay on the configured regional endpoint and never fall back across
>   regions. For strict in-region data residency compliance, install with
>   `MODEL_ID="gemini-2.5-flash"` (or set `"judge_model": "gemini-2.5-flash"`
>   in `test_case_json`).
> - **Judge Model Selection:** The default primary judge model is
>   `gemini-3.8-flash` (routed to `global` unless a per-row `location` is set),
>   with `gemini-2.5-flash` (routed to the configured regional endpoint) as the
>   fallback model on the final retry attempt for transient or invalid-JSON
>   evaluation errors (deterministic HTTP 4xx errors other than 408/429 fail
>   fast without retrying). When installing with `MODEL_ID="gemini-2.5-flash"`
>   (or setting `"judge_model": "gemini-2.5-flash"`), the fallback model also
>   defaults to `gemini-2.5-flash` unless `fallback_judge_model` is overridden.
>   You can override the model at install time via `MODEL_ID` (5th argument to
>   `./install.sh`) or per query via `judge_model`, `fallback_judge_model`, and
>   `location` in `test_case_json`.
> - **Memory & Batching:** Because `deepeval` has a substantial runtime
>   dependency tree, the UDF sets `max_batching_rows = 1`, caches the judge
>   model client at module level, and lazily imports metric classes inside
>   `_build_metric` to minimize baseline memory overhead. For large multi-turn
>   evaluation queries or long conversation histories, consider evaluating in
>   batches (for example, using `LIMIT` / partition filters) or configuring
>   higher container memory limits if your BigQuery environment supports custom
>   container resource options.

---

## Step 2: Quick Smoke Test

Using the same terminal session where `PROJECT_ID`, `DATASET_ID`, and
`LOCATION` are exported, run a quick single-turn evaluation with `bq query`:

```bash
bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false <<EOF
SELECT
  \`${PROJECT_ID}.${DATASET_ID}.deepeval_evaluate\`(
    TO_JSON_STRING(STRUCT(
      'Where is my order #999?' AS input,
      'Order #999 shipped yesterday via UPS (tracking: 1Z999).' AS actual_output,
      ['Order #999 status: Shipped yesterday via UPS, tracking number 1Z999.'] AS retrieval_context
    )),
    [
      'answer_relevancy',
      'faithfulness',
      'geval:Completeness:Verify that the response includes both the shipping carrier and tracking number.'
    ]
  ) AS eval_results;
EOF
```

The UDF returns an `ARRAY<STRUCT<...>>` (`metric`, `score`, `passed`, `reason`,
`judge_input_tokens`, and `judge_output_tokens`) containing the numeric score
(`0.0` to `1.0`), pass/fail status, LLM judge reasoning, and the **Gemini judge
model's** evaluation token usage (`judge_input_tokens` and
`judge_output_tokens`, including thinking tokens, consumed by the judge model to
score each metric — not the tokens consumed by the agent turn being evaluated).

---

## Step 3: Function Reference & Supported Metrics

### Signatures

```sql
-- 1. Offline Trace Parser (no external packages or connection needed; zero Gemini quota cost)
`${PROJECT_ID}.${DATASET_ID}.deepeval_parse_trace`(
  events_json STRING
)
RETURNS STRING

-- 2. Online DeepEval Metric Evaluator
`${PROJECT_ID}.${DATASET_ID}.deepeval_evaluate`(
  test_case_json STRING,
  metrics ARRAY<STRING>
)
RETURNS ARRAY<STRUCT<
  metric STRING,
  score FLOAT64,
  passed BOOL,
  reason STRING,
  judge_input_tokens INT64,   -- Judge model input tokens consumed during metric evaluation
  judge_output_tokens INT64   -- Judge model output + thinking tokens consumed during metric evaluation
>>
```

### Supported Input Formats

`deepeval_parse_trace` normalizes **BigQuery Agent Analytics (BQAA)** event
rows (`BigQueryAgentAnalyticsPlugin`) or canonical **`AgentData`** JSON into the
normalized `test_case_json` object consumed by `deepeval_evaluate` (so you can
materialize or inspect parsed traces in a table or view before spending Gemini
quota):

1. **BQAA Event Array (`deepeval_parse_trace(TO_JSON_STRING(events))`)**:
   - Parse rows from any BQAA table (`agent_events` / `agent_events_v2`):
     - **Single-Turn (`GROUP BY session_id, invocation_id`)**: Builds a
       single-turn trace (`LLMTestCase`).
     - **Multi-Turn (`GROUP BY session_id`)**: Automatically groups BQAA events
       by `invocation_id` in chronological order into a multi-turn trace
       (`ConversationalTestCase`), while carrying over prior turns' dialogue and
       tool outputs in top-level `retrieval_context` if evaluated with
       single-turn metrics.
   - **Performance Best Practice**:
     - Apply `GROUP BY` (and `ORDER BY ... LIMIT` when sampling) in a CTE
       **before** calling `deepeval_parse_trace`, and project only the 9 BQAA
       columns read by the parser (`timestamp`, `event_type`, `invocation_id`,
       `agent`, `span_id`, `error_message`, `attributes`, `content`,
       `content_parts`) rather than `ARRAY_AGG(t)` in the same `SELECT` block as
       `LIMIT` (in GoogleSQL, `SELECT` expressions run before
       `ORDER BY ... LIMIT`). If your `BigQueryAgentAnalyticsPlugin` uses
       `payload_column_denylist` to omit `content`, `content_parts`, or
       `attributes` from the table schema, drop those denied columns from the
       `STRUCT(...)` projection as well.
   - For multi-agent ADK apps that delegate to sub-agents via `AgentTool`,
     filter `WHERE t.agent = 'YOUR_ROOT_AGENT'` before `GROUP BY` to drop
     sub-agent groups directly in SQL (note: row-level `WHERE t.agent` applies
     to `AgentTool` delegates, not `sub_agents` / `transfer_to_agent` handoffs
     where the sub-agent emits the final response inside the root session).
     Alternatively, pass `'YOUR_ROOT_AGENT' AS root_agent` in the wrapper
     `STRUCT` so `deepeval_parse_trace` strips internal `AgentTool` sub-agent
     events (while preserving `transfer_to_agent` handoffs) and returns SQL
     `NULL` for sub-agent-only groups; when relying on `root_agent` in the
     wrapper `STRUCT`, filter `WHERE parsed_trace IS NOT NULL` before calling
     `deepeval_evaluate`.
   - To supply optional reference or parser fields (such as `expected_output`,
     `expected_tools`, `root_agent`, `output_response_field`, or `judge_model`)
     alongside BQAA rows, wrap the event
     array in a `STRUCT` passed to `deepeval_parse_trace`:
     ```sql
     `${PROJECT_ID}.${DATASET_ID}.deepeval_parse_trace`(
       TO_JSON_STRING(STRUCT(
         events,  -- 9-column BQAA event array from grouped_invocations CTE
         'YOUR_ROOT_AGENT' AS root_agent,
         'Expected response text' AS expected_output,
         [STRUCT('OrderTool' AS name, '{"order_id": "999"}' AS args)] AS expected_tools,
         'response' AS output_response_field
       ))
     )
     ```
2. **Canonical `AgentData` JSON (`deepeval_parse_trace(agent_data_json)`)**:
   - Normalizes `AgentData` JSON objects (`{"agents": {...}, "turns": [{"turn_index": 0, "turn_id": "...", "events": [...]}]}` or `{agent_eval_data: ...}` / `{trace: ...}`).
3. **Ad-Hoc Key-Value Test Case `STRUCT` (`TO_JSON_STRING(STRUCT(...))`)**:
   - Can be passed directly to `deepeval_evaluate` (or through
     `deepeval_parse_trace`) for inline SQL smoke tests and custom table schemas
     (see fields below).

### Mode A: Single-Turn Evaluation (`LLMTestCase`)

When evaluating a single BQAA invocation (`GROUP BY session_id, invocation_id`)
or a single-turn `STRUCT`, `deepeval_evaluate` builds a DeepEval `LLMTestCase`.

#### Optional / Ad-Hoc JSON Fields (`test_case_json`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `events` | `ARRAY<STRUCT>` | Direct BQAA | Array of the 9 required BQAA event columns (`ARRAY_AGG(STRUCT(t.timestamp, t.event_type, t.invocation_id, t.agent, t.span_id, t.error_message, t.attributes, t.content, t.content_parts) ORDER BY t.timestamp)`). You can also pass `TO_JSON_STRING(events)` directly as the top-level JSON array to `deepeval_parse_trace`. |
| `input` (or `input_query`) | `STRING` | Automatic from BQAA | The user prompt or question sent to the agent (automatically extracted from `USER_MESSAGE_RECEIVED` / `LLM_REQUEST` when using BQAA input). |
| `actual_output` | `STRING` | Automatic from BQAA | The agent's final response (automatically extracted from `AGENT_RESPONSE` / `LLM_RESPONSE` when using BQAA input). |
| `retrieval_context` | `ARRAY<STRING>` | Automatic from BQAA | Grounding context or tool outputs retrieved by the agent (automatically extracted from `TOOL_COMPLETED` / `TOOL_ERROR`—plus prior-turn user-assistant dialogue for multi-turn traces—when using `deepeval_parse_trace`; required for `faithfulness`, `contextual_relevancy`, `contextual_precision`, `contextual_recall`, with `Prior turn N:` entries filtered out for `contextual_*`). |
| `context` | `ARRAY<STRING>` | Conditional | Ground-truth context (defaults to `retrieval_context` excluding `Prior turn N:` entries if omitted, or the extracted agent instruction when no tool outputs exist; required for `hallucination`). |
| `expected_output` | `STRING` | Conditional | Reference/golden answer (required for `contextual_precision` and `contextual_recall`). |
| `tools_called` | `ARRAY<STRUCT>` | Automatic from BQAA | Tools invoked by the agent (automatically extracted from `TOOL_STARTING` / `TOOL_COMPLETED` / `TOOL_ERROR` when using BQAA input; used by `task_completion` and `tool_correctness`). |
| `expected_tools` | `ARRAY<STRUCT>` | Conditional | Expected tool calls (required for `tool_correctness`). |
| `evaluation_steps` | `ARRAY<STRING>` | Optional | Explicit scoring steps for `geval` / `conversational_geval` (skips Pass 1 step generation and evaluates in a single LLM pass). |
| `root_agent` | `STRING` | Optional (in `deepeval_parse_trace`) | Filters BQAA events to invocations containing the specified root coordinator agent name (excluding internal `AgentTool` sub-agent events while preserving `transfer_to_agent` handoffs and shared-invocation workflow agents; keep `t.attributes` in the projection so single-turn `GROUP BY session_id, invocation_id` queries retain follow-up turns logged under a transferred sub-agent via `attributes.root_agent_name`; returns `NULL` when no invocation in the group matches). |
| `output_response_field` | `STRING` | Optional (in `deepeval_parse_trace`) | Unwraps a structured JSON response field (such as `'response'` when ADK `output_schema` emits `{"response": "...", "usefulness": 0.95}`) and strips JSON-schema output constraints from `chatbot_role` so `role_adherence` does not penalize unwrapped Markdown. Also auto-detected when the trace-extracted agent instruction mentions `output_schema` / `AgentResponse` / `strictly as a JSON object` and `actual_output` is a multi-key JSON object containing a `"response"` string (pass `''` to opt out). |
| `judge_model` | `STRING` | Optional | Primary Gemini judge model ID (defaults to `'gemini-3.8-flash'`). |
| `fallback_judge_model` | `STRING` | Optional | Fallback Gemini judge model ID used on final retry (defaults to `'gemini-2.5-flash'`). |
| `location` | `STRING` | Optional | Per-row Gemini endpoint location override (defaults to `'global'` for Gemini 3.x models and the configured regional endpoint for regional models such as `'gemini-2.5-flash'`). |

#### Supported Single-Turn Metrics (`metrics`)

- `'answer_relevancy'`: Measures whether `actual_output` directly addresses
  `input`. *(Note: `answer_relevancy` only inspects `input` and
  `actual_output` without `chatbot_role`; if your agent's system prompt
  mandates proactive boilerplate such as governance contacts, documentation
  links, or closing disclaimers, prefer `'task_completion'` or `'geval'` so
  required footers are not penalized as irrelevant statements.)*
- `'faithfulness'`: Checks whether claims in `actual_output` are grounded in
  `retrieval_context` without hallucinations.
- `'task_completion'`: Evaluates whether the agent accomplished the user's goal
  across its `input`, `tools_called`, and `actual_output`.
- `'tool_correctness'`: Compares `tools_called` against `expected_tools`.
- `'contextual_relevancy'`: Measures the relevance of `retrieval_context` to
  `input`.
- `'contextual_precision'`: Evaluates whether relevant nodes in
  `retrieval_context` are ranked higher than irrelevant ones (requires
  `expected_output`).
- `'contextual_recall'`: Evaluates whether `retrieval_context` contains all
  facts needed to produce `expected_output`.
- `'hallucination'`: Measures contradiction against `context` (lower score =
  less hallucination).
- `'geval:<Name>:<Criteria>'` (or single-pass
  `'geval:<Name>:steps:<Step 1>;<Step 2>'`): Custom LLM-as-a-Judge rubric using
  DeepEval's built-in `GEval` class (an implementation of the academic G-Eval
  scoring framework, unrelated to GEAP; for example,
  `'geval:Technical Precision:Evaluate whether the response cites exact IDs and
  numbers from the tool output.'`). Prefixing `<Criteria>` with `steps:` (or
  supplying `evaluation_steps` in `test_case_json`) passes explicit
  `evaluation_steps` to `GEval`, skipping LLM step-generation and cutting
  evaluation from 2 LLM passes to 1 pass. When every entry in `tools_called` has
  an output, the UDF automatically strips auto-extracted
  `Tool <name> returned: ...` entries from `retrieval_context` (and omits
  `RETRIEVAL_CONTEXT` altogether when no other retrieval context such as
  prior-turn dialogue remains) to avoid sending duplicate tool payloads in the
  prompt.

---

### Mode B: Multi-Turn Conversational Evaluation (`ConversationalTestCase`)

When `test_case_json` is a multi-turn trace produced by `deepeval_parse_trace`
(grouped by `session_id`) or a `STRUCT` with a `turns` array,
`deepeval_evaluate` constructs a DeepEval `ConversationalTestCase`.

#### Supported JSON Fields
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `events` (or top-level BQAA array in `deepeval_parse_trace`) | `ARRAY<STRUCT>` | BQAA Input | Array of the 9 required BQAA event columns across a session (`ARRAY_AGG(STRUCT(t.timestamp, t.event_type, t.invocation_id, t.agent, t.span_id, t.error_message, t.attributes, t.content, t.content_parts) ORDER BY t.timestamp)`), automatically grouped by `deepeval_parse_trace` into ordered `turns` by `invocation_id`. |
| `turns` | `ARRAY<STRUCT>` | Required in `test_case_json` | Ordered array of conversation turns (automatically populated by `deepeval_parse_trace`, or supplied as explicit `{role, content, tools_called, retrieval_context}` / `{input, actual_output, tools_called, retrieval_context}` structs). |
| `chatbot_role` (or `instruction`) | `STRING` | Optional | Persona/system role description (automatically extracted from `AGENT_STARTING` / `LLM_REQUEST` by `deepeval_parse_trace`; required for `role_adherence`, defaults to `'Helpful AI Assistant'`). |
| `scenario` | `STRING` | Optional | Scenario description for the conversation. |
| `expected_outcome` | `STRING` | Optional | Expected outcome of the multi-turn session. |
| `evaluation_steps` | `ARRAY<STRING>` | Optional | Explicit scoring steps for `conversational_geval` (skips Pass 1 step generation and evaluates in a single LLM pass). |
| `root_agent` | `STRING` | Optional (in `deepeval_parse_trace`) | Filters BQAA events to invocations containing the specified root coordinator agent name (excluding internal `AgentTool` sub-agent events while preserving `transfer_to_agent` handoffs; returns `NULL` when no invocation in the session contains a `root_agent` row). |
| `output_response_field` | `STRING` | Optional (in `deepeval_parse_trace`) | Unwraps a structured JSON response field across all turns and strips JSON-schema output constraints from `chatbot_role` so `role_adherence` does not penalize unwrapped Markdown. Also auto-detected when the trace-extracted agent instruction mentions `output_schema` / `AgentResponse` / `strictly as a JSON object` and `actual_output` is a multi-key JSON object containing a `"response"` string (pass `''` to opt out). |
| `judge_model` | `STRING` | Optional | Primary Gemini judge model ID (defaults to `'gemini-3.8-flash'`). |
| `fallback_judge_model` | `STRING` | Optional | Fallback Gemini judge model ID used on final retry (defaults to `'gemini-2.5-flash'`). |
| `location` | `STRING` | Optional | Per-row Gemini endpoint location override (defaults to `'global'` for Gemini 3.x models and the configured regional endpoint for regional models such as `'gemini-2.5-flash'`). |

#### Supported Multi-Turn Metrics (`metrics`)

- `'turn_relevancy'`: Evaluates whether each assistant turn stays relevant to
  the preceding conversation context.
- `'knowledge_retention'`: Checks whether the assistant retains facts and
  context established in earlier turns without forgetting or re-asking.
- `'conversation_completeness'`: Evaluates whether all user intentions across
  the multi-turn session were satisfied by the end of the conversation.
- `'role_adherence'`: Measures whether the assistant adheres to `chatbot_role`
  across all turns.
- `'conversational_geval:<Name>:<Criteria>'` (or single-pass
  `'conversational_geval:<Name>:steps:<Step 1>;<Step 2>'`): Custom multi-turn
  G-Eval rubric (`deepeval.metrics.ConversationalGEval`) evaluated over the full
  conversation trajectory.

---

## Step 4: End-to-End Walkthrough on a Public BQAA Table (`agent_events_v2`)

The file [`agent_events_v2_eval_queries.sql`](agent_events_v2_eval_queries.sql)
demonstrates how to parse and evaluate real
[Agent Development Kit (ADK)](https://google.github.io/adk-docs/) BigQuery
Agent Analytics (BQAA) logs from the public BigQuery dataset
**`bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2`** using
`deepeval_parse_trace` and `deepeval_evaluate`.

> **Note on Cost & Slot Optimization:**
> - `deepeval_parse_trace` runs offline in Python with zero Gemini quota cost,
>   allowing you to materialize and inspect parsed traces in a table before
>   invoking `deepeval_evaluate`.
> - Grouping, ordering, and applying `LIMIT` in a CTE **before** calling
>   `deepeval_parse_trace` (and projecting only the 9 required BQAA columns
>   rather than `ARRAY_AGG(t)`) prevents BigQuery from running the Python UDF on
>   all partitions prior to `LIMIT` and substantially reduces serialized JSON
>   payload size and slot consumption.

You can run the full workflow in
[`agent_events_v2_eval_queries.sql`](agent_events_v2_eval_queries.sql) at once
from the command line, or run each step individually below:

```bash
sed \
  -e "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" \
  -e "s/YOUR_DATASET_ID/${DATASET_ID}/g" \
  agent_events_v2_eval_queries.sql \
  | bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false
```

### 1. Parse Raw BQAA Events (`deepeval_parse_trace`) & Run Single-Turn Evaluation (`deepeval_evaluate`)

Group the BQAA table by `(session_id, invocation_id)` in a CTE, parse the event
array into a trace table with `deepeval_parse_trace`, and evaluate with
`deepeval_evaluate`:

```bash
bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false <<EOF
-- Materialize parsed invocation traces (runs offline without spending Gemini quota)
CREATE OR REPLACE TABLE \`${PROJECT_ID}.${DATASET_ID}.parsed_invocation_traces\` AS
WITH grouped_invocations AS (
  SELECT
    ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
    t.session_id,
    t.invocation_id,
    MIN(t.timestamp) AS started_at,
    ARRAY_AGG(
      STRUCT(
        t.timestamp,
        t.event_type,
        t.invocation_id,
        t.agent,
        t.span_id,
        t.error_message,
        t.attributes,
        t.content,
        t.content_parts
      )
      ORDER BY t.timestamp
    ) AS events
  FROM \`bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2\` AS t
  -- Optional for AgentTool multi-agent apps (do not enable for sub_agents / transfer_to_agent handoffs):
  -- WHERE t.agent = 'YOUR_ROOT_AGENT'
  GROUP BY t.session_id, t.invocation_id
)
SELECT
  agent,
  session_id,
  invocation_id,
  started_at,
  \`${PROJECT_ID}.${DATASET_ID}.deepeval_parse_trace\`(
    TO_JSON_STRING(events)
  ) AS parsed_trace
FROM grouped_invocations;

-- Evaluate the parsed invocation traces with DeepEval metrics
SELECT
  agent,
  session_id,
  invocation_id,
  \`${PROJECT_ID}.${DATASET_ID}.deepeval_evaluate\`(
    parsed_trace,
    [
      'answer_relevancy',
      'faithfulness',
      'task_completion',
      'geval:Technical Precision:Evaluate whether the response provides exact names, IDs, or numbers grounded in the tool output without hallucinating.'
    ]
  ) AS eval_results
FROM \`${PROJECT_ID}.${DATASET_ID}.parsed_invocation_traces\`
ORDER BY started_at
LIMIT 10;
EOF
```

### 2. Run Multi-Turn Session Evaluation (`deepeval_parse_trace` + `deepeval_evaluate`)

Group the BQAA table by `session_id` and apply `LIMIT` in `grouped_sessions`,
parse the multi-turn session trace with `deepeval_parse_trace`, and pass the
parsed trace to `deepeval_evaluate`:

```bash
bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false <<EOF
WITH grouped_sessions AS (
  SELECT
    ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
    t.session_id,
    COUNT(DISTINCT t.invocation_id) AS num_invocations,
    ARRAY_AGG(
      STRUCT(
        t.timestamp,
        t.event_type,
        t.invocation_id,
        t.agent,
        t.span_id,
        t.error_message,
        t.attributes,
        t.content,
        t.content_parts
      )
      ORDER BY t.timestamp
    ) AS events
  FROM \`bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2\` AS t
  GROUP BY t.session_id
  HAVING COUNT(DISTINCT t.invocation_id) > 1
  ORDER BY t.session_id
  LIMIT 10
),
parsed_session_traces AS (
  SELECT
    agent,
    session_id,
    num_invocations,
    \`${PROJECT_ID}.${DATASET_ID}.deepeval_parse_trace\`(
      TO_JSON_STRING(events)
    ) AS parsed_trace
  FROM grouped_sessions
)
SELECT
  agent,
  session_id,
  num_invocations,
  \`${PROJECT_ID}.${DATASET_ID}.deepeval_evaluate\`(
    parsed_trace,
    [
      'turn_relevancy',
      'knowledge_retention',
      'conversation_completeness',
      'role_adherence',
      'conversational_geval:Cross-Turn Context Carryover:Evaluate whether the assistant remembers the project_id discussed in earlier turns when the user asks follow-up questions about a dataset or table without repeating the project_id.'
    ]
  ) AS multi_turn_results
FROM parsed_session_traces
ORDER BY session_id;
EOF
```

---

## Files in This Directory

- [`install.sh`](install.sh): One-command installer script (`gcloud` + `bq`)
  that provisions APIs, the BigQuery dataset, the Cloud Resource Connection,
  IAM permissions (`roles/aiplatform.user`), and the `deepeval_parse_trace` and
  `deepeval_evaluate` Python UDFs.
- [`deepeval_udfs.sql`](deepeval_udfs.sql): `CREATE OR REPLACE FUNCTION` DDL for
  the `deepeval_parse_trace` and `deepeval_evaluate` BigQuery Managed Python
  UDFs.
- [`agent_events_v2_eval_queries.sql`](agent_events_v2_eval_queries.sql):
  Ready-to-run SQL queries for parsing and evaluating single-turn and
  multi-turn BQAA agent traces from
  `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2`.
