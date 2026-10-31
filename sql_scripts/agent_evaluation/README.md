# Evaluate AI Agent Traces in BigQuery Using Managed Python UDFs

This directory contains ready-to-deploy **BigQuery Managed Python UDF**
integrations and SQL query templates for evaluating **Single-Turn** and
**Multi-Turn** AI agent execution traces directly inside BigQuery.

The integration authenticates using Application Default Credentials
(`google.auth.default()`) attached to a BigQuery Cloud Resource Connection with
the **Agent Platform User** role (`roles/aiplatform.user`), so **no API keys or
Secret Manager secrets are required**.

---

## Available Integrations

| Directory | UDF Names | Evaluation Backend | Supported Capabilities |
| :--- | :--- | :--- | :--- |
| **[`deepeval/`](deepeval/)** | `deepeval_parse_trace`, `deepeval_evaluate` | [DeepEval](https://github.com/confident-ai/deepeval) with [Gemini on Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models) (`gemini-2.5-flash`) | Offline BQAA / `AgentData` trace normalization (`deepeval_parse_trace`) plus single-turn `LLMTestCase` metrics (`answer_relevancy`, `faithfulness`, `task_completion`, `tool_correctness`, `contextual_relevancy`, `contextual_precision`, `contextual_recall`, `hallucination`, `geval`) and multi-turn `ConversationalTestCase` metrics (`turn_relevancy`, `knowledge_retention`, `conversation_completeness`, `role_adherence`, `conversational_geval`). |

---

## UDF Signatures, BQAA Input Format & Output Schema

`deepeval_parse_trace` normalizes **BigQuery Agent Analytics (BQAA)** event
rows (`BigQueryAgentAnalyticsPlugin`) via
`TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))` (as well as canonical
`AgentData` JSON) into a trace JSON string without spending Gemini quota, and
`deepeval_evaluate` evaluates the parsed trace:

```sql
-- 1. Offline Trace Parser
`<PROJECT_ID>.<DATASET_ID>.deepeval_parse_trace`(
  events_json STRING
)
RETURNS STRING

-- 2. Online DeepEval Metric Evaluator
`<PROJECT_ID>.<DATASET_ID>.deepeval_evaluate`(
  test_case_json STRING,
  metrics ARRAY<STRING>
)
RETURNS ARRAY<STRUCT<
  metric STRING,
  score FLOAT64,
  passed BOOL,
  reason STRING
>>
```

### Usage on a BQAA Table (`agent_events` / `agent_events_v2`)

- **Single-Turn Evaluation (`GROUP BY session_id, invocation_id`)**:
  ```sql
  -- Step 1: Materialize parsed invocation traces (runs offline without spending Gemini quota)
  CREATE OR REPLACE TABLE `<PROJECT_ID>.<DATASET_ID>.parsed_invocation_traces` AS
  SELECT
    t.session_id,
    t.invocation_id,
    `<PROJECT_ID>.<DATASET_ID>.deepeval_parse_trace`(
      TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))
    ) AS parsed_trace
  FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
  GROUP BY t.session_id, t.invocation_id;

  -- Step 2: Evaluate parsed invocation traces
  SELECT
    session_id,
    invocation_id,
    `<PROJECT_ID>.<DATASET_ID>.deepeval_evaluate`(
      parsed_trace,
      ['answer_relevancy', 'faithfulness', 'task_completion']
    ) AS eval_results
  FROM `<PROJECT_ID>.<DATASET_ID>.parsed_invocation_traces`
  LIMIT 10;
  ```
- **Multi-Turn Session Evaluation (`GROUP BY session_id`)**:
  ```sql
  WITH parsed_session_traces AS (
    SELECT
      t.session_id,
      COUNT(DISTINCT t.invocation_id) AS num_invocations,
      `<PROJECT_ID>.<DATASET_ID>.deepeval_parse_trace`(
        TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))
      ) AS parsed_trace
    FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
    GROUP BY t.session_id
    HAVING COUNT(DISTINCT t.invocation_id) > 1
    LIMIT 10
  )
  SELECT
    session_id,
    num_invocations,
    `<PROJECT_ID>.<DATASET_ID>.deepeval_evaluate`(
      parsed_trace,
      ['turn_relevancy', 'knowledge_retention', 'conversation_completeness']
    ) AS multi_turn_results
  FROM parsed_session_traces;
  ```

---

## Quick Start

[`deepeval/`](deepeval/) includes:

- **`install.sh`**: One-command installer (`gcloud` + `bq`) that enables
  required APIs, creates the BigQuery dataset and Cloud Resource Connection,
  grants the **Agent Platform User** role (`roles/aiplatform.user`) to the
  connection service account, and deploys the `deepeval_parse_trace` and
  `deepeval_evaluate` Python UDFs.
- **`deepeval_udfs.sql`**: `CREATE OR REPLACE FUNCTION` DDL for the BigQuery
  Managed Python UDFs.
- **`agent_events_v2_eval_queries.sql`**: End-to-end SQL queries that evaluate
  [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) BigQuery
  Agent Analytics (BQAA) logs directly from the public dataset
  `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2`.
- **`README.md`**: Step-by-step setup, CLI smoke tests, metric reference, and
  walkthrough instructions.

See [`deepeval/README.md`](deepeval/README.md) to get started.

---

## Running Unit Tests

The unit tests in [`tests/`](tests/) use only the Python standard library
(`unittest`) with lightweight fakes for hermetic CI (no network calls or
external packages required). When upgrading the pinned package versions in
[`deepeval/deepeval_udfs.sql`](deepeval/deepeval_udfs.sql) (such as
`deepeval==3.8.0` or `google-genai==1.15.0`), run a manual end-to-end smoke
test in BigQuery in addition to the unit tests:

```bash
python3 -m unittest discover -s sql_scripts/agent_evaluation/tests -p "*_test.py" -v
```
