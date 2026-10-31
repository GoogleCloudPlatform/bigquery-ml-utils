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
-- BQAA Agent Evaluation Queries for DeepEval BigQuery UDFs
-- (`deepeval_parse_trace` + `deepeval_evaluate`)
--
-- Public BQAA Sample Dataset:
--   `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2`
-- Target Project/Dataset Placeholders:
--   `YOUR_PROJECT_ID.YOUR_DATASET_ID` (default dataset in install.sh: agent_analytics)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Parse Raw BQAA Events into a Trace Table (or View) with `deepeval_parse_trace`
-- -----------------------------------------------------------------------------
-- Materializing a TABLE (`CREATE OR REPLACE TABLE`) is recommended for benchmark
-- datasets or repeated evaluation runs because BigQuery does not cache Python
-- UDF results across queries; a TABLE runs `deepeval_parse_trace` once and lets
-- you inspect the parsed traces before spending Gemini quota. (Use
-- `CREATE OR REPLACE VIEW` instead if you prefer a zero-storage virtual view
-- over a live streaming `agent_events` table.)
CREATE OR REPLACE TABLE `YOUR_PROJECT_ID.YOUR_DATASET_ID.parsed_invocation_traces` AS
SELECT
  ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
  t.session_id,
  t.invocation_id,
  MIN(t.timestamp) AS started_at,
  `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_parse_trace`(
    TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))
  ) AS parsed_trace
FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
GROUP BY t.session_id, t.invocation_id;

-- Inspect the parsed traces without spending Gemini quota:
SELECT
  agent,
  session_id,
  invocation_id,
  JSON_VALUE(parsed_trace, '$.input') AS user_input,
  JSON_VALUE(parsed_trace, '$.actual_output') AS actual_output,
  JSON_QUERY(parsed_trace, '$.tools_called') AS tools_called
FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.parsed_invocation_traces`
ORDER BY started_at
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 2. Single-Turn Evaluation on Parsed Invocation Traces (`deepeval_evaluate`)
-- -----------------------------------------------------------------------------
-- Evaluates each parsed invocation trace against:
--   - answer_relevancy
--   - faithfulness
--   - task_completion
--   - geval:<Name>:<Criteria>
SELECT
  agent,
  session_id,
  invocation_id,
  `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_evaluate`(
    parsed_trace,
    [
      'answer_relevancy',
      'faithfulness',
      'task_completion',
      'geval:Technical Precision:Evaluate whether the response provides exact names, IDs, or numbers grounded in the tool output without hallucinating.'
    ]
  ) AS eval_results
FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.parsed_invocation_traces`
ORDER BY started_at
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 3. Multi-Turn Session Evaluation (`deepeval_parse_trace` + `deepeval_evaluate`)
-- -----------------------------------------------------------------------------
-- Groups raw BQAA event rows by `session_id` across multiple turns, parses the
-- multi-turn session trace with `deepeval_parse_trace`, and evaluates with
-- `deepeval_evaluate`:
--   - turn_relevancy
--   - knowledge_retention
--   - conversation_completeness
--   - role_adherence
--   - conversational_geval:<Name>:<Criteria>
WITH parsed_session_traces AS (
  SELECT
    ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
    t.session_id,
    COUNT(DISTINCT t.invocation_id) AS num_invocations,
    `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_parse_trace`(
      TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))
    ) AS parsed_trace
  FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
  GROUP BY t.session_id
  HAVING COUNT(DISTINCT t.invocation_id) > 1
  ORDER BY t.session_id
  LIMIT 10
)
SELECT
  agent,
  session_id,
  num_invocations,
  `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_evaluate`(
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
