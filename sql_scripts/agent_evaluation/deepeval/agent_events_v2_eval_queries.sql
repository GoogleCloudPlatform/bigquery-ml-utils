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
-- Direct BQAA Agent Evaluation Queries for DeepEval BigQuery UDF (`deepeval_evaluate`)
--
-- Public BQAA Sample Dataset:
--   `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2`
-- Target Project/Dataset Placeholders:
--   `YOUR_PROJECT_ID.YOUR_DATASET_ID` (default dataset in install.sh: agent_analytics)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Single-Turn Evaluation Directly on a BQAA Table (`agent_events_v2`)
-- -----------------------------------------------------------------------------
-- Groups raw BigQuery Agent Analytics (BQAA) event rows by
-- `(session_id, invocation_id)` and passes `TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))`
-- directly to `deepeval_evaluate`:
--   - answer_relevancy
--   - faithfulness
--   - task_completion
--   - geval:<Name>:<Criteria>
SELECT
  ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
  t.session_id,
  t.invocation_id,
  `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_evaluate`(
    TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp)),
    [
      'answer_relevancy',
      'faithfulness',
      'task_completion',
      'geval:Technical Precision:Evaluate whether the response provides exact names, IDs, or numbers grounded in the tool output without hallucinating.'
    ]
  ) AS eval_results
FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
GROUP BY t.session_id, t.invocation_id
ORDER BY MIN(t.timestamp)
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 2. Multi-Turn Session Evaluation Directly on a BQAA Table (`agent_events_v2`)
-- -----------------------------------------------------------------------------
-- Groups raw BQAA event rows by `session_id` across multiple turns and passes
-- `TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp))` directly to
-- `deepeval_evaluate`:
--   - turn_relevancy
--   - knowledge_retention
--   - conversation_completeness
--   - role_adherence
--   - conversational_geval:<Name>:<Criteria>
SELECT
  ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
  t.session_id,
  COUNT(DISTINCT t.invocation_id) AS num_invocations,
  `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_evaluate`(
    TO_JSON_STRING(ARRAY_AGG(t ORDER BY t.timestamp)),
    [
      'turn_relevancy',
      'knowledge_retention',
      'conversation_completeness',
      'role_adherence',
      'conversational_geval:Cross-Turn Context Carryover:Evaluate whether the assistant remembers the project_id discussed in earlier turns when the user asks follow-up questions about a dataset or table without repeating the project_id.'
    ]
  ) AS multi_turn_results
FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
GROUP BY t.session_id
HAVING COUNT(DISTINCT t.invocation_id) > 1
ORDER BY t.session_id
LIMIT 10;
