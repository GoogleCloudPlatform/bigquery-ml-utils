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
-- Optional read-only readiness check over eval_sessions and eval_turns.
-- Calls no judge; its only cost is scanning the events table.
--
-- Replace the placeholders before running in BigQuery:
--   YOUR_PROJECT_ID         Project where eval_sessions/eval_turns live
--   YOUR_DATASET_ID         Dataset where eval_sessions/eval_turns live
--   YOUR_EVENTS_PROJECT     Project of the BQAA events table
--   YOUR_EVENTS_DATASET     Dataset of the BQAA events table
--   YOUR_EVENTS_TABLE_NAME  Table name of the BQAA events table
-- =============================================================================

DECLARE start_ts TIMESTAMP DEFAULT TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY);
-- 5 MiB minus 64 KiB headroom, matching geap_evaluate.sql.
DECLARE trace_size_limit_bytes INT64 DEFAULT 5177344;

-- 1. Readiness and trace size summary (sessions and turns).
SELECT
  'sessions' AS kind,
  COUNT(*) AS total_count,
  COUNTIF(eval_ready) AS ready_count,
  COUNTIF(has_truncated_content) AS truncated_count,
  COUNTIF(NOT eval_ready AND NOT has_truncated_content) AS other_not_ready_count,
  APPROX_QUANTILES(BYTE_LENGTH(TO_JSON_STRING(trace)), 2)[SAFE_OFFSET(1)] AS median_trace_bytes,
  MAX(BYTE_LENGTH(TO_JSON_STRING(trace))) AS max_trace_bytes,
  COUNTIF(BYTE_LENGTH(TO_JSON_STRING(trace)) > trace_size_limit_bytes) AS over_limit_count
FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.eval_sessions`(start_ts)
UNION ALL
SELECT
  'turns' AS kind,
  COUNT(*) AS total_count,
  COUNTIF(eval_ready) AS ready_count,
  COUNTIF(has_truncated_content) AS truncated_count,
  COUNTIF(NOT eval_ready AND NOT has_truncated_content) AS other_not_ready_count,
  APPROX_QUANTILES(BYTE_LENGTH(TO_JSON_STRING(trace)), 2)[SAFE_OFFSET(1)] AS median_trace_bytes,
  MAX(BYTE_LENGTH(TO_JSON_STRING(trace))) AS max_trace_bytes,
  COUNTIF(BYTE_LENGTH(TO_JSON_STRING(trace)) > trace_size_limit_bytes) AS over_limit_count
FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.eval_turns`(start_ts);

-- 2. Conversations per root agent (helps spot sub-agent sessions).
SELECT
  root_agent,
  COUNT(*) AS session_count,
  COUNTIF(eval_ready) AS ready_count
FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.eval_sessions`(start_ts)
GROUP BY root_agent
ORDER BY session_count DESC;

-- 3. Schema version label on the events table.
SELECT
  t.table_name,
  REGEXP_EXTRACT(o.option_value, r'STRUCT\("adk_schema_version",\s*"([^"]*)"\)')
    AS adk_schema_version
FROM `YOUR_EVENTS_PROJECT.YOUR_EVENTS_DATASET.INFORMATION_SCHEMA.TABLES` AS t
LEFT JOIN `YOUR_EVENTS_PROJECT.YOUR_EVENTS_DATASET.INFORMATION_SCHEMA.TABLE_OPTIONS` AS o
  ON
    o.table_name = t.table_name
    AND o.option_name = 'labels'
WHERE
  t.table_name = 'YOUR_EVENTS_TABLE_NAME';
