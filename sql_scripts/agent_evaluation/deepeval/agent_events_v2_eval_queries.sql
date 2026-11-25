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
--
-- Performance Tip: Grouping in a CTE and projecting only the 9 BQAA columns
-- read by the parser (rather than `ARRAY_AGG(t ORDER BY t.timestamp)`)
-- substantially reduces serialized JSON payload size and slot consumption; for
-- sampled queries (such as Section 3's `grouped_sessions`), applying
-- `ORDER BY ... LIMIT` inside the grouping CTE also ensures BigQuery applies
-- `LIMIT` before executing `deepeval_parse_trace`.
-- Shared-Table Isolation: When multiple agents write to a shared `agent_events`
-- table, `session_id` can collide across agents and `t.agent` is a mutable
-- display or sub-agent name. Computing `agent_scope_id` once in `scoped_events`
-- using an immutable-first fallback chain keeps colliding sessions separate
-- while keeping `transfer_to_agent` sub-agent handoffs unified inside a single
-- trace, and stays compatible with 17-column tables without a top-level
-- `app_name` column.
CREATE OR REPLACE TABLE `YOUR_PROJECT_ID.YOUR_DATASET_ID.parsed_invocation_traces` AS
WITH scoped_events AS (
  SELECT
    t.*,
    COALESCE(
      IF(
        STARTS_WITH(JSON_VALUE(t.attributes, '$.adk.app_name'), 'projects/'),
        JSON_VALUE(t.attributes, '$.adk.app_name'),
        NULL
      ),
      IF(
        STARTS_WITH(JSON_VALUE(t.attributes, '$.session_metadata.app_name'), 'projects/'),
        JSON_VALUE(t.attributes, '$.session_metadata.app_name'),
        NULL
      ),
      NULLIF(JSON_VALUE(t.attributes, '$.session_metadata.state."data-agent-id"'), ''),
      NULLIF(JSON_VALUE(t.attributes, '$.adk.app_name'), ''),
      NULLIF(JSON_VALUE(t.attributes, '$.session_metadata.app_name'), ''),
      NULLIF(JSON_VALUE(t.attributes, '$.root_agent_name'), ''),
      NULLIF(t.agent, '')
    ) AS agent_scope_id
  FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
),
grouped_invocations AS (
  SELECT
    t.agent_scope_id,
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
  FROM scoped_events AS t
  -- Optional: uncomment to isolate a single agent in a shared table (for
  -- AgentTool delegates, pass `root_agent` in the `deepeval_parse_trace`
  -- wrapper STRUCT instead of filtering `t.agent` when `sub_agents` /
  -- `transfer_to_agent` handoffs are used):
  -- WHERE agent_scope_id = 'YOUR_AGENT_SCOPE_ID'
  GROUP BY t.agent_scope_id, t.session_id, t.invocation_id
)
SELECT
  agent_scope_id,
  agent,
  session_id,
  invocation_id,
  started_at,
  `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_parse_trace`(
    TO_JSON_STRING(events)
  ) AS parsed_trace
FROM grouped_invocations;

-- Inspect the parsed traces without spending Gemini quota:
SELECT
  agent_scope_id,
  agent,
  session_id,
  invocation_id,
  JSON_VALUE(parsed_trace, '$.input') AS user_input,
  JSON_VALUE(parsed_trace, '$.actual_output') AS actual_output,
  JSON_QUERY(parsed_trace, '$.tools_called') AS tools_called
FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.parsed_invocation_traces`
-- Optional: isolate a single agent in a shared trace table:
-- WHERE agent_scope_id = 'YOUR_AGENT_SCOPE_ID'
ORDER BY started_at
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 2. Single-Turn Evaluation on Parsed Invocation Traces (`deepeval_evaluate`)
-- -----------------------------------------------------------------------------
-- Evaluates each parsed invocation trace against:
--   - answer_relevancy
--   - faithfulness
--   - task_completion
--   - geval:<Name>:<Criteria> (or single-pass `geval:<Name>:steps:<Step 1>;<Step 2>`)
-- Each element in `eval_results` is a STRUCT<metric, score, passed, reason, judge_input_tokens, judge_output_tokens>.
SELECT
  agent_scope_id,
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
-- Optional: isolate a single agent in a shared trace table:
-- WHERE agent_scope_id = 'YOUR_AGENT_SCOPE_ID'
ORDER BY started_at
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 3. Multi-Turn Session Evaluation (`deepeval_parse_trace` + `deepeval_evaluate`)
-- -----------------------------------------------------------------------------
-- Groups raw BQAA event rows by `(agent_scope_id, session_id)` across multiple
-- turns and applies `ORDER BY ... LIMIT` in `grouped_sessions` BEFORE invoking
-- `deepeval_parse_trace`, then evaluates with `deepeval_evaluate`:
--   - turn_relevancy
--   - knowledge_retention (extracts personal user-profile facts in deepeval==3.8.0;
--     returns NULL / N/A on technical Q&A sessions without personal profile facts)
--   - conversation_completeness
--   - role_adherence
--   - conversational_geval:<Name>:steps:<Step 1>;<Step 2> (recommended single-pass
--     rubric for cross-turn context retention in analytical / data Q&A sessions)
WITH scoped_events AS (
  SELECT
    t.*,
    COALESCE(
      IF(
        STARTS_WITH(JSON_VALUE(t.attributes, '$.adk.app_name'), 'projects/'),
        JSON_VALUE(t.attributes, '$.adk.app_name'),
        NULL
      ),
      IF(
        STARTS_WITH(JSON_VALUE(t.attributes, '$.session_metadata.app_name'), 'projects/'),
        JSON_VALUE(t.attributes, '$.session_metadata.app_name'),
        NULL
      ),
      NULLIF(JSON_VALUE(t.attributes, '$.session_metadata.state."data-agent-id"'), ''),
      NULLIF(JSON_VALUE(t.attributes, '$.adk.app_name'), ''),
      NULLIF(JSON_VALUE(t.attributes, '$.session_metadata.app_name'), ''),
      NULLIF(JSON_VALUE(t.attributes, '$.root_agent_name'), ''),
      NULLIF(t.agent, '')
    ) AS agent_scope_id
  FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
),
grouped_sessions AS (
  SELECT
    t.agent_scope_id,
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
  FROM scoped_events AS t
  -- Optional: uncomment to isolate a single agent in a shared table:
  -- WHERE agent_scope_id = 'YOUR_AGENT_SCOPE_ID'
  GROUP BY t.agent_scope_id, t.session_id
  HAVING COUNT(DISTINCT t.invocation_id) > 1
  ORDER BY t.session_id, t.agent_scope_id
  LIMIT 10
),
parsed_session_traces AS (
  SELECT
    agent_scope_id,
    agent,
    session_id,
    num_invocations,
    `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_parse_trace`(
      TO_JSON_STRING(events)
    ) AS parsed_trace
  FROM grouped_sessions
)
SELECT
  agent_scope_id,
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
      'conversational_geval:Cross-Turn Context Carryover:steps:Identify entities, project IDs, dataset names, or filters established in earlier turns;Verify follow-up assistant turns retain that context without re-asking or losing scope;Penalize any follow-up turn that drops or re-asks previously established context'
    ]
  ) AS multi_turn_results
FROM parsed_session_traces
ORDER BY session_id, agent_scope_id;


-- -----------------------------------------------------------------------------
-- 4. On-Demand Incremental Multi-Turn Session Evaluation (`MERGE`)
-- -----------------------------------------------------------------------------
-- Production pattern for scheduled or on-demand batch evaluation:
--   1. Prunes BQAA table partitions using `scan_floor` (1-day lookback before the
--      latest evaluated event timestamp, or epoch on first run) and bounds
--      full-session history reconstruction via `history_floor` (30-day lookback
--      before `scan_floor`, compatible with `require_partition_filter = TRUE`),
--      isolating shared multi-agent tables by `(agent_scope_id, session_id)` and
--      skipping rows with `NULL` `session_id` or `agent_scope_id`.
--   2. Applies a 5-minute settle window so active in-flight sessions are skipped
--      until idle, anti-joins `session_eval_results` to identify new, extended,
--      or retryable errored (`Error: ...` with `eval_attempts < 3`) sessions,
--      selects up to `LIMIT 100` multi-turn sessions before reading heavy event
--      payloads (`content`, `content_parts`), prioritizing new/extended sessions
--      ahead of retries, and only then aggregates full event payloads before
--      invoking `deepeval_parse_trace` and `deepeval_evaluate`.
CREATE TABLE IF NOT EXISTS `YOUR_PROJECT_ID.YOUR_DATASET_ID.session_eval_results` (
  agent_scope_id STRING,
  agent STRING,
  session_id STRING,
  started_at TIMESTAMP,
  last_event_at TIMESTAMP,
  num_invocations INT64,
  eval_attempts INT64,
  eval_results ARRAY<STRUCT<
    metric STRING,
    score FLOAT64,
    passed BOOL,
    reason STRING,
    judge_input_tokens INT64,
    judge_output_tokens INT64
  >>
);

BEGIN
  DECLARE scan_floor TIMESTAMP DEFAULT (
    SELECT
      TIMESTAMP_SUB(
        COALESCE(MAX(last_event_at), TIMESTAMP('1970-01-01')),
        INTERVAL 1 DAY
      )
    FROM `YOUR_PROJECT_ID.YOUR_DATASET_ID.session_eval_results`
    -- Optional: uncomment to isolate a single agent in a shared results table:
    -- WHERE agent_scope_id = 'YOUR_AGENT_SCOPE_ID'
  );
  DECLARE history_floor TIMESTAMP DEFAULT TIMESTAMP_SUB(scan_floor, INTERVAL 30 DAY);

  MERGE `YOUR_PROJECT_ID.YOUR_DATASET_ID.session_eval_results` AS target
  USING (
    WITH scoped_events AS (
      SELECT
        t.*,
        COALESCE(
          IF(
            STARTS_WITH(JSON_VALUE(t.attributes, '$.adk.app_name'), 'projects/'),
            JSON_VALUE(t.attributes, '$.adk.app_name'),
            NULL
          ),
          IF(
            STARTS_WITH(JSON_VALUE(t.attributes, '$.session_metadata.app_name'), 'projects/'),
            JSON_VALUE(t.attributes, '$.session_metadata.app_name'),
            NULL
          ),
          NULLIF(JSON_VALUE(t.attributes, '$.session_metadata.state."data-agent-id"'), ''),
          NULLIF(JSON_VALUE(t.attributes, '$.adk.app_name'), ''),
          NULLIF(JSON_VALUE(t.attributes, '$.session_metadata.app_name'), ''),
          NULLIF(JSON_VALUE(t.attributes, '$.root_agent_name'), ''),
          NULLIF(t.agent, '')
        ) AS agent_scope_id
      -- Replace with your partitioned BQAA table (e.g., `YOUR_PROJECT_ID.YOUR_DATASET_ID.agent_events_v2`):
      FROM `bigquery-public-data.agent_analytic_sample_dataset.agent_events_v2` AS t
      WHERE t.timestamp >= history_floor
        AND t.session_id IS NOT NULL
    ),
    filtered_events AS (
      SELECT *
      FROM scoped_events
      -- Optional: uncomment to isolate a single agent in a shared table:
      -- WHERE agent_scope_id = 'YOUR_AGENT_SCOPE_ID'
    ),
    recent_activity AS (
      SELECT
        t.agent_scope_id,
        t.session_id,
        MAX(t.timestamp) AS recent_last_event_at
      FROM filtered_events AS t
      WHERE t.timestamp >= scan_floor
        AND t.agent_scope_id IS NOT NULL
      GROUP BY t.agent_scope_id, t.session_id
      HAVING MAX(t.timestamp) < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
    ),
    changed_sessions AS (
      SELECT
        a.agent_scope_id,
        a.session_id,
        (
          r.session_id IS NOT NULL
          AND a.recent_last_event_at <= r.last_event_at
        ) AS is_error_retry
      FROM recent_activity AS a
      LEFT JOIN `YOUR_PROJECT_ID.YOUR_DATASET_ID.session_eval_results` AS r
        ON r.agent_scope_id = a.agent_scope_id
        AND r.session_id = a.session_id
      WHERE r.session_id IS NULL
        OR a.recent_last_event_at > r.last_event_at
        OR (
          COALESCE(r.eval_attempts, 1) < 3
          AND EXISTS (
            SELECT 1
            FROM UNNEST(r.eval_results) AS e
            WHERE STARTS_WITH(e.reason, 'Error:')
          )
        )
    ),
    selected_sessions AS (
      SELECT
        t.agent_scope_id,
        t.session_id,
        ANY_VALUE(c.is_error_retry) AS is_error_retry,
        MIN(t.timestamp) AS started_at,
        MAX(t.timestamp) AS last_event_at,
        COUNT(DISTINCT t.invocation_id) AS num_invocations
      FROM filtered_events AS t
      INNER JOIN changed_sessions AS c
        ON t.agent_scope_id = c.agent_scope_id
        AND t.session_id = c.session_id
      GROUP BY t.agent_scope_id, t.session_id
      HAVING COUNT(DISTINCT t.invocation_id) > 1
        AND MAX(t.timestamp) < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
      ORDER BY is_error_retry ASC, last_event_at ASC
      LIMIT 100
    ),
    grouped_sessions AS (
      SELECT
        t.agent_scope_id,
        ARRAY_FIRST(ARRAY_AGG(t.agent IGNORE NULLS ORDER BY t.timestamp)) AS agent,
        t.session_id,
        ANY_VALUE(s.started_at) AS started_at,
        ANY_VALUE(s.last_event_at) AS last_event_at,
        ANY_VALUE(s.num_invocations) AS num_invocations,
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
      FROM filtered_events AS t
      INNER JOIN selected_sessions AS s
        ON t.agent_scope_id = s.agent_scope_id
        AND t.session_id = s.session_id
      WHERE t.timestamp <= s.last_event_at
      GROUP BY t.agent_scope_id, t.session_id
    ),
    sessions_to_score AS (
      SELECT
        agent_scope_id,
        agent,
        session_id,
        started_at,
        last_event_at,
        num_invocations,
        `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_parse_trace`(
          TO_JSON_STRING(events)
        ) AS parsed_trace
      FROM grouped_sessions
    ),
    scored_sessions AS (
      SELECT
        agent_scope_id,
        agent,
        session_id,
        started_at,
        last_event_at,
        num_invocations,
        `YOUR_PROJECT_ID.YOUR_DATASET_ID.deepeval_evaluate`(
          parsed_trace,
          [
            'turn_relevancy',
            'conversation_completeness',
            'role_adherence',
            'conversational_geval:Cross-Turn Context Carryover:steps:Identify entities, project IDs, dataset names, or filters established in earlier turns;Verify follow-up assistant turns retain that context without re-asking or losing scope;Penalize any follow-up turn that drops or re-asks previously established context'
          ]
        ) AS eval_results
      FROM sessions_to_score
      WHERE parsed_trace IS NOT NULL
    )
    SELECT
      agent_scope_id,
      agent,
      session_id,
      started_at,
      last_event_at,
      num_invocations,
      eval_results,
      EXISTS (
        SELECT 1
        FROM UNNEST(eval_results) AS e
        WHERE STARTS_WITH(e.reason, 'Error:')
      ) AS has_error
    FROM scored_sessions
  ) AS src
  ON target.agent_scope_id = src.agent_scope_id
    AND target.session_id = src.session_id
  WHEN MATCHED THEN
    UPDATE SET
      agent = src.agent,
      last_event_at = src.last_event_at,
      num_invocations = src.num_invocations,
      eval_attempts = IF(
        src.last_event_at <= target.last_event_at AND src.has_error,
        COALESCE(target.eval_attempts, 1) + 1,
        1
      ),
      eval_results = src.eval_results
  WHEN NOT MATCHED THEN
    INSERT (
      agent_scope_id,
      agent,
      session_id,
      started_at,
      last_event_at,
      num_invocations,
      eval_attempts,
      eval_results
    )
    VALUES (
      src.agent_scope_id,
      src.agent,
      src.session_id,
      src.started_at,
      src.last_event_at,
      src.num_invocations,
      1,
      src.eval_results
    );
END;
