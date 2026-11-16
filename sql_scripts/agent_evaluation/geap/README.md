# Evaluate agent traces in BigQuery with GEAP

This directory provides BigQuery routines that score AI agent execution traces
against Gemini Enterprise Agent Platform (GEAP) evaluation metrics.

---

## 1. What this is, and what it costs

The `geap_evaluate` SQL function wraps an internal Python UDF that connects to
GEAP's evaluation service (`aiplatform.googleapis.com`). Each call sends one
trace and one metric, returning a numeric score, a natural-language explanation,
and fine-grained rubric verdicts.

### Cost and quota

Every row and metric combination issues one evaluation request, or up to 5
attempts if rate limits, server errors, timeouts, or dropped connections trigger
retries. A retry after a timeout can pay for the same judge call twice.

*   **Pricing:** Evaluation calls and model usage are billed to the install
    project according to the published [evaluation service pricing](https://docs.cloud.google.com/gemini-enterprise-agent-platform/pricing)
    and [BigQuery compute pricing](https://cloud.google.com/bigquery/pricing).
*   **Quota:** Evaluation requests consume regional judge quota according to the
    published [evaluation service quotas](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/quotas#eval-quotas).
    Run large scoring runs off-peak or in a separate project so judge calls do
    not compete with production agent traffic for model quota. Score one version
    of each metric rather than every version.
*   **Sample before scoring:** BigQuery can evaluate scalar UDFs across every
    matched row before applying an outer limit. Calling `geap_evaluate`
    directly on a large table with `LIMIT 10` can trigger hundreds of unneeded
    judge calls. Always copy sampled traces into a staging table before scoring.

---

## 2. Terms

*   **GEAP evaluation service:** The evaluation API in
    `aiplatform.googleapis.com`.
*   **Judge:** The Gemini model GEAP uses to evaluate and score traces.
*   **Managed metric:** A predefined agent evaluation metric hosted by GEAP.
*   **ADK and BQAA plugin:** The Agent Development Kit and the plugin that logs
    one BigQuery row per agent event.
*   **Session and turn:** A session is a complete conversation. A turn is one
    user message plus everything the agent did in response.
*   **AgentData:** The trace format GEAP accepts, holding agent declarations
    and an ordered list of turns and events.

---

## 3. Workflow

| Step | Action | Frequency | Notes |
| :--- | :--- | :--- | :--- |
| **1. Install** | Run `install.sh`, then grant IAM | Once per dataset | Deploys routines and the connection |
| **2. Sample traces** | Materialize staging table | Every scoring run | Prevents excess slot and API costs |
| **3. Score traces** | Call `geap_evaluate` | Every scoring run | One call per metric; writes to results table |

### Step 1: Install routines

Create the dataset first, in the location you want to evaluate in. The
installer reads the dataset's location and picks the matching GEAP region; it
never creates the dataset.

```bash
./install.sh [PROJECT_ID] [DATASET_ID] [CONNECTION_ID]
```

All arguments are optional (and can also be set via environment variables):
`PROJECT_ID` defaults to `gcloud config get-value project`, `DATASET_ID` to
`agent_analytics`, and `CONNECTION_ID` to `gemini_enterprise_connection`. The
script enables the required APIs, creates the Cloud Resource connection in the
dataset's location if it is missing, deploys `geap_evaluate`, and runs a smoke
test call.

The script never changes IAM. It prints a grant command and a revoke command
for the connection's service account. Someone with rights on the project must
run the grant command before scoring:

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=serviceAccount:CONNECTION_SERVICE_ACCOUNT \
  --role=roles/aiplatform.user --condition=None
```

Until the grant is in place, every row returns a `PERMISSION_DENIED` status.
`roles/aiplatform.user` is the narrowest role that allows evaluation calls, and
it also allows calling any model in the project. To remove access, run the
printed revoke command (`gcloud projects remove-iam-policy-binding` with the
same arguments).

### Step 2: Sample traces

Sample from the eval views that the ADK analytics plugin creates. The views
exist only when the plugin runs with `create_eval_views=True` (and use its
`view_prefix`, which defaults to `v_`). If you can't enable the views, build a
table that holds one AgentData trace per row yourself. Raw rows from the events
table are individual events, not traces, so don't score them directly.

Materialize a sample of ready conversations from `v_eval_session` for the
multi-turn metrics:

```sql
CREATE OR REPLACE TABLE `my-project.my_dataset.sampled_session_traces` AS
SELECT
  session_id,
  trace AS trace_json
FROM `my-project.my_dataset.v_eval_session`
WHERE eval_ready = TRUE
  AND session_end_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
LIMIT 50;
```

Materialize a sample of ready turns from `v_eval_turn` for the single-turn
metrics:

```sql
CREATE OR REPLACE TABLE `my-project.my_dataset.sampled_turn_traces` AS
SELECT
  session_id,
  invocation_id,
  trace AS trace_json
FROM `my-project.my_dataset.v_eval_turn`
WHERE eval_ready = TRUE
  AND turn_end_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
LIMIT 50;
```

### Step 3: Score traces

Score the sampled traces and store results in a table:

```sql
CREATE OR REPLACE TABLE `my-project.my_dataset.evaluation_results` AS
SELECT
  session_id,
  `my-project.my_dataset.geap_evaluate`(
    trace_json,
    'multi_turn_task_success_v1',
    NULL
  ) AS task_success,
  `my-project.my_dataset.geap_evaluate`(
    trace_json,
    'multi_turn_trajectory_quality_v1',
    NULL
  ) AS trajectory_quality
FROM `my-project.my_dataset.sampled_session_traces`;
```

---

## 4. Metrics

`geap_evaluate` accepts twelve managed metric names, all scored from 0 to 1.

### Single-turn metrics (score a sample from `v_eval_turn`)

| Metric | Versions | What it checks |
| :--- | :--- | :--- |
| Tool use quality | `tool_use_quality_v2`, `tool_use_quality_v1` | Whether tool calls were necessary, received valid arguments, and ran in a sensible order |
| Final response quality | `final_response_quality_v2`, `final_response_quality_v1` | Whether the final response answered the user directly, clearly, and completely |
| Hallucination | `hallucination_v2`, `hallucination_v1` | Whether statements in the response are grounded in tool outputs and retrieved context |
| Safety | `safety_v3`, `safety_v2`, `safety_v1` | Whether the response complies with safety policies and avoids harmful content |

Use the `_v2` or `_v3` names. The single-turn `_v1` names only work in projects
that have used Gemini 2.5 Pro before; in other projects every row returns a
`NOT_FOUND` status that says the judge model was not found.

### Multi-turn metrics (score a sample from `v_eval_session`)

| Metric | What it checks |
| :--- | :--- |
| `multi_turn_task_success_v1` | Whether the agent accomplished the user's primary goal across the conversation |
| `multi_turn_tool_use_quality_v1` | Tool selection consistency and coordination across turns |
| `multi_turn_trajectory_quality_v1` | Conversation flow, progress toward the goal, and absence of repetitive loops |

A single-turn metric on a multi-turn trace returns an `INVALID_ARGUMENT` status
without calling GEAP.

---

## 5. Custom metrics

To evaluate custom criteria, pass a prompt template as the third argument and
set the metric name to `NULL`:

```sql
SELECT
  session_id,
  `my-project.my_dataset.geap_evaluate`(
    trace_json,
    NULL,
    '''
    You are evaluating an agent conversation trace.
    Trace: {agent_data}

    Score the agent on professionalism and politeness from 1.0 to 5.0.
    Output JSON with "score" and "explanation".
    '''
  ) AS professionalism
FROM `my-project.my_dataset.sampled_session_traces`;
```

### Supported placeholders

*   `{agent_data}`: The full trace JSON.
*   `{events}`: Tool calls and responses.
*   `{tool_declarations}`: Tool declarations and schemas.
*   `{developer_instruction}`: System instructions for all agents in the trace.
*   `{prompt}`, `{response}`: The user prompt and final assistant response
    (valid for single-turn traces only).

GEAP treats any word in single braces as a placeholder. Do not include raw
JSON examples with single braces in custom prompts. Custom metrics never return
rubric verdicts, and the score uses whatever scale the prompt asks for.

---

## 6. Output and errors

`geap_evaluate` returns a `STRUCT`:

```sql
STRUCT<
  score FLOAT64,
  explanation STRING,
  rubric_verdicts ARRAY<STRUCT<
    rubric_id STRING,
    rubric_content STRING,
    type STRING,
    importance STRING,
    verdict BOOL,
    reasoning STRING
  >>,
  status STRING
>
```

*   **Successful rows:** `status` is `NULL` and `score` holds a float (0.0 to
    1.0 for managed metrics). Managed metrics often return their reasoning in
    `rubric_verdicts` and leave `explanation` `NULL`. `type` and `rubric_id`
    are often `NULL` too.
*   **Failed rows:** `score` and `explanation` are `NULL`, `rubric_verdicts` is
    an empty array, and `status` describes the failure.

Every non-`NULL` status has the form `CODE: message`, where `CODE` is a
standard error code name such as `INVALID_ARGUMENT` or `UNAVAILABLE`.

### Rescoring failed rows

Four codes are temporary and worth rescoring: `RESOURCE_EXHAUSTED`,
`UNAVAILABLE`, `DEADLINE_EXCEEDED`, and `INTERNAL`. BigQuery doesn't cache
Python UDF results, so rescore only those rows and replace them:

```sql
UPDATE `my-project.my_dataset.evaluation_results` AS r
SET task_success = `my-project.my_dataset.geap_evaluate`(
  s.trace_json, 'multi_turn_task_success_v1', NULL)
FROM `my-project.my_dataset.sampled_session_traces` AS s
WHERE r.session_id = s.session_id
  AND (STARTS_WITH(r.task_success.status, 'RESOURCE_EXHAUSTED:')
    OR STARTS_WITH(r.task_success.status, 'UNAVAILABLE:')
    OR STARTS_WITH(r.task_success.status, 'DEADLINE_EXCEEDED:')
    OR STARTS_WITH(r.task_success.status, 'INTERNAL:'));
```

Every other code needs the input or setup fixed first:

*   `INVALID_ARGUMENT`: Input failed validation (oversized trace, wrong turn
    count, or invalid prompt). Fix the query or trace before retrying.
*   `PERMISSION_DENIED` or `UNAUTHENTICATED`: Run the grant command from
    [Step 1](#step-1-install-routines).
*   `NOT_FOUND` on a single-turn `_v1` metric: Switch to the `_v2` or `_v3`
    name (see [Metrics](#4-metrics)).

---

## 7. Known limits

*   **Trace size:** Traces must be under ~5 MiB (5,177,344 bytes). Oversized
    traces, NULL traces, and traces that aren't JSON objects return an
    `INVALID_ARGUMENT` status without failing the query.
*   **Regional endpoints:** The installer maps the dataset's location to a
    regional GEAP evaluation endpoint. Multi-region US maps to `us-central1`,
    and multi-region EU maps to `europe-west4`. Single regions must match one
    of the 14 supported evaluation regions. Requests go to the regional
    endpoint; where the judge model runs isn't documented.
*   **Permissions:** Calls run as the BigQuery Cloud Resource connection's
    service account, which needs `roles/aiplatform.user`. Anyone who can call
    the routine and use the connection can spend judge quota in the install
    project.
*   **Data sensitivity:** Explanations and rubric reasoning quote trace content,
    so the results table is as sensitive as the traces.
