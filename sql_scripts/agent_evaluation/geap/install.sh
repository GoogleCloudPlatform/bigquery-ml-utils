#!/usr/bin/env bash
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
# ==============================================================================
# Installer for the GEAP BigQuery Evaluation Routine (geap_evaluate)
#
# Usage:
#   ./install.sh [PROJECT_ID] [DATASET_ID] [CONNECTION_ID] [EVENTS_TABLE]
#
# Environment variable overrides are also supported:
#   PROJECT_ID=my-gcp-project DATASET_ID=agent_analytics EVENTS_TABLE=my-gcp-project.bqaa.agent_events ./install.sh
#
# The dataset must already exist; its location decides the GEAP region. The
# script never changes IAM. It prints the grant and revoke commands instead.
# ==============================================================================
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./install.sh [PROJECT_ID] [DATASET_ID] [CONNECTION_ID] [EVENTS_TABLE]

Arguments:
  PROJECT_ID     Google Cloud Project ID (default: gcloud config get-value project)
  DATASET_ID     Existing BigQuery dataset to host the routines (default: agent_analytics)
  CONNECTION_ID  BigQuery Cloud Resource Connection ID (default: gemini_enterprise_connection)
  EVENTS_TABLE   Optional BQAA events table (project.dataset.table) to deploy eval_sessions and eval_turns

The dataset must already exist. Its location decides the GEAP region.

Examples:
  ./install.sh my-gcp-project
  ./install.sh my-gcp-project agent_analytics gemini_enterprise_connection
  ./install.sh my-gcp-project agent_analytics gemini_enterprise_connection my-gcp-project.bqaa.agent_events
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_GCLOUD_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ "${DEFAULT_GCLOUD_PROJECT}" == "(unset)" ]]; then
  DEFAULT_GCLOUD_PROJECT=""
fi

PROJECT_ID="${1:-${PROJECT_ID:-${DEFAULT_GCLOUD_PROJECT}}}"
DATASET_ID="${2:-${DATASET_ID:-agent_analytics}}"
CONNECTION_ID="${3:-${CONNECTION_ID:-gemini_enterprise_connection}}"
EVENTS_TABLE="${4:-${EVENTS_TABLE:-}}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID is not set." >&2
  echo "Pass it as './install.sh <PROJECT_ID>' or run 'gcloud config set project <PROJECT_ID>'." >&2
  exit 1
fi

# Validate Project ID format (lowercase letters, digits, and hyphens; 6 to 30 characters)
PROJECT_ID_REGEX="^[a-z][a-z0-9-]{4,28}[a-z0-9]$"
if ! [[ "${PROJECT_ID}" =~ ${PROJECT_ID_REGEX} ]]; then
  echo "Error: PROJECT_ID must start with a lowercase letter and end with a lowercase letter or number, and contain only lowercase letters, numbers, and hyphens." >&2
  exit 1
fi

# Validate Dataset ID format (letters, digits, and underscores)
DATASET_ID_REGEX="^[a-zA-Z0-9_]+$"
if ! [[ "${DATASET_ID}" =~ ${DATASET_ID_REGEX} ]]; then
  echo "Error: DATASET_ID must contain only letters, numbers, and underscores." >&2
  exit 1
fi

# Validate Connection ID format (letters, digits, and underscores)
CONNECTION_ID_REGEX="^[a-zA-Z0-9_]+$"
if ! [[ "${CONNECTION_ID}" =~ ${CONNECTION_ID_REGEX} ]]; then
  echo "Error: CONNECTION_ID must contain only letters, numbers, and underscores." >&2
  exit 1
fi

if [[ -n "${EVENTS_TABLE}" ]]; then
  EVENTS_TABLE_REGEX="^[a-z][a-z0-9-]{4,28}[a-z0-9]\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_-]+$"
  if ! [[ "${EVENTS_TABLE}" =~ ${EVENTS_TABLE_REGEX} ]]; then
    echo "Error: EVENTS_TABLE must be a valid project.dataset.table identifier." >&2
    exit 1
  fi
fi

resolve_geap_region() {
  local loc="$1"
  local loc_lower
  loc_lower="$(printf '%s' "${loc}" | tr '[:upper:]' '[:lower:]')"
  case "${loc_lower}" in
    us)
      GEAP_REGION="us-central1"
      ;;
    eu)
      GEAP_REGION="europe-west4"
      ;;
    us-central1|us-east1|us-east4|us-east5|us-south1|us-west1|us-west4|\
    europe-central2|europe-north1|europe-southwest1|europe-west1|europe-west4|europe-west8|europe-west9)
      GEAP_REGION="${loc_lower}"
      ;;
    *)
      echo "Error: Unsupported location '${loc}' for GEAP evaluation service." >&2
      echo "Supported regions are: us (maps to us-central1), eu (maps to europe-west4), us-central1, us-east1, us-east4, us-east5, us-south1, us-west1, us-west4, europe-central2, europe-north1, europe-southwest1, europe-west1, europe-west4, europe-west8, europe-west9." >&2
      exit 1
      ;;
  esac
}

# 1. Enable required Google Cloud APIs before any BigQuery operations
echo "[1/5] Enabling BigQuery, BigQuery Connection, and GEAP APIs..."
if ! gcloud services enable \
  bigquery.googleapis.com \
  bigqueryconnection.googleapis.com \
  aiplatform.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet; then
  echo "Warning: Could not enable Google Cloud APIs automatically (continuing in case they are already enabled)." >&2
fi

# Read the location from the existing dataset. Creating a missing dataset in a
# default location could put customer data outside the region they chose.
DS_SHOW_RC=0
DS_JSON="$(bq --project_id="${PROJECT_ID}" show --format=json --dataset "${PROJECT_ID}:${DATASET_ID}" 2>&1)" || DS_SHOW_RC=$?
if [[ ${DS_SHOW_RC} -ne 0 ]]; then
  DS_JSON_LOWER="$(printf '%s' "${DS_JSON}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${DS_JSON_LOWER}" == *"not found"* ]]; then
    echo "Error: Dataset ${PROJECT_ID}:${DATASET_ID} does not exist." >&2
    echo "Create it in the location you want to evaluate in, then rerun this script, for example:" >&2
    echo "  bq --location=EU mk --dataset ${PROJECT_ID}:${DATASET_ID}" >&2
  else
    echo "Error: Failed to inspect dataset ${PROJECT_ID}:${DATASET_ID}: ${DS_JSON}" >&2
  fi
  exit 1
fi
LOCATION="$(printf '%s' "${DS_JSON}" | sed -n 's/.*"location":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
if [[ -z "${LOCATION}" ]]; then
  echo "Error: Could not read the location of dataset ${PROJECT_ID}:${DATASET_ID}." >&2
  exit 1
fi
resolve_geap_region "${LOCATION}"

if [[ -n "${EVENTS_TABLE}" ]]; then
  EVENTS_PROJECT="${EVENTS_TABLE%%.*}"
  EVENTS_REST="${EVENTS_TABLE#*.}"
  EVENTS_DATASET="${EVENTS_REST%%.*}"
  EVENTS_TABLE_NAME="${EVENTS_REST#*.}"
  TBL_ERR_FILE="$(mktemp)"
  TBL_SHOW_RC=0
  TBL_JSON="$(bq --project_id="${PROJECT_ID}" show --format=json "${EVENTS_PROJECT}:${EVENTS_DATASET}.${EVENTS_TABLE_NAME}" 2>"${TBL_ERR_FILE}")" || TBL_SHOW_RC=$?
  TBL_ERR="$(cat "${TBL_ERR_FILE}")"
  rm -f "${TBL_ERR_FILE}"
  if [[ ${TBL_SHOW_RC} -ne 0 ]]; then
    echo "Error: Failed to inspect events table ${EVENTS_TABLE}: ${TBL_JSON} ${TBL_ERR}" >&2
    exit 1
  fi

  EXPECTED_SCHEMA_VERSION="$(sed -n 's/.*schema_version=\([0-9][0-9]*\).*/\1/p' "${SCRIPT_DIR}/traces/eval_traces.sql" | head -n 1)"
  # Depth-aware JSON scan in POSIX awk (avoids requiring python3 at runtime).
  # Tracks string quotes (in_str/esc) and object nesting (depth) so values are
  # only read from their exact path in BigQuery's tables.get JSON:
  #   - depth 1: top-level "location" (ignores "location" in labels/descriptions)
  #   - depth 2 (labels): "adk_schema_version"
  #   - depth 3 (schema.fields[]): top-level column "name" (ignores nested RECORD subfields)
  TBL_CHECK_OUT="$(printf '%s' "${TBL_JSON}" | awk '
BEGIN { depth = 0; in_str = 0; esc = 0; tok = ""; expect_val = 0; key = ""; in_labels = 0; in_schema = 0; in_fields = 0; loc = ""; label = ""; cols = "" }
{
  n = length($0)
  for (i = 1; i <= n; i++) {
    c = substr($0, i, 1)
    if (in_str) {
      if (esc) { esc = 0; tok = tok c; continue }
      if (c == "\\") { esc = 1; continue }
      if (c == "\"") {
        in_str = 0
        if (expect_val) {
          if (depth == 1 && key == "location") loc = tok
          else if (depth == 2 && in_labels && key == "adk_schema_version") label = tok
          else if (depth == 3 && in_fields && key == "name") cols = cols "|" tolower(tok) "|"
          expect_val = 0; key = ""
        } else {
          key = tok
        }
        tok = ""
      } else { tok = tok c }
      continue
    }
    if (c == "\"") { in_str = 1; tok = ""; continue }
    if (c == ":") { expect_val = 1; continue }
    if (c == "," || c == "}" || c == "]") {
      if (expect_val && c != "}") { expect_val = 0; key = "" }
    }
    if (c == "{") {
      depth++
      if (depth == 2 && expect_val && key == "labels") in_labels = 1
      else if (depth == 2 && expect_val && key == "schema") in_schema = 1
      expect_val = 0; key = ""
    } else if (c == "}") {
      if (depth == 2) { in_labels = 0; in_schema = 0 }
      depth--
    } else if (c == "[") {
      if (depth == 2 && in_schema && expect_val && key == "fields") in_fields = 1
      expect_val = 0; key = ""
    } else if (c == "]") {
      if (depth == 2 && in_fields) in_fields = 0
    }
  }
}
END { print loc; print label; print cols }
')"
  TABLE_LOCATION="$(printf '%s\n' "${TBL_CHECK_OUT}" | sed -n '1p')"
  TABLE_SCHEMA_VERSION="$(printf '%s\n' "${TBL_CHECK_OUT}" | sed -n '2p')"
  TABLE_COLS="$(printf '%s\n' "${TBL_CHECK_OUT}" | sed -n '3p')"
  MISSING_COLS=""
  for col in timestamp event_id event_type agent session_id invocation_id user_id content content_parts attributes latency_ms status error_message is_truncated; do
    if [[ "${TABLE_COLS}" != *"|${col}|"* ]]; then
      MISSING_COLS="${MISSING_COLS:+${MISSING_COLS},}${col}"
    fi
  done

  if [[ -z "${TABLE_LOCATION}" ]]; then
    echo "Error: Could not read the location of events table ${EVENTS_TABLE}." >&2
    exit 1
  fi
  TABLE_LOC_LOWER="$(printf '%s' "${TABLE_LOCATION}" | tr '[:upper:]' '[:lower:]')"
  DS_LOC_LOWER="$(printf '%s' "${LOCATION}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${TABLE_LOC_LOWER}" != "${DS_LOC_LOWER}" ]]; then
    echo "Error: Events table ${EVENTS_TABLE} is in location '${TABLE_LOCATION}', which does not match dataset ${PROJECT_ID}:${DATASET_ID} location '${LOCATION}'." >&2
    exit 1
  fi
  if [[ -n "${MISSING_COLS}" ]]; then
    echo "Error: Events table ${EVENTS_TABLE} is missing required columns: ${MISSING_COLS}." >&2
    exit 1
  fi
  if [[ -z "${TABLE_SCHEMA_VERSION}" ]]; then
    echo "Warning: Events table ${EVENTS_TABLE} has no 'adk_schema_version' label (expected '${EXPECTED_SCHEMA_VERSION}'). Trace SQL may misread events." >&2
  elif [[ -n "${EXPECTED_SCHEMA_VERSION}" && "${TABLE_SCHEMA_VERSION}" != "${EXPECTED_SCHEMA_VERSION}" ]]; then
    echo "Warning: Events table ${EVENTS_TABLE} has adk_schema_version='${TABLE_SCHEMA_VERSION}' (expected '${EXPECTED_SCHEMA_VERSION}'). Trace SQL may misread events." >&2
  fi
fi

echo "=================================================================="
echo "Installing GEAP BigQuery Agent Evaluation Routine (geap_evaluate)"
echo "  Project ID    : ${PROJECT_ID}"
echo "  Dataset ID    : ${DATASET_ID}"
echo "  Location      : ${LOCATION}"
echo "  GEAP Region   : ${GEAP_REGION}"
echo "  Connection ID : ${CONNECTION_ID}"
if [[ -n "${EVENTS_TABLE}" ]]; then
  echo "  Events Table  : ${EVENTS_TABLE}"
fi
echo "=================================================================="

# 2. Create BigQuery Cloud Resource Connection if it does not exist
echo "[2/5] Ensuring Cloud Resource Connection ${PROJECT_ID}.${LOCATION}.${CONNECTION_ID} exists..."
if ! bq --project_id="${PROJECT_ID}" show --connection "${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}" >/dev/null 2>&1; then
  bq --project_id="${PROJECT_ID}" mk \
    --connection \
    --connection_type=CLOUD_RESOURCE \
    --location="${LOCATION}" \
    "${CONNECTION_ID}"
fi

CONN_SA="$(
  bq --project_id="${PROJECT_ID}" show --format=json --connection "${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}" \
    | sed -n 's/.*"serviceAccountId":[[:space:]]*"\([^"]*\)".*/\1/p'
)"
if [[ -z "${CONN_SA}" ]]; then
  echo "Error: Could not retrieve serviceAccountId for connection ${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}." >&2
  exit 1
fi

# 3. Deploy geap_evaluate and geap_evaluate_impl SQL/Python routines
echo "[3/5] Deploying geap_evaluate and geap_evaluate_impl from geap_evaluate.sql..."
sed \
  -e "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" \
  -e "s/YOUR_DATASET_ID/${DATASET_ID}/g" \
  -e "s/YOUR_CONNECTION_LOCATION/${LOCATION}/g" \
  -e "s/YOUR_CONNECTION_ID/${CONNECTION_ID}/g" \
  -e "s/YOUR_GEAP_REGION/${GEAP_REGION}/g" \
  "${SCRIPT_DIR}/geap_evaluate.sql" \
  | bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false

if [[ -n "${EVENTS_TABLE}" ]]; then
  echo "[3b/5] Deploying eval_sessions and eval_turns from traces/eval_traces.sql..."
  sed \
    -e "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" \
    -e "s/YOUR_DATASET_ID/${DATASET_ID}/g" \
    -e "s/YOUR_EVENTS_TABLE/${EVENTS_TABLE}/g" \
    "${SCRIPT_DIR}/traces/eval_traces.sql" \
    | bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false
fi

# 4. Run smoke test call with placeholder metric name to verify routines install and execute locally
echo "[4/5] Running smoke test call with placeholder metric name..."
SMOKE_TEST_SQL="SELECT \`${PROJECT_ID}.${DATASET_ID}.geap_evaluate\`(JSON '{\"turns\": [{\"id\": \"smoke_turn\", \"events\": []}]}', 'smoke_test_metric', NULL).status AS status"
if ! SMOKE_RESULT="$(bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false --format=csv "${SMOKE_TEST_SQL}" 2>&1)"; then
  echo "Error: Smoke test query failed." >&2
  echo "Output: ${SMOKE_RESULT}" >&2
  exit 1
fi
if [[ "${SMOKE_RESULT}" != *"INVALID_ARGUMENT: Managed metric smoke_test_metric is not supported"* ]]; then
  echo "Error: Smoke test call failed to return expected INVALID_ARGUMENT status." >&2
  echo "Output: ${SMOKE_RESULT}" >&2
  exit 1
fi

# 5. Print, but don't run, the IAM commands. Changing IAM needs someone with
# rights on the project, and roles/aiplatform.user also allows calling any
# model in the project.
echo "Smoke test passed (local checks functional)."
echo "[5/5] IAM setup (not run by this script)."
echo "=================================================================="
echo "Successfully deployed geap_evaluate to ${PROJECT_ID}.${DATASET_ID}!"
echo ""
echo "Before scoring, someone with rights on ${PROJECT_ID} must grant the"
echo "connection's service account the Agent Platform User role. Until then,"
echo "every row returns a PERMISSION_DENIED status. The role also allows"
echo "calling any model in the project."
echo ""
echo "  Grant  : gcloud projects add-iam-policy-binding ${PROJECT_ID} --member=serviceAccount:${CONN_SA} --role=roles/aiplatform.user --condition=None"
echo "  Revoke : gcloud projects remove-iam-policy-binding ${PROJECT_ID} --member=serviceAccount:${CONN_SA} --role=roles/aiplatform.user --condition=None"
echo "=================================================================="
