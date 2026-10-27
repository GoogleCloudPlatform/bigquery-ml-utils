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
# One-Command Installer for the DeepEval BigQuery Managed Python UDF
#
# Usage:
#   ./install.sh [PROJECT_ID] [DATASET_ID] [LOCATION] [CONNECTION_ID] [MODEL_ID]
#
# Environment variable overrides are also supported:
#   PROJECT_ID=my-gcp-project DATASET_ID=agent_analytics ./install.sh
# ==============================================================================
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./install.sh [PROJECT_ID] [DATASET_ID] [LOCATION] [CONNECTION_ID] [MODEL_ID]

Arguments (optional if gcloud default project is configured):
  PROJECT_ID     Google Cloud Project ID (default: gcloud config get-value project)
  DATASET_ID     BigQuery dataset name to host the UDF (default: agent_analytics)
  LOCATION       BigQuery dataset & connection location (default: us)
  CONNECTION_ID  BigQuery Cloud Resource Connection ID (default: gemini_enterprise_connection)
  MODEL_ID       Primary Gemini judge model ID (default: gemini-2.5-flash)

Examples:
  ./install.sh my-gcp-project
  ./install.sh my-gcp-project agent_analytics us gemini_enterprise_connection gemini-2.5-flash
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
LOCATION="${3:-${LOCATION:-us}}"
CONNECTION_ID="${4:-${CONNECTION_ID:-gemini_enterprise_connection}}"
MODEL_ID="${5:-${MODEL_ID:-gemini-2.5-flash}}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID is not set." >&2
  echo "Pass it as './install.sh <PROJECT_ID>' or run 'gcloud config set project <PROJECT_ID>'." >&2
  exit 1
fi

if [[ ! "${PROJECT_ID}" =~ ^[a-zA-Z0-9_.:-]+$ ]]; then
  echo "Error: Invalid PROJECT_ID '${PROJECT_ID}'. Expected alphanumeric/colon/dot/hyphen characters." >&2
  exit 1
fi
for var_name in DATASET_ID LOCATION CONNECTION_ID; do
  var_val="${!var_name}"
  if [[ ! "${var_val}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: Invalid ${var_name} '${var_val}'. Expected a short identifier (letters, digits, '_', or '-')." >&2
    exit 1
  fi
done
if [[ ! "${MODEL_ID}" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo "Error: Invalid MODEL_ID '${MODEL_ID}'. Expected alphanumeric/dot/underscore/hyphen characters." >&2
  exit 1
fi

echo "=================================================================="
echo "Installing DeepEval BigQuery Python UDF (deepeval_evaluate)"
echo "  Project ID    : ${PROJECT_ID}"
echo "  Dataset ID    : ${DATASET_ID}"
echo "  Location      : ${LOCATION}"
echo "  Connection ID : ${CONNECTION_ID}"
echo "  Judge Model   : ${MODEL_ID}"
echo "=================================================================="

# 1. Enable required Google Cloud APIs
echo "[1/5] Enabling BigQuery, BigQuery Connection, and Gemini Enterprise Agent Platform APIs..."
gcloud services enable \
  bigquery.googleapis.com \
  bigqueryconnection.googleapis.com \
  aiplatform.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

# 2. Create BigQuery dataset if it does not exist (or verify its location matches)
echo "[2/5] Ensuring BigQuery dataset ${PROJECT_ID}:${DATASET_ID} exists..."
if DS_JSON="$(bq --project_id="${PROJECT_ID}" show --format=json --dataset "${PROJECT_ID}:${DATASET_ID}" 2>/dev/null)"; then
  EXISTING_LOC="$(printf '%s' "${DS_JSON}" | sed -n 's/.*"location":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
  if [[ -n "${EXISTING_LOC}" && "${EXISTING_LOC,,}" != "${LOCATION,,}" ]]; then
    echo "Error: Dataset ${PROJECT_ID}:${DATASET_ID} already exists in location '${EXISTING_LOC}', which does not match requested LOCATION '${LOCATION}'." >&2
    exit 1
  fi
else
  bq --project_id="${PROJECT_ID}" --location="${LOCATION}" mk --dataset "${PROJECT_ID}:${DATASET_ID}"
fi

# 3. Create BigQuery Cloud Resource Connection if it does not exist
echo "[3/5] Ensuring Cloud Resource Connection ${PROJECT_ID}.${LOCATION}.${CONNECTION_ID} exists..."
if ! bq --project_id="${PROJECT_ID}" show --connection "${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}" >/dev/null 2>&1; then
  bq --project_id="${PROJECT_ID}" mk \
    --connection \
    --connection_type=CLOUD_RESOURCE \
    --location="${LOCATION}" \
    "${CONNECTION_ID}"
fi

# 4. Grant Agent Platform User (roles/aiplatform.user) to the Connection Service Account
CONN_SA="$(
  bq --project_id="${PROJECT_ID}" show --format=json --connection "${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}" \
    | sed -n 's/.*"serviceAccountId":[[:space:]]*"\([^"]*\)".*/\1/p'
)"
if [[ -z "${CONN_SA}" ]]; then
  echo "Error: Could not retrieve serviceAccountId for connection ${PROJECT_ID}.${LOCATION}.${CONNECTION_ID}." >&2
  exit 1
fi
echo "[4/5] Granting Agent Platform User (roles/aiplatform.user) to connection service account (${CONN_SA})..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CONN_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  --quiet >/dev/null

# 5. Deploy the unified deepeval_evaluate Python UDF into the target project, dataset, and connection
echo "[5/5] Deploying deepeval_evaluate Python UDF from deepeval_udfs.sql..."
sed \
  -e "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" \
  -e "s/YOUR_DATASET_ID/${DATASET_ID}/g" \
  -e "s/YOUR_LOCATION/${LOCATION}/g" \
  -e "s/YOUR_CONNECTION_ID/${CONNECTION_ID}/g" \
  -e "s/YOUR_MODEL_ID/${MODEL_ID}/g" \
  "${SCRIPT_DIR}/deepeval_udfs.sql" \
  | bq query --project_id="${PROJECT_ID}" --location="${LOCATION}" --use_legacy_sql=false

echo "=================================================================="
echo "Successfully deployed deepeval_evaluate to ${PROJECT_ID}.${DATASET_ID}!"
echo "Note: IAM policy bindings can take 1-2 minutes to propagate before the first UDF query."
echo ""
echo "Quick test command:"
echo "  bq query --project_id=${PROJECT_ID} --location=${LOCATION} --use_legacy_sql=false \\"
echo "    \"SELECT \\\`${PROJECT_ID}.${DATASET_ID}.deepeval_evaluate\\\`(TO_JSON_STRING(STRUCT('Where is order #999?' AS input, 'Order #999 shipped via UPS.' AS actual_output)), ['answer_relevancy'])\""
echo "=================================================================="
