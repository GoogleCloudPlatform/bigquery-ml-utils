#!/usr/bin/env bash
# Copyright 2023 Google LLC
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
set -e
set -x

PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/bigquery_ml_utils/"

function main() {
  while [[ ! -z "${1}" ]]; do
    DEST=${1}
    shift
  done

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p ${DEST}
  DEST=$(readlink -f "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy TensorFlow Custom op files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}README.md "${TMPDIR}"
  mkdir "${TMPDIR}"/bigquery_ml_utils
  cp ${PIP_FILE_PREFIX}__init__.py "${TMPDIR}"/bigquery_ml_utils
  rsync -avm -L ${PIP_FILE_PREFIX}inference "${TMPDIR}"/bigquery_ml_utils
  rsync -avm -L ${PIP_FILE_PREFIX}model_generator "${TMPDIR}"/bigquery_ml_utils
  rsync -avm -L ${PIP_FILE_PREFIX}tensorflow_ops "${TMPDIR}"/bigquery_ml_utils

  # Read PYTHON_BIN_PATH that is set by configure.py.
  if [[ -e python_bin_path.sh ]]; then
    source python_bin_path.sh
  fi

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  "${PYTHON_BIN_PATH:-python3}" setup.py bdist_wheel > /dev/null

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"