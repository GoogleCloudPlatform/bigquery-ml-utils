#!/bin/bash
set -e

docker_install() {
  echo "=== Setting up Bazelisk to pick up Bazel version in .bazelversion"
  dnf -y install wget rsync
  wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64" -O $BAZEL_FILE
  chmod +x $BAZEL_FILE

  echo "=== Cloning bigquery-ml-utils.git"
  git clone https://github.com/GoogleCloudPlatform/bigquery-ml-utils.git && cd bigquery-ml-utils
}

function main() {
  # Python 3.7 ~ 3.10.
  SUPPORTED_PY_VERSIONS=( 7 8 9 10 )
  BAZEL_FILE=/usr/bin/bazel

  USAGE='release.sh -d WHEEL_DIST'

  while getopts 'd:' arg; do
    case "${arg}" in
      d) WHEEL_DIST="${OPTARG}" ;;
      *) echo "$USAGE" >&2; return 1 ;;
    esac
  done

  if [[ -z ${WHEEL_DIST} ]]; then
    echo "ERROR: No WHEEL_DIST provided" >&2
    echo "$USAGE" >&2
    exit 1
  fi

  echo "=== Setting up manylinux release environment"
  docker_install

  for V in "${SUPPORTED_PY_VERSIONS[@]}"
  do
    echo "=== Switching to Python 3.$V"
    PY_V=python3."$V"
    ${PY_V} -m venv ~/.virtualenvs/env3$V && source ~/.virtualenvs/env3$V/bin/activate

    # Retry in case newly installed tf is not recognized right away.
    echo "=== Generating .bazelrc"
    max_retries=2
    for i in $(seq 1 $max_retries)
    do
      yes '' | python3 configure.py >& /dev/null
      result=$?
      if [[ $result -eq 0 ]]
      then
        break
      fi
    done

    if [[ $result -ne 0 ]]
    then
      echo "ERROR: Failed to generate .bazelrc" >&2
      exit 1
    fi

    echo "=== Building bigquery-ml-utils pip package"
    bazel clean --expunge && bazel build build_pip_pkg
    bazel-bin/build_pip_pkg artifacts
    deactivate
  done

  echo "=== Repairing the wheels to be manylinux compatible"
  auditwheel repair --exclude libtensorflow_framework.so.2 --plat manylinux_2_17_x86_64 -w $WHEEL_DIST artifacts/*.whl

  echo "=== Generated bigquery-ml-utils wheels in $WHEEL_DIST"
}

main "$@"
