#!/bin/bash
set -e

local_install() {
  echo "=== Setting up pyenv"
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"

  echo "=== Installing Python 3.10"
  pyenv install 3.10
  pyenv global 3.10

  echo "=== Setting up Bazelisk"
  sudo wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64" -O $BAZEL_FILE
  sudo chmod +x $BAZEL_FILE
}

docker_install() {
  echo "=== Setting up pyenv"
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"

  echo "=== Installing Python 3.7"
  pyenv install 3.7

  echo "=== Installing Python 3.8"
  pyenv install 3.8

  echo "=== Installing Python 3.9"
  pyenv install 3.9

  echo "=== Installing Python 3.10"
  pyenv install 3.10
  pyenv global 3.10

  echo "=== Setting up Bazelisk to pick up Bazel version in .bazelversion"
  wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64" -O $BAZEL_FILE
  chmod +x $BAZEL_FILE

  echo "=== Cloning bigquery-ml-utils.git"
  git clone https://github.com/GoogleCloudPlatform/bigquery-ml-utils.git && cd bigquery-ml-utils
}

function main() {
  # Python 3.10.
  TEST_PY_VERSION=10
  # Python 3.7 ~ 3.10.
  SUPPORTED_PY_VERSIONS=( 7 8 9 10 )
  BAZEL_FILE=/usr/bin/bazel

  USAGE='release.sh -t -d WHEEL_DIST'
  TEST_MODE=false

  while getopts 'tv:d:' arg; do
    case "${arg}" in
      t) TEST_MODE=true ;;
      d) WHEEL_DIST="${OPTARG}" ;;
      *) echo "$USAGE" >&2; return 1 ;;
    esac
  done

  if [[ -z ${WHEEL_DIST} ]]; then
    echo "ERROR: No WHEEL_DIST provided" >&2
    echo "$USAGE" >&2
    exit 1
  fi

  if [[ $TEST_MODE == true ]]
  then
    echo "=== Setting up local test environment"
    SUPPORTED_PY_VERSIONS=( $TEST_PY_VERSION )
    local_install
  else
    echo "=== Setting up manylinxu release environment"
    docker_install
  fi

  for V in "${SUPPORTED_PY_VERSIONS[@]}"
  do
    echo "=== Switching to Python 3.$V"
    pyenv global 3.$V
    python3 -m pip install --upgrade pip
    python3 -m pip install virtualenv && python3 -m virtualenv ~/.virtualenvs/env3$V && source ~/.virtualenvs/env3$V/bin/activate

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

  if [[ $TEST_MODE == true ]]
  then
    cp -r artifacts/* $WHEEL_DIST
  else
    echo "=== Repairing the wheels to be manylinux compatible"
    auditwheel repair --exclude libtensorflow_framework.so.2 --plat manylinux_2_17_x86_64 -w $WHEEL_DIST artifacts/*.whl
  fi

  echo "=== Generated bigquery-ml-utils wheels in $WHEEL_DIST"
}

main "$@"
