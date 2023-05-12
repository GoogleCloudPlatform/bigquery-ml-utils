#!/bin/bash
set -e

local_install() {
  echo "=== Installing Python 3.$TEST_PY_VERSION"
  sudo apt update && sudo apt install python3.$TEST_PY_VERSION rsync wget

  echo "=== Setting up Bazelisk"
  sudo wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64" -O $BAZEL_FILE
  sudo chmod +x $BAZEL_FILE
}

docker_install() {
  echo "=== Installing Python 3.8 and Python 3.9"
  dnf update && dnf install python38 python39 rsync wget yum-utils make gcc openssl-devel bzip2-devel libffi-devel zlib-devel -y

  echo "=== Installing Python 3.7"
  wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz && tar xzf Python-3.7.12.tgz
  pushd Python-3.7.12 && ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions
  make -j ${nproc} && make altinstall
  popd

  echo "=== Installing Python 3.10"
  wget https://www.python.org/ftp/python/3.10.5/Python-3.10.5.tgz && tar xzf Python-3.10.5.tgz
  pushd Python-3.10.5 && ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions
  make -j ${nproc} && make altinstall
  popd

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

  USAGE='release.sh -t -v PKG_VERSION -d WHEEL_DIST'
  TEST_MODE=false

  while getopts 'tv:d:' arg; do
    case "${arg}" in
      t) TEST_MODE=true ;;
      v) PKG_VERSION="${OPTARG}" ;;
      d) WHEEL_DIST="${OPTARG}" ;;
      *) echo "$USAGE" >&2; return 1 ;;
    esac
  done

  if [[ -z ${PKG_VERSION} ]]; then
    echo "ERROR: No PKG_VERSION provided" >&2
    echo "$USAGE" >&2
    exit 1
  fi
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
    PY_V=python3."$V"
    ${PY_V} -m pip install --upgrade pip
    ${PY_V} -m pip install virtualenv && ${PY_V} -m virtualenv ~/.virtualenvs/env3$V && source ~/.virtualenvs/env3$V/bin/activate

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

    echo "=== Building bigquery-ml-utils pip package $PKG_VERSION"
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
