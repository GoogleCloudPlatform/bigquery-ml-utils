#!/bin/bash
# local_release.sh: Build bigquery_ml_utils wheel from local source.
# This script is designed to be run inside a manylinux_2_28 Docker container.
set -x
set -e

BAZEL_FILE=/usr/bin/bazel
WHEEL_DIST=$1

if [[ -z ${WHEEL_DIST} ]]; then
  echo "ERROR: No WHEEL_DIST provided" >&2
  exit 1
fi

echo "=== Setting up Bazelisk"
dnf -y install wget rsync
wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64" -O $BAZEL_FILE
chmod +x $BAZEL_FILE

# The script is run from /workspace
cd /workspace

# Manylinux 2.28 has python in /opt/python/
# Find the correct python 3.10 path for manylinux
PY310_BIN=$(ls -d /opt/python/cp310-cp310/bin/python3)

echo "=== Switching to Python 3.10 using ${PY310_BIN}"
${PY310_BIN} -m venv ~/env310 && source ~/env310/bin/activate

echo "=== Generating .bazelrc"
export USE_BAZEL_VERSION=5.3.0
# Ensure we are using the venv python for the build
python3 configure.py <<EOF

EOF

echo "=== Building bigquery-ml-utils pip package"
bazel clean --expunge && bazel build --copt=-fno-exceptions build_pip_pkg
bazel-bin/build_pip_pkg artifacts

echo "=== Repairing the wheels to be manylinux compatible"
# We exclude libtensorflow_framework.so.2 as it is provided by the tensorflow pip package at runtime.
auditwheel repair --exclude libtensorflow_framework.so.2 --plat manylinux_2_28_x86_64 -w $WHEEL_DIST artifacts/*.whl

echo "=== Generated bigquery-ml-utils wheels in $WHEEL_DIST"
deactivate
