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
PIP="pip3"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Check if Tensorflow's installed
if [[ $(${PIP} show tensorflow-cpu) == *tensorflow-cpu* ]] || [[ $(${PIP} show tf-nightly-cpu) == *tf-nightly-cpu* ]] ; then
  echo 'Using installed tensorflow'
else
  # Uninstall GPU version if it is installed.
  if [[ $(${PIP} show tensorflow) == *tensorflow* ]]; then
    echo 'Already have gpu version of tensorflow installed. Uninstalling......\n'
    ${PIP} uninstall tensorflow
  elif [[ $(${PIP} show tf-nightly) == *tf-nightly* ]]; then
    echo 'Already have gpu version of tensorflow installed. Uninstalling......\n'
    ${PIP} uninstall tf-nightly
  fi
  # Install CPU version
  echo 'Installing tensorflow-cpu......\n'
  ${PIP} install tensorflow-cpu
fi

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build --experimental_repo_remote_exec"
write_to_bazelrc "build --experimental_cc_shared_library"
write_to_bazelrc "build --action_env=BAZEL_CXXOPTS="-std=c++17""
write_to_bazelrc "build -c opt"

SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.so"
fi

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_LFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
