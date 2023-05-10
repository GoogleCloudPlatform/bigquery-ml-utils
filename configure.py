# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Usage: configure.py [--quiet] [--no-deps]
#
# Options:
#  --quiet  Give less output.
#  --no-deps  Don't install Python dependencies
"""Configures BigQuery ML Utils to be built from source."""

import argparse
import logging
import os
import subprocess
import sys
from typing import List

_BAZELRC = '.bazelrc'
_BAZEL_QUERY = '.bazel-query.sh'
_PYTHON_BIN_PATH = 'python_bin_path.sh'


# Writes variables to bazelrc file
def write_to_bazelrc(line: str):
  with open(_BAZELRC, 'a') as f:
    f.write(line + '\n')


def write_action_env(var_name: str, var: str):
  write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))
  with open(_BAZEL_QUERY, 'a') as f:
    f.write('{}="{}" '.format(var_name, var))


def generate_shared_lib_name(tf_lflags: List[str]) -> str:
  """Converts the linkflag namespec to the full shared library name.

  Args:
    tf_lflags: List of linkflag namespec. The first entry specifies the
      directory containing the TensorFlow framework library. The second entry
      specifies the name of the Tensorflow shared lib (For Linux,
      '-l:libtensorflow_framework.so.%s' % version).

  Returns:
    Name of the Tensorflow shared lib.
  """
  # Assume Linux for now
  return tf_lflags[1][3:]


def create_build_configuration():
  """Main function to create build configuration."""
  if os.path.isfile(_BAZELRC):
    os.remove(_BAZELRC)
  if os.path.isfile(_BAZEL_QUERY):
    os.remove(_BAZEL_QUERY)
  if os.path.isfile(_PYTHON_BIN_PATH):
    os.remove(_PYTHON_BIN_PATH)

  environ_cp = dict(os.environ)
  setup_python(environ_cp)

  print()
  print('Configuring BigQuery ML Utils to be built from source...')

  pip_install_options = ['--upgrade']
  parser = argparse.ArgumentParser()
  parser.add_argument('--quiet', action='store_true', help='Give less output.')
  parser.add_argument(
      '--no-deps',
      action='store_true',
      help='Do not check and install Python dependencies.',
  )
  args = parser.parse_args()
  if args.quiet:
    pip_install_options.append('--quiet')

  with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

  print()
  if args.no_deps:
    print('> Using pre-installed Tensorflow.')
  else:
    print('> Installing', required_packages)
    install_cmd = [environ_cp['PYTHON_BIN_PATH'], '-m', 'pip', 'install']
    install_cmd.extend(pip_install_options)
    install_cmd.extend(required_packages)
    subprocess.check_call(install_cmd)

  logging.disable(logging.WARNING)

  import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top

  # pylint: disable=invalid-name
  _TF_CFLAGS = tf.sysconfig.get_compile_flags()
  _TF_LFLAGS = tf.sysconfig.get_link_flags()
  _TF_CXX11_ABI_FLAG = tf.sysconfig.CXX11_ABI_FLAG

  _TF_SHARED_LIBRARY_NAME = generate_shared_lib_name(_TF_LFLAGS)
  _TF_HEADER_DIR = _TF_CFLAGS[0][2:]
  _TF_SHARED_LIBRARY_DIR = _TF_LFLAGS[0][2:]
  # pylint: enable=invalid-name

  write_action_env('TF_HEADER_DIR', _TF_HEADER_DIR)
  write_action_env('TF_SHARED_LIBRARY_DIR', _TF_SHARED_LIBRARY_DIR)
  write_action_env('TF_SHARED_LIBRARY_NAME', _TF_SHARED_LIBRARY_NAME)
  write_action_env('TF_CXX11_ABI_FLAG', _TF_CXX11_ABI_FLAG)
  write_action_env('BAZEL_CXXOPTS', '-std=c++17')

  write_to_bazelrc('build --spawn_strategy=standalone')
  write_to_bazelrc('build --strategy=Genrule=standalone')
  write_to_bazelrc('build --experimental_repo_remote_exec')
  write_to_bazelrc('build --experimental_cc_shared_library')
  write_to_bazelrc('build -c opt')

  print()
  print('Build configurations successfully written to', _BAZELRC)
  print()

  with open(_BAZEL_QUERY, 'a') as f:
    f.write('bazel query "$@"')


def setup_python(environ_cp):
  """Setup python related env variables."""
  # Get PYTHON_BIN_PATH, default is the current running python.
  default_python_bin_path = sys.executable
  ask_python_bin_path = (
      'Please specify the location of python. [Enter to use the default: {}]: '
  ).format(default_python_bin_path)
  while True:
    python_bin_path = get_from_env_or_user_or_default(
        environ_cp,
        'PYTHON_BIN_PATH',
        ask_python_bin_path,
        default_python_bin_path,
    )
    # Check if the path is valid
    if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
      break
    elif not os.path.exists(python_bin_path):
      print('Invalid python path: {} cannot be found.'.format(python_bin_path))
    else:
      print(
          '{} is not executable.  Is it the python binary?'.format(
              python_bin_path
          )
      )
    environ_cp['PYTHON_BIN_PATH'] = ''

  # Get PYTHON_LIB_PATH
  python_lib_path = environ_cp.get('PYTHON_LIB_PATH')
  if not python_lib_path:
    python_lib_paths = get_python_path(environ_cp, python_bin_path)
    if environ_cp.get('USE_DEFAULT_PYTHON_LIB_PATH') == '1':
      python_lib_path = python_lib_paths[0]
    else:
      print(
          'Found possible Python library paths:\n  %s'
          % '\n  '.join(python_lib_paths)
      )
      default_python_lib_path = python_lib_paths[0]
      python_lib_path = get_input(
          'Please input the desired Python library path to use.  '
          'Enter to use the default: [{}]\n'.format(python_lib_paths[0])
      )
      if not python_lib_path:
        python_lib_path = default_python_lib_path
    environ_cp['PYTHON_LIB_PATH'] = python_lib_path

  # Set-up env variables used by python_configure.bzl
  write_action_env('PYTHON_BIN_PATH', python_bin_path)
  write_action_env('PYTHON_LIB_PATH', python_lib_path)
  write_to_bazelrc('build --python_path="{}"'.format(python_bin_path))
  environ_cp['PYTHON_BIN_PATH'] = python_bin_path

  # If choosen python_lib_path is from a path specified in the PYTHONPATH
  # variable, need to tell bazel to include PYTHONPATH
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
    if python_lib_path in python_paths:
      write_action_env('PYTHONPATH', environ_cp.get('PYTHONPATH'))

  # Write tools/python_bin_path.sh
  with open(_PYTHON_BIN_PATH, 'a') as f:
    f.write('export PYTHON_BIN_PATH="{}"'.format(python_bin_path))


def get_from_env_or_user_or_default(
    environ_cp, var_name, ask_for_var, var_default
):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def get_python_path(environ_cp, python_bin_path):
  """Get the python site package paths."""
  python_paths = []
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
  try:
    stderr = open(os.devnull, 'wb')
    library_paths = run_shell(
        [
            python_bin_path,
            '-c',
            'import site; print("\\n".join(site.getsitepackages()))',
        ],
        stderr=stderr,
    ).split('\n')
  except subprocess.CalledProcessError:
    library_paths = [
        run_shell([
            python_bin_path,
            '-c',
            (
                'from distutils.sysconfig import get_python_lib;'
                'print(get_python_lib())'
            ),
        ])
    ]

  all_paths = set(python_paths + library_paths)
  # Sort set so order is deterministic
  all_paths = sorted(all_paths)

  paths = []
  for path in all_paths:
    if os.path.isdir(path):
      paths.append(path)
  return paths


def get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def run_shell(cmd, allow_non_zero=False, stderr=None):
  if stderr is None:
    stderr = sys.stdout
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd, stderr=stderr)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd, stderr=stderr)
  return output.decode('UTF-8').strip()


if __name__ == '__main__':
  create_build_configuration()
