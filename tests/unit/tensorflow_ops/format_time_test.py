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

"""Tests for BigQuery FormatTime custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class FormatTimeTest(tf.test.TestCase):

  def test_format_time(self):
    time = tf.constant(['07:31:15.000000', '06:22:23.000000'])
    self.assertAllEqual(
        time_ops.format_time('%R', time),
        tf.constant(['07:31', '06:22']),
    )

  def test_format_time_invalid_time(self):
    time = tf.constant(['07:31:15.00000a', '06:22:23.000000'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:31:15.00000a"',
    ):
      self.evaluate(time_ops.format_time('%R', time))

  def test_format_time_invalid_format(self):
    time = tf.constant(['07:31:15.000000', '06:22:23.000000'])
    self.assertAllEqual(
        time_ops.format_time('abc', time),
        tf.constant(['abc', 'abc']),
    )


if __name__ == '__main__':
  tf.test.main()
