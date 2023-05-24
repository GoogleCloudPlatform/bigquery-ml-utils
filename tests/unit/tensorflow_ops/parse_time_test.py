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

"""Tests for BigQuery ParseTime custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class ParseTimeTest(tf.test.TestCase):

  def test_parse_time(self):
    time = tf.constant(['07:31:15', '06:22:23'])
    self.assertAllEqual(
        time_ops.parse_time('%I:%M:%S', time),
        tf.constant(['07:31:15.000000', '06:22:23.000000']),
    )

  def test_parse_time_invalid_time(self):
    time = tf.constant(['07:30:00.00000', '06:30:00.000000'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:30:00.00000"',
    ):
      self.evaluate(time_ops.parse_time('%I:%M:%S', time))

  def test_parse_time_invalid_format(self):
    time = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        "Mismatch between format character 'a' and string character '0'",
    ):
      self.evaluate(time_ops.parse_time('abc', time))


if __name__ == '__main__':
  tf.test.main()
