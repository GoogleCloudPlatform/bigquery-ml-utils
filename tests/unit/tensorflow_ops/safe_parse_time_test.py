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

"""Tests for BigQuery SafeParseTime custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class SafeParseTimeTest(tf.test.TestCase):

  def test_safe_parse_time(self):
    time = tf.constant(['07:31:15', '06:22:23'])
    self.assertAllEqual(
        time_ops.safe_parse_time('%I:%M:%S', time),
        tf.constant(['07:31:15', '06:22:23']),
    )

  def test_safe_parse_time_invalid_time(self):
    time = tf.constant(['07:31:15', '06:22:23', 'invalid_time'])
    self.assertAllEqual(
        time_ops.safe_parse_time('%I:%M:%S', time),
        tf.constant(['07:31:15', '06:22:23', '12:34:56.123456']),
    )

  def test_safe_parse_time_invalid_format(self):
    time = tf.constant(['07:31:15', '06:22:23'])
    self.assertAllEqual(
        time_ops.safe_parse_time('invalid_format', time),
        tf.constant(['12:34:56.123456', '12:34:56.123456']),
    )


if __name__ == '__main__':
  tf.test.main()
