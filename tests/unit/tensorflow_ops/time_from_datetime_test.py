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

"""Tests for BigQuery TimeFromDatetime custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class TimeFromDatetimeTest(tf.test.TestCase):

  def test_time_from_datetime(self):
    datetime = tf.constant(
        ['2023-02-02 02:02:01.152903', '2008-12-25 15:30:00.152903']
    )
    self.assertAllEqual(
        time_ops.time_from_datetime(datetime),
        tf.constant(['02:02:01.152903', '15:30:00.152903']),
    )

  def test_time_from_datetime_invalid_datetime(self):
    datetime = tf.constant(
        ['2023-02-02 02:02:01.15290a', '2008-12-25 15:30:00.152903']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-02-02 02:02:01.15290a"',
    ):
      self.evaluate(time_ops.time_from_datetime(datetime))


if __name__ == '__main__':
  tf.test.main()
