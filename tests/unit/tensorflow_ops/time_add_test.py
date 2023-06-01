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

"""Tests for BigQuery TimeAdd custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class TimeAddTest(tf.test.TestCase):

  def test_time_add(self):
    time = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    self.assertAllEqual(
        time_ops.time_add(time, interval, 'MICROSECOND'),
        tf.constant(['07:30:00.000002', '06:30:00.000002']),
    )
    self.assertAllEqual(
        time_ops.time_add(time, interval, 'MILLISECOND'),
        tf.constant(['07:30:00.002', '06:30:00.002']),
    )
    self.assertAllEqual(
        time_ops.time_add(time, interval, 'SECOND'),
        tf.constant(['07:30:02', '06:30:02']),
    )
    self.assertAllEqual(
        time_ops.time_add(time, interval, 'MINUTE'),
        tf.constant(['07:32:00', '06:32:00']),
    )
    self.assertAllEqual(
        time_ops.time_add(time, interval, 'HOUR'),
        tf.constant(['09:30:00', '08:30:00']),
    )
    # Test time can handle 3 formats of CAST AS STRING.
    time = tf.constant(['07:30:00', '06:30:00.010', '06:30:00.000100'])
    interval = tf.constant([2, 3, 4], dtype=tf.int64)
    self.assertAllEqual(
        time_ops.time_add(time, interval, 'MINUTE'),
        tf.constant(['07:32:00', '06:33:00.010', '06:34:00.000100']),
    )

  def test_time_add_invalid_time(self):
    time = tf.constant(['07:30:00.00000a', '06:30:00.000000'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:30:00.00000a"',
    ):
      self.evaluate(time_ops.time_add(time, interval, 'HOUR'))

  def test_time_add_invalid_part(self):
    time = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in TimeAdd: DAY',
    ):
      self.evaluate(time_ops.time_add(time, interval, 'DAY'))

  def test_time_add_different_shapes(self):
    time = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    interval = tf.constant([2, 2, 3], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'time and interval must have the same shape, but are 2, 3',
    ):
      self.evaluate(time_ops.time_add(time, interval, 'HOUR'))


if __name__ == '__main__':
  tf.test.main()
