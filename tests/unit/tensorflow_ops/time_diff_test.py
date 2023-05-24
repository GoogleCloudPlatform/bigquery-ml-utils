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

"""Tests for BigQuery TimeDiff custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class TimeDiffTest(tf.test.TestCase):

  def test_time_diff(self):
    time_a = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    time_b = tf.constant(['06:30:00.000000', '07:30:00.000000'])
    self.assertAllEqual(
        time_ops.time_diff(time_a, time_b, 'MICROSECOND'),
        tf.constant([3600000000, -3600000000], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.time_diff(time_a, time_b, 'MILLISECOND'),
        tf.constant([3600000, -3600000], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.time_diff(time_a, time_b, 'SECOND'),
        tf.constant([3600, -3600], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.time_diff(time_a, time_b, 'MINUTE'),
        tf.constant([60, -60], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.time_diff(time_a, time_b, 'HOUR'),
        tf.constant([1, -1], dtype=tf.int64),
    )

  def test_time_diff_invalid_time(self):
    time_a = tf.constant(['07:30:00.00000a', '06:30:00.000000'])
    time_b = tf.constant(['06:30:00.000000', '07:30:00.000000'])
    time_c = tf.constant(['06:30:00.000000', '07:30:00.00000b'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:30:00.00000a"',
    ):
      self.evaluate(time_ops.time_diff(time_a, time_b, 'HOUR'))
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:30:00.00000b"',
    ):
      self.evaluate(time_ops.time_diff(time_b, time_c, 'HOUR'))

  def test_time_diff_invalid_part(self):
    time_a = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    time_b = tf.constant(['06:30:00.000000', '07:30:00.000000'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in TimeDiff: DAY',
    ):
      self.evaluate(time_ops.time_diff(time_a, time_b, 'DAY'))

  def test_time_diff_different_shapes(self):
    time_a = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    time_b = tf.constant(
        ['06:30:00.000000', '07:30:00.000000', '07:30:00.000000']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'time_a and time_b must have the same shape, but are 2, 3',
    ):
      self.evaluate(time_ops.time_diff(time_a, time_b, 'HOUR'))


if __name__ == '__main__':
  tf.test.main()
