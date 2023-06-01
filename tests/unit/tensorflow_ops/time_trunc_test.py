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

"""Tests for BigQuery TimeTrunc custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class TimeTruncTest(tf.test.TestCase):

  def test_time_trunc(self):
    time = tf.constant(['07:31:15.123456', '06:22:23.123456'])
    self.assertAllEqual(
        time_ops.time_trunc(time, 'MICROSECOND'),
        tf.constant(['07:31:15.123456', '06:22:23.123456'], dtype=tf.string),
    )
    self.assertAllEqual(
        time_ops.time_trunc(time, 'MILLISECOND'),
        tf.constant(['07:31:15.123', '06:22:23.123'], dtype=tf.string),
    )
    self.assertAllEqual(
        time_ops.time_trunc(time, 'SECOND'),
        tf.constant(['07:31:15', '06:22:23'], dtype=tf.string),
    )
    self.assertAllEqual(
        time_ops.time_trunc(time, 'MINUTE'),
        tf.constant(['07:31:00', '06:22:00'], dtype=tf.string),
    )
    self.assertAllEqual(
        time_ops.time_trunc(time, 'HOUR'),
        tf.constant(['07:00:00', '06:00:00'], dtype=tf.string),
    )

  def test_time_trunc_invalid_time(self):
    time = tf.constant(['07:31:15.123456a', '06:22:23.123456'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:31:15.123456a"',
    ):
      self.evaluate(time_ops.time_trunc(time, 'HOUR'))

  def test_time_trunc_invalid_part(self):
    time = tf.constant(['07:31:15.123456', '06:22:23.123456'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in TimeTrunc: DAY',
    ):
      self.evaluate(time_ops.time_trunc(time, 'DAY'))


if __name__ == '__main__':
  tf.test.main()
