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

"""Tests for BigQuery TimeFromComponents custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class TimeFromComponentsTest(tf.test.TestCase):

  def test_time_from_components(self):
    hour = tf.constant([10, 20], dtype=tf.int64)
    minute = tf.constant([14, 35], dtype=tf.int64)
    second = tf.constant([1, 5], dtype=tf.int64)

    self.assertAllEqual(
        time_ops.time_from_components(hour, minute, second),
        tf.constant(['10:14:01', '20:35:05']),
    )

  def test_time_from_components_invalid_input(self):
    hour = tf.constant([25, 20], dtype=tf.int64)
    minute = tf.constant([14, 35], dtype=tf.int64)
    second = tf.constant([1, 5], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Input calculates to invalid time: 25:14:01',
    ):
      self.evaluate(time_ops.time_from_components(hour, minute, second))

  def test_time_from_components_differenet_shapes(self):
    hour = tf.constant([25, 20], dtype=tf.int64)
    minute = tf.constant([14, 35], dtype=tf.int64)
    second = tf.constant([1], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        (
            'Errors in TimeFromComponents: Inputs must have the same shape, but'
            ' are: 2, 2, 1'
        ),
    ):
      self.evaluate(time_ops.time_from_components(hour, minute, second))


if __name__ == '__main__':
  tf.test.main()
