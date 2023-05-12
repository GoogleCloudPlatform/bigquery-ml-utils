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

"""Tests for BigQuery datetime_from_components custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeFromComponentsTest(tf.test.TestCase):

  def test_datetime_from_components(self):
    year = tf.constant([2014, 2023], dtype=tf.int64)
    month = tf.constant([11, 10], dtype=tf.int64)
    day = tf.constant([1, 5], dtype=tf.int64)
    hour = tf.constant([10, 20], dtype=tf.int64)
    minute = tf.constant([14, 35], dtype=tf.int64)
    second = tf.constant([1, 5], dtype=tf.int64)

    self.assertAllEqual(
        datetime_ops.datetime_from_components(
            year, month, day, hour, minute, second
        ),
        tf.constant(
            ['2014-11-01 10:14:01.000000', '2023-10-05 20:35:05.000000']
        ),
    )

  def test_datetime_from_components_invalid(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        (
            "Errors in DatetimeFromComponents with input '20', '20', '20',"
            " '20', '20', '20': OUT_OF_RANGE: Input calculates to invalid"
            ' datetime: 0020-20-20 0020:20:20'
        ),
    ):
      self.evaluate(
          datetime_ops.datetime_from_components(
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        (
            'Invalid input in DatetimeFromComponents: all the inputs must have'
            ' the same shape.'
        ),
    ):
      self.evaluate(
          datetime_ops.datetime_from_components(
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20, 30], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
              tf.constant([10, 20], dtype=tf.int64),
          )
      )


if __name__ == '__main__':
  tf.test.main()
