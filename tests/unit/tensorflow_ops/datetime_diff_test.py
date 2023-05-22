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

"""Tests for BigQuery datetime_diff custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeDiffTest(tf.test.TestCase):

  def test_datetime_diff(self):
    datetime_a = tf.constant(['2022-01-09 12:34:00', '2023-03-14 23:45:12.3'])
    datetime_b = tf.constant(['2023-01-10 12:34:56.7', '2022-03-04 23:45:00'])

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'MICROSECOND'),
        tf.constant([-31622456700000, 32400012300000], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'MILLISECOND'),
        tf.constant([-31622456700, 32400012300], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'second'),
        tf.constant([-31622456, 32400012], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'minute'),
        tf.constant([-527040, 540000], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'hour'),
        tf.constant([-8784, 9000], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'day'),
        tf.constant([-366, 375], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'Week'),
        tf.constant([-52, 54], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'week_monday'),
        tf.constant([-53, 54], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'week_tuesday'),
        tf.constant([-53, 54], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'week_wednesday'),
        tf.constant([-52, 53], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'week_thursday'),
        tf.constant([-52, 53], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'week_friday'),
        tf.constant([-52, 53], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'week_saturday'),
        tf.constant([-52, 54], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'isoweek'),
        tf.constant([-53, 54], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'month'),
        tf.constant([-12, 12], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'QUARTER'),
        tf.constant([-4, 4], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'year'),
        tf.constant([-1, 1], dtype=tf.int64),
    )

    self.assertAllEqual(
        datetime_ops.datetime_diff(datetime_a, datetime_b, 'isoyear'),
        tf.constant([-1, 1], dtype=tf.int64),
    )

  def test_datetime_diff_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-01-10"',
    ):
      self.evaluate(
          datetime_ops.datetime_diff(
              tf.constant(['2023-01-10', '2023-03-14']),
              tf.constant(['2023-01-10', '2023-03-14']),
              'MICROSECOND',
          )
      )

  def test_datetime_diff_invalid_part(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in DatetimeDiff: MICRO',
    ):
      self.evaluate(datetime_ops.datetime_diff(datetime, datetime, 'MICRO'))

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in DatetimeDiff: DATE',
    ):
      self.evaluate(datetime_ops.datetime_diff(datetime, datetime, 'DATE'))


if __name__ == '__main__':
  tf.test.main()
