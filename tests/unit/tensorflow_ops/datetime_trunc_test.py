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

"""Tests for BigQuery datetime_trunc custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeTruncTest(tf.test.TestCase):

  def test_datetime_trunc(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'MICROSECOND'),
        tf.constant(
            ['2023-01-10 12:34:56.700000', '2023-03-14 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'MILLISECOND'),
        tf.constant(
            ['2023-01-10 12:34:56.700000', '2023-03-14 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'second'),
        tf.constant(
            ['2023-01-10 12:34:56.000000', '2023-03-14 23:45:12.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'minute'),
        tf.constant(
            ['2023-01-10 12:34:00.000000', '2023-03-14 23:45:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'hour'),
        tf.constant(
            ['2023-01-10 12:00:00.000000', '2023-03-14 23:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'day'),
        tf.constant(
            ['2023-01-10 00:00:00.000000', '2023-03-14 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'Week'),
        tf.constant(
            ['2023-01-08 00:00:00.000000', '2023-03-12 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'week_monday'),
        tf.constant(
            ['2023-01-09 00:00:00.000000', '2023-03-13 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'week_tuesday'),
        tf.constant(
            ['2023-01-10 00:00:00.000000', '2023-03-14 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'week_wednesday'),
        tf.constant(
            ['2023-01-04 00:00:00.000000', '2023-03-08 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'week_thursday'),
        tf.constant(
            ['2023-01-05 00:00:00.000000', '2023-03-09 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'week_friday'),
        tf.constant(
            ['2023-01-06 00:00:00.000000', '2023-03-10 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'week_saturday'),
        tf.constant(
            ['2023-01-07 00:00:00.000000', '2023-03-11 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'isoweek'),
        tf.constant(
            ['2023-01-09 00:00:00.000000', '2023-03-13 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'month'),
        tf.constant(
            ['2023-01-01 00:00:00.000000', '2023-03-01 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'QUARTER'),
        tf.constant(
            ['2023-01-01 00:00:00.000000', '2023-01-01 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'year'),
        tf.constant(
            ['2023-01-01 00:00:00.000000', '2023-01-01 00:00:00.000000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_trunc(datetime, 'isoyear'),
        tf.constant(
            ['2023-01-02 00:00:00.000000', '2023-01-02 00:00:00.000000']
        ),
    )

  def test_datetime_trunc_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-01-10"',
    ):
      self.evaluate(
          datetime_ops.datetime_trunc(
              tf.constant(['2023-01-10', '2023-03-14']),
              'MICROSECOND',
          )
      )

  def test_datetime_trunc_invalid_part(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in DatetimeTrunc: MICRO',
    ):
      self.evaluate(datetime_ops.datetime_trunc(datetime, 'MICRO'))

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in DatetimeTrunc: DATE',
    ):
      self.evaluate(datetime_ops.datetime_trunc(datetime, 'DATE'))


if __name__ == '__main__':
  tf.test.main()
