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

"""Tests for BigQuery datetime_add custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeAddTest(tf.test.TestCase):

  def test_datetime_add(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    interval = tf.constant([10, 20], dtype=tf.int64)

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'MICROSECOND'),
        tf.constant(
            ['2023-01-10 12:34:56.700010', '2023-03-14 23:45:12.300020']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'MILLISECOND'),
        tf.constant(
            ['2023-01-10 12:34:56.710000', '2023-03-14 23:45:12.320000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'SECOND'),
        tf.constant(
            ['2023-01-10 12:35:06.700000', '2023-03-14 23:45:32.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'minute'),
        tf.constant(
            ['2023-01-10 12:44:56.700000', '2023-03-15 00:05:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'hour'),
        tf.constant(
            ['2023-01-10 22:34:56.700000', '2023-03-15 19:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'day'),
        tf.constant(
            ['2023-01-20 12:34:56.700000', '2023-04-03 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'week'),
        tf.constant(
            ['2023-03-21 12:34:56.700000', '2023-08-01 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'month'),
        tf.constant(
            ['2023-11-10 12:34:56.700000', '2024-11-14 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'quarter'),
        tf.constant(
            ['2025-07-10 12:34:56.700000', '2028-03-14 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_add(datetime, interval, 'year'),
        tf.constant(
            ['2033-01-10 12:34:56.700000', '2043-03-14 23:45:12.300000']
        ),
    )

  def test_datetime_add_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid datetime in DatetimeAdd: 2023-01-10',
    ):
      self.evaluate(
          datetime_ops.datetime_add(
              tf.constant(['2023-01-10', '2023-03-14']),
              tf.constant([10, 20], dtype=tf.int64),
              'MICROSECOND',
          )
      )

  def test_datetime_add_invalid_part(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in DatetimeAdd: MICRO',
    ):
      self.evaluate(
          datetime_ops.datetime_add(
              datetime, tf.constant([10, 20], dtype=tf.int64), 'MICRO'
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in DatetimeAdd: DATE',
    ):
      self.evaluate(
          datetime_ops.datetime_add(
              datetime, tf.constant([10, 20], dtype=tf.int64), 'DATE'
          )
      )


if __name__ == '__main__':
  tf.test.main()
