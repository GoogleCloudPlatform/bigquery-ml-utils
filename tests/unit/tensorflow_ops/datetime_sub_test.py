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

"""Tests for BigQuery datetime_sub custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeSubTest(tf.test.TestCase):

  def test_datetime_sub(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    interval = tf.constant([10, 20], dtype=tf.int64)

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'MICROSECOND'),
        tf.constant(
            ['2023-01-10 12:34:56.699990', '2023-03-14 23:45:12.299980']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'MILLISECOND'),
        tf.constant(
            ['2023-01-10 12:34:56.690000', '2023-03-14 23:45:12.280000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'SECOND'),
        tf.constant(
            ['2023-01-10 12:34:46.700000', '2023-03-14 23:44:52.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'minute'),
        tf.constant(
            ['2023-01-10 12:24:56.700000', '2023-03-14 23:25:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'hour'),
        tf.constant(
            ['2023-01-10 02:34:56.700000', '2023-03-14 03:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'day'),
        tf.constant(
            ['2022-12-31 12:34:56.700000', '2023-02-22 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'week'),
        tf.constant(
            ['2022-11-01 12:34:56.700000', '2022-10-25 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'month'),
        tf.constant(
            ['2022-03-10 12:34:56.700000', '2021-07-14 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'quarter'),
        tf.constant(
            ['2020-07-10 12:34:56.700000', '2018-03-14 23:45:12.300000']
        ),
    )

    self.assertAllEqual(
        datetime_ops.datetime_sub(datetime, interval, 'year'),
        tf.constant(
            ['2013-01-10 12:34:56.700000', '2003-03-14 23:45:12.300000']
        ),
    )

  def test_datetime_sub_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid datetime in DatetimeSub: 2023-01-10',
    ):
      self.evaluate(
          datetime_ops.datetime_sub(
              tf.constant(['2023-01-10', '2023-03-14']),
              tf.constant([10, 20], dtype=tf.int64),
              'MICROSECOND',
          )
      )

  def test_datetime_sub_invalid_part(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in DatetimeSub: MICRO',
    ):
      self.evaluate(
          datetime_ops.datetime_sub(
              datetime, tf.constant([10, 20], dtype=tf.int64), 'MICRO'
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in DatetimeSub: DATE',
    ):
      self.evaluate(
          datetime_ops.datetime_sub(
              datetime, tf.constant([10, 20], dtype=tf.int64), 'DATE'
          )
      )


if __name__ == '__main__':
  tf.test.main()
