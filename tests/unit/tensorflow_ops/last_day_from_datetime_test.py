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

"""Tests for BigQuery LAST_DAY from datetime custom ops."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class LastDayFromDatetimeTest(tf.test.TestCase):

  def test_last_day_from_datetime(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'WEEK'),
        tf.constant(['2023-01-14', '2023-03-18']),
    )

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'WEEK_TUESDAY'),
        tf.constant(['2023-01-16', '2023-03-20']),
    )

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'ISOWEEK'),
        tf.constant(['2023-01-15', '2023-03-19']),
    )

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'MONTH'),
        tf.constant(['2023-01-31', '2023-03-31']),
    )

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'QUARTER'),
        tf.constant(['2023-03-31', '2023-03-31']),
    )

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'YEAR'),
        tf.constant(['2023-12-31', '2023-12-31']),
    )

    self.assertAllEqual(
        datetime_ops.last_day_from_datetime(datetime, 'ISOYEAR'),
        tf.constant(['2023-12-31', '2023-12-31']),
    )

  def test_last_day_from_datetime_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid datetime in LastDayFromDatetime: 2023-01-10',
    ):
      self.evaluate(
          datetime_ops.last_day_from_datetime(
              tf.constant(['2023-01-10', '2023-03-14']), 'YEAR'
          )
      )

  def test_last_day_from_datetime_invalid_part(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in LastDayFromDatetime: MICRO',
    ):
      self.evaluate(datetime_ops.last_day_from_datetime(datetime, 'MICRO'))

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in LastDayFromDatetime: DATE',
    ):
      self.evaluate(datetime_ops.last_day_from_datetime(datetime, 'DATE'))


if __name__ == '__main__':
  tf.test.main()
