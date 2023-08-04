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

"""Tests for BigQuery LastDayFromDate custom ops."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class LastDayFromDateTest(tf.test.TestCase):

  def test_last_day_from_date(self):
    date = tf.constant(['2023-01-10', '2023-03-14'])

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'WEEK'),
        tf.constant(['2023-01-14', '2023-03-18']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'WEEK_TUESDAY'),
        tf.constant(['2023-01-16', '2023-03-20']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'ISOWEEK'),
        tf.constant(['2023-01-15', '2023-03-19']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'MONTH'),
        tf.constant(['2023-01-31', '2023-03-31']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'QUARTER'),
        tf.constant(['2023-03-31', '2023-03-31']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'YEAR'),
        tf.constant(['2023-12-31', '2023-12-31']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(date, 'ISOYEAR'),
        tf.constant(['2023-12-31', '2023-12-31']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(tf.constant(['2008-11-25']), 'MONTH'),
        tf.constant(['2008-11-30']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(tf.constant(['2008-11-25'])),
        tf.constant(['2008-11-30']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(tf.constant(['2008-11-25']), 'YEAR'),
        tf.constant(['2008-12-31']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(tf.constant(['2008-11-10']), 'WEEK'),
        tf.constant(['2008-11-15']),
    )

    self.assertAllEqual(
        date_ops.last_day_from_date(tf.constant(['2008-11-10']), 'WEEK_MONDAY'),
        tf.constant(['2008-11-16']),
    )

  def test_last_day_from_date_invalid_date(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-01-40"',
    ):
      self.evaluate(
          date_ops.last_day_from_date(
              tf.constant(['2023-01-40', '2023-03-140']), 'YEAR'
          )
      )

  def test_last_day_from_date_invalid_part(self):
    date = tf.constant(['2023-01-10', '2023-03-14'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in LastDayFromDate: MICRO',
    ):
      self.evaluate(date_ops.last_day_from_date(date, 'MICRO'))

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in LastDayFromDate: DATE',
    ):
      self.evaluate(date_ops.last_day_from_date(date, 'DATE'))


if __name__ == '__main__':
  tf.test.main()
