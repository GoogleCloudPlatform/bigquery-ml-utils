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

"""Tests for BigQuery EXTRACT from date custom ops."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class ExtractFromDateTest(tf.test.TestCase):

  def test_extract_from_date(self):
    date = tf.constant(['2023-01-10', '2023-03-14'])

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'DAYOFWEEK'),
        tf.constant([3, 3]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'DAY'),
        tf.constant([10, 14]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'DAYOFYEAR'),
        tf.constant([10, 73]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'WEEK'),
        tf.constant([2, 11]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'WEEK_TUESDAY'),
        tf.constant([2, 11]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'ISOWEEK'),
        tf.constant([2, 11]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'MONTH'),
        tf.constant([1, 3]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'QUARTER'),
        tf.constant([1, 1]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'YEAR'),
        tf.constant([2023, 2023]),
    )

    self.assertAllEqual(
        date_ops.extract_from_date(date, 'ISOYEAR'),
        tf.constant([2023, 2023]),
    )

  def test_extract_from_date_invalid_date(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid date in ExtractFromDate: 2023-01-32',
    ):
      self.evaluate(
          date_ops.extract_from_date(
              tf.constant(['2023-01-32', '2023-03-14']), 'DAYOFWEEK'
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid date in ExtractFromDate: 2023-03',
    ):
      self.evaluate(
          date_ops.extract_from_date(
              tf.constant(['2023-01-12', '2023-03']), 'ISOWEEK'
          )
      )

  def test_extract_from_date_invalid_part(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in ExtractFromDate: hour',
    ):
      self.evaluate(
          date_ops.extract_from_date(
              tf.constant(['2023-01-10', '2023-03-14']), 'HOUR'
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in ExtractFromDate: second',
    ):
      self.evaluate(
          date_ops.extract_from_date(
              tf.constant(['2023-01-10', '2023-03-14']), 'SECOND'
          )
      )


if __name__ == '__main__':
  tf.test.main()
