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

"""Tests for BigQuery EXTRACT from datetime custom ops."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class ExtractFromDatetimeTest(tf.test.TestCase):

  def test_extract_from_datetime(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'MICROSECOND'),
        tf.constant([700000, 300000]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'MILLISECOND'),
        tf.constant([700, 300]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'SECOND'),
        tf.constant([56, 12]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'MINUTE'),
        tf.constant([34, 45]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'HOUR'),
        tf.constant([12, 23]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'DAYOFWEEK'),
        tf.constant([3, 3]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'DAY'),
        tf.constant([10, 14]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'DAYOFYEAR'),
        tf.constant([10, 73]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'WEEK'),
        tf.constant([2, 11]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'WEEK_TUESDAY'),
        tf.constant([2, 11]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'ISOWEEK'),
        tf.constant([2, 11]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'MONTH'),
        tf.constant([1, 3]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'QUARTER'),
        tf.constant([1, 1]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'YEAR'),
        tf.constant([2023, 2023]),
    )

    self.assertAllEqual(
        datetime_ops.extract_from_datetime(datetime, 'ISOYEAR'),
        tf.constant([2023, 2023]),
    )

  def test_extract_from_datetime_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-01-10"',
    ):
      self.evaluate(
          datetime_ops.extract_from_datetime(
              tf.constant(['2023-01-10', '2023-03-14']), 'MICROSECOND'
          )
      )

  def test_extract_from_datetime_invalid_part(self):
    datetime = tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in ExtractFromDatetime: MICRO',
    ):
      self.evaluate(datetime_ops.extract_from_datetime(datetime, 'MICRO'))

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in ExtractFromDatetime: DATE',
    ):
      self.evaluate(datetime_ops.extract_from_datetime(datetime, 'DATE'))


if __name__ == '__main__':
  tf.test.main()
