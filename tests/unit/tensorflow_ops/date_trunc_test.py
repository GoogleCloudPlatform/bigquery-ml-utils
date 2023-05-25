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

"""Tests for BigQuery DateTrunc custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateTruncTest(tf.test.TestCase):

  def test_date_trunc(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    self.assertAllEqual(
        date_ops.date_trunc(date, 'DAY'),
        tf.constant(['2008-12-25', '2023-02-02']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'WEEK'),
        tf.constant(['2008-12-21', '2023-01-29']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'WEEK_TUESDAY'),
        tf.constant(['2008-12-23', '2023-01-31']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'ISOWEEK'),
        tf.constant(['2008-12-22', '2023-01-30']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'MONTH'),
        tf.constant(['2008-12-01', '2023-02-01']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'QUARTER'),
        tf.constant(['2008-10-01', '2023-01-01']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'YEAR'),
        tf.constant(['2008-01-01', '2023-01-01']),
    )
    self.assertAllEqual(
        date_ops.date_trunc(date, 'ISOYEAR'),
        tf.constant(['2007-12-31', '2023-01-02']),
    )

  def test_date_trunc_invalid_date(self):
    date = tf.constant(['2008-12-25 a', '2023-02-02'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 a"',
    ):
      self.evaluate(date_ops.date_trunc(date, 'DAY'))

  def test_date_trunc_invalid_part(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in DateTrunc: SECOND',
    ):
      self.evaluate(date_ops.date_trunc(date, 'SECOND'))


if __name__ == '__main__':
  tf.test.main()
