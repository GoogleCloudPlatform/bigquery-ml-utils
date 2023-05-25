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

"""Tests for BigQuery DateSub custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateSubTest(tf.test.TestCase):

  def test_date_sub(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    self.assertAllEqual(
        date_ops.date_sub(date, interval, 'DAY'),
        tf.constant(['2008-12-23', '2023-01-31']),
    )
    self.assertAllEqual(
        date_ops.date_sub(date, interval, 'WEEK'),
        tf.constant(['2008-12-11', '2023-01-19']),
    )
    self.assertAllEqual(
        date_ops.date_sub(date, interval, 'MONTH'),
        tf.constant(['2008-10-25', '2022-12-02']),
    )
    self.assertAllEqual(
        date_ops.date_sub(date, interval, 'QUARTER'),
        tf.constant(['2008-06-25', '2022-08-02']),
    )
    self.assertAllEqual(
        date_ops.date_sub(date, interval, 'YEAR'),
        tf.constant(['2006-12-25', '2021-02-02']),
    )

  def test_date_sub_invalid_date(self):
    date = tf.constant(['2008-12-25 a', '2023-02-02'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 a"',
    ):
      self.evaluate(date_ops.date_sub(date, interval, 'DAY'))

  def test_date_sub_invalid_part(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in DateSub: SECOND',
    ):
      self.evaluate(date_ops.date_sub(date, interval, 'SECOND'))


if __name__ == '__main__':
  tf.test.main()
