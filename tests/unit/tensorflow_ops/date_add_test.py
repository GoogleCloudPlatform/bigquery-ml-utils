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

"""Tests for BigQuery DateAdd custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateAddTest(tf.test.TestCase):

  def test_date_add(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    self.assertAllEqual(
        date_ops.date_add(date, interval, 'DAY'),
        tf.constant(['2008-12-27', '2023-02-04']),
    )
    self.assertAllEqual(
        date_ops.date_add(date, interval, 'WEEK'),
        tf.constant(['2009-01-08', '2023-02-16']),
    )
    self.assertAllEqual(
        date_ops.date_add(date, interval, 'MONTH'),
        tf.constant(['2009-02-25', '2023-04-02']),
    )
    self.assertAllEqual(
        date_ops.date_add(date, interval, 'QUARTER'),
        tf.constant(['2009-06-25', '2023-08-02']),
    )
    self.assertAllEqual(
        date_ops.date_add(date, interval, 'YEAR'),
        tf.constant(['2010-12-25', '2025-02-02']),
    )

  def test_date_add_invalid_date(self):
    date = tf.constant(['2008-12-25 a', '2023-02-02'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 a"',
    ):
      self.evaluate(date_ops.date_add(date, interval, 'DAY'))

  def test_date_add_invalid_part(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    interval = tf.constant([2, 2], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in DateAdd: SECOND',
    ):
      self.evaluate(date_ops.date_add(date, interval, 'SECOND'))


if __name__ == '__main__':
  tf.test.main()
