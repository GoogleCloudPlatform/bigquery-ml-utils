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

"""Tests for BigQuery DateDiff custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateDiffTest(tf.test.TestCase):

  def test_date_diff(self):
    date_a = tf.constant(['2023-02-02', '2008-12-25'])
    date_b = tf.constant(['2008-12-25', '2023-02-02'])
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'DAY'),
        tf.constant([5152, -5152], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'WEEK'),
        tf.constant([736, -736], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'WEEK_TUESDAY'),
        tf.constant([736, -736], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'ISOWEEK'),
        tf.constant([736, -736], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'MONTH'),
        tf.constant([170, -170], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'QUARTER'),
        tf.constant([57, -57], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'YEAR'),
        tf.constant([15, -15], dtype=tf.int64),
    )
    self.assertAllEqual(
        date_ops.date_diff(date_a, date_b, 'ISOYEAR'),
        tf.constant([15, -15], dtype=tf.int64),
    )

  def test_date_diff_invalid_date(self):
    date_a = tf.constant(['2008-12-25 a', '2023-02-02'])
    date_b = tf.constant(['2008-12-25', '2023-02-02'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 a"',
    ):
      self.evaluate(date_ops.date_diff(date_a, date_b, 'DAY'))

  def test_date_diff_invalid_part(self):
    date_a = tf.constant(['2008-12-25', '2023-02-02'])
    date_b = tf.constant(['2008-12-25', '2023-02-02'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in DateDiff: SECOND',
    ):
      self.evaluate(date_ops.date_diff(date_a, date_b, 'SECOND'))


if __name__ == '__main__':
  tf.test.main()
