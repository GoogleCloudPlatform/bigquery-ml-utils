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

"""Tests for BigQuery FormatDate custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class FormatDateTest(tf.test.TestCase):

  def test_format_date(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    self.assertAllEqual(
        date_ops.format_date('%A %b %e %Y', date),
        tf.constant(['Thursday Dec 25 2008', 'Thursday Feb  2 2023']),
    )

  def test_format_date_invalid_date(self):
    date = tf.constant(['2008-12-25 a', '2023-02-02'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 a"',
    ):
      self.evaluate(date_ops.format_date('%A %b %e %Y', date))

  def test_format_date_invalid_format(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    self.assertAllEqual(
        date_ops.format_date('abc', date),
        tf.constant(['abc', 'abc']),
    )


if __name__ == '__main__':
  tf.test.main()
