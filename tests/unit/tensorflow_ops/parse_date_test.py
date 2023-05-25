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

"""Tests for BigQuery ParseDate custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class ParseDateTest(tf.test.TestCase):

  def test_parse_date(self):
    date = tf.constant(['Thursday Dec 25 2008', 'Thursday Feb  2 2023'])
    self.assertAllEqual(
        date_ops.parse_date('%A %b %e %Y', date),
        tf.constant(['2008-12-25', '2023-02-02']),
    )

  def test_parse_date_invalid_date(self):
    date = tf.constant(['Thursday aaa 25 2008', 'Thursday Feb  2 2023'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "Thursday aaa 25 2008"',
    ):
      self.evaluate(date_ops.parse_date('%A %b %e %Y', date))

  def test_parse_date_invalid_format(self):
    date = tf.constant(['2008-12-25', '2023-02-02'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        "Mismatch between format character 'a' and string character '2'",
    ):
      self.evaluate(date_ops.parse_date('abc', date))


if __name__ == '__main__':
  tf.test.main()
