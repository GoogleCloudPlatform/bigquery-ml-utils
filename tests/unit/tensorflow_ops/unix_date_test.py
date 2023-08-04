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

"""Tests for BQML UnixDate custom ops."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class UnixDateTest(tf.test.TestCase):

  def test_unix_date(self):
    date = tf.constant(['2008-12-25', '1970-01-01'])
    self.assertAllEqual(
        date_ops.unix_date(date),
        tf.constant([14238, 0], dtype=tf.int64),
    )

  def test_unix_date_invalid_date(self):
    date = tf.constant(['2008-12-25 abc', 'abc'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 abc"',
    ):
      self.evaluate(date_ops.unix_date(date))


if __name__ == '__main__':
  tf.test.main()
