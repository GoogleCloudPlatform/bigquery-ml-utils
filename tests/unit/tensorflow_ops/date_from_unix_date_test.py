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

"""Tests for BigQuery DateFromUnixDate custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateFromUnixDateTest(tf.test.TestCase):

  def test_date_from_unix_date(self):
    int64_ = tf.constant([14238, 14239, 0], dtype=tf.int64)
    self.assertAllEqual(
        date_ops.date_from_unix_date(int64_),
        tf.constant(['2008-12-25', '2008-12-26', '1970-01-01']),
    )

  def test_date_from_unix_date_invalid_int64_input(self):
    int64_ = tf.constant(
        [-1000000, 10000000],
        dtype=tf.int64,
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'DATE value is out of allowed range: from 0001-01-01 to 9999-12-31.',
    ):
      self.evaluate(date_ops.date_from_unix_date(int64_))


if __name__ == '__main__':
  tf.test.main()
