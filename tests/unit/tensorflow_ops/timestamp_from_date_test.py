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

"""Tests for BQML TIMESTAMP from date custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampFromDateTest(tf.test.TestCase):

  def test_timestamp_from_date(self):
    date = tf.constant(['2023-02-02', '2021-03-05'])
    self.assertAllEqual(
        timestamp_ops.timestamp_from_date(date),
        tf.constant(
            ['2023-02-02 00:00:00.0 +0000', '2021-03-05 00:00:00.0 +0000']
        ),
    )

  def test_timestamp_from_date_tz(self):
    date = tf.constant(['2023-02-02', '2021-03-05'])
    self.assertAllEqual(
        timestamp_ops.timestamp_from_date(date, 'America/Los_Angeles'),
        tf.constant(
            ['2023-02-02 08:00:00.0 +0000', '2021-03-05 08:00:00.0 +0000']
        ),
    )

  def test_timestamp_from_date_invalid_timezone(self):
    date = tf.constant(['2023-02-02', '2021-03-05'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Invalid time zone: UtC',
    ):
      self.evaluate(timestamp_ops.timestamp_from_date(date, 'UtC'))

  def test_timestamp_from_date_invalid_date(self):
    date = tf.constant(['2021-03-05 00:00:00 UTC', '2021-03-05'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2021-03-05 00:00:00 UTC"',
    ):
      self.evaluate(timestamp_ops.timestamp_from_date(date))


if __name__ == '__main__':
  tf.test.main()
