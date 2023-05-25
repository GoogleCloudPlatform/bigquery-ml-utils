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

"""Tests for BigQuery DateFromTimestamp custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateFromTimestampTest(tf.test.TestCase):

  def test_date_from_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 06:30:00+00', '2023-11-11 05:30:00+00']
    )
    self.assertAllEqual(
        date_ops.date_from_timestamp(timestamp),
        tf.constant(['2008-12-25', '2023-11-11']),
    )
    self.assertAllEqual(
        date_ops.date_from_timestamp(timestamp, 'America/Los_Angeles'),
        tf.constant(['2008-12-24', '2023-11-10']),
    )

  def test_date_from_timestamp_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 15:30:00 abc"',
    ):
      self.evaluate(date_ops.date_from_timestamp(timestamp))

  def test_date_from_timestamp_invalid_tz(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Invalid time zone: UtC',
    ):
      self.evaluate(date_ops.date_from_timestamp(timestamp, 'UtC'))


if __name__ == '__main__':
  tf.test.main()
