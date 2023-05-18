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

"""Tests for BQML TIMESTAMP from string custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampFromStringTest(tf.test.TestCase):

  def test_timestamp_from_string(self):
    timestamp = tf.constant(['2008-12-25 15:30:00+00', '2023-11-11 14:30:00'])
    self.assertAllEqual(
        timestamp_ops.timestamp_from_string(timestamp),
        tf.constant(
            ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
        ),
    )

  def test_timestamp_from_string_tz(self):
    timestamp = tf.constant(['2008-12-25 15:30:00', '2023-11-11 14:30:00'])
    self.assertAllEqual(
        timestamp_ops.timestamp_from_string(timestamp, 'America/Los_Angeles'),
        tf.constant(
            ['2008-12-25 23:30:00.0 +0000', '2023-11-11 22:30:00.0 +0000']
        ),
    )

  def test_timestamp_from_string_invalid_timezone(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid time zone in TimestampFromString: UtC',
    ):
      self.evaluate(timestamp_ops.timestamp_from_string(timestamp, 'UtC'))

  def test_timestamp_from_string_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00 111', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timestamp in TimestampFromString: 2008-12-25 15:30:00',
    ):
      self.evaluate(timestamp_ops.timestamp_from_string(timestamp))


if __name__ == '__main__':
  tf.test.main()
