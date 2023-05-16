# Copyright 2022 Google LLC
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

"""Tests for BQML STRING from timestamp custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class StringFromTimestampTest(tf.test.TestCase):

  def test_string_from_timestamp(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
    )
    self.assertAllEqual(
        timestamp_ops.string_from_timestamp(timestamp),
        tf.constant(
            ['2023-01-10 00:00:56.700+00', '2023-03-14 11:11:12.300+00']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.string_from_timestamp(timestamp, 'America/Los_Angeles'),
        tf.constant(
            ['2023-01-09 16:00:56.700-08', '2023-03-14 04:11:12.300-07']
        ),
    )

  def test_string_from_timestamp_invalid_timezone(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timezone in StringFromTimestamp: UtC',
    ):
      self.evaluate(timestamp_ops.string_from_timestamp(timestamp, 'UtC'))

  def test_string_from_timestamp_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3 +1234']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timestamp in StringFromTimestamp: 2023-01-10 12:34:56.7',
    ):
      self.evaluate(timestamp_ops.string_from_timestamp(timestamp))


if __name__ == '__main__':
  tf.test.main()
