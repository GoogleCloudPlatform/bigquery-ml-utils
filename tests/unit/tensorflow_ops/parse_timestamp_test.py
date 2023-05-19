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

"""Tests for BQML PARSE_TIMESTAMP custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class ParseTimestampTest(tf.test.TestCase):

  def test_parse_timestamp(self):
    timestamp = tf.constant(
        ['Thu Dec 25 15:30:00 2008', 'Sat Nov 11 14:30:00 2023']
    )
    self.assertAllEqual(
        timestamp_ops.parse_timestamp('%c', timestamp, 'America/Los_Angeles'),
        tf.constant(
            ['2008-12-25 23:30:00.0 +0000', '2023-11-11 22:30:00.0 +0000']
        ),
    )

    timestamp = tf.constant(['Dec-25-2008', 'Nov-11-2023'])
    self.assertAllEqual(
        timestamp_ops.parse_timestamp('%b-%d-%Y', timestamp),
        tf.constant(
            ['2008-12-25 00:00:00.0 +0000', '2023-11-11 00:00:00.0 +0000']
        ),
    )

  def test_parse_timestamp_invalid_timestamp(self):
    timestamp = tf.constant(
        ['Thu Dec 25 15:30:00 abc', 'Sat Nov 11 14:30:00 2023']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        "Mismatch between format '%c' and string 'Thu Dec 25 15:30:00 abc' in"
        ' ParseTimestamp',
    ):
      self.evaluate(
          timestamp_ops.parse_timestamp('%c', timestamp, 'America/Los_Angeles')
      )

  def test_parse_timestamp_invalid_zone(self):
    timestamp = tf.constant(
        ['Thu Dec 25 15:30:00 abc', 'Sat Nov 11 14:30:00 2023']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid time zone in ParseTimestamp: uTc',
    ):
      self.evaluate(timestamp_ops.parse_timestamp('%c', timestamp, 'uTc'))

  def test_parse_timestamp_invalid_format(self):
    timestamp = tf.constant(
        ['Thu Dec 25 15:30:00 abc', 'Sat Nov 11 14:30:00 2023']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        "Mismatch between format 'abc' and string 'Thu Dec 25 15:30:00 abc' in"
        ' ParseTimestamp',
    ):
      self.evaluate(timestamp_ops.parse_timestamp('abc', timestamp))


if __name__ == '__main__':
  tf.test.main()
