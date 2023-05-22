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

"""Tests for BQML FORMAT_TIMESTAMP custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class FormatTimestampTest(tf.test.TestCase):

  def test_format_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    self.assertAllEqual(
        timestamp_ops.format_timestamp('%c', timestamp, 'America/Los_Angeles'),
        tf.constant(['Thu Dec 25 07:30:00 2008', 'Sat Nov 11 06:30:00 2023']),
    )
    self.assertAllEqual(
        timestamp_ops.format_timestamp('%b-%d-%Y', timestamp),
        tf.constant(['Dec-25-2008', 'Nov-11-2023']),
    )

  def test_format_timestamp_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 15:30:00 abc"',
    ):
      self.evaluate(
          timestamp_ops.format_timestamp('%c', timestamp, 'America/Los_Angeles')
      )

  def test_format_timestamp_invalid_zone(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Invalid time zone: uTc',
    ):
      self.evaluate(timestamp_ops.format_timestamp('%c', timestamp, 'uTc'))

  def test_format_invalid_format(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    self.assertAllEqual(
        timestamp_ops.format_timestamp('abc', timestamp, 'America/Los_Angeles'),
        tf.constant(['abc', 'abc']),
    )


if __name__ == '__main__':
  tf.test.main()
