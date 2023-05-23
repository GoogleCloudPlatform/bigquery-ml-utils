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

"""Tests for BQML TIMESTAMP_ADD custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampAddTest(tf.test.TestCase):

  def test_timestamp_add(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    diff = tf.constant([500000, 500000], dtype=tf.int64)
    self.assertAllEqual(
        timestamp_ops.timestamp_add(timestamp, diff, 'MICROSECOND'),
        tf.constant(
            ['2008-12-25 15:30:00.5 +0000', '2023-11-11 14:30:00.5 +0000']
        ),
    )
    diff = tf.constant([5000, 5000], dtype=tf.int64)
    self.assertAllEqual(
        timestamp_ops.timestamp_add(timestamp, diff, 'MILLISECOND'),
        tf.constant(
            ['2008-12-25 15:30:05.0 +0000', '2023-11-11 14:30:05.0 +0000']
        ),
    )
    diff = tf.constant([50, 50], dtype=tf.int64)
    self.assertAllEqual(
        timestamp_ops.timestamp_add(timestamp, diff, 'SECOND'),
        tf.constant(
            ['2008-12-25 15:30:50.0 +0000', '2023-11-11 14:30:50.0 +0000']
        ),
    )
    diff = tf.constant([1, 1], dtype=tf.int64)
    self.assertAllEqual(
        timestamp_ops.timestamp_add(timestamp, diff, 'MINUTE'),
        tf.constant(
            ['2008-12-25 15:31:00.0 +0000', '2023-11-11 14:31:00.0 +0000']
        ),
    )
    diff = tf.constant([2, 2], dtype=tf.int64)
    self.assertAllEqual(
        timestamp_ops.timestamp_add(timestamp, diff, 'HOUR'),
        tf.constant(
            ['2008-12-25 17:30:00.0 +0000', '2023-11-11 16:30:00.0 +0000']
        ),
    )
    diff = tf.constant([3, 3], dtype=tf.int64)
    self.assertAllEqual(
        timestamp_ops.timestamp_add(timestamp, diff, 'DAY'),
        tf.constant(
            ['2008-12-28 15:30:00.0 +0000', '2023-11-14 14:30:00.0 +0000']
        ),
    )

  def test_timestamp_add_invalid_part(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    diff = tf.constant([500000, 500000], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in TimestampAdd: RandomPart',
    ):
      self.evaluate(timestamp_ops.timestamp_add(timestamp, diff, 'RandomPart'))

  def test_timestamp_add_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00+00']
    )
    diff = tf.constant([500000, 500000], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 15:30:00 abc"',
    ):
      self.evaluate(timestamp_ops.timestamp_add(timestamp, diff, 'DAY'))


if __name__ == '__main__':
  tf.test.main()
