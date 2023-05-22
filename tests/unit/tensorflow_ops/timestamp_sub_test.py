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

"""Tests for BQML TIMESTAMP_SUB custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampSubTest(tf.test.TestCase):

  def test_timestamp_sub(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_sub(timestamp, 500000, 'MICROSECOND'),
        tf.constant(
            ['2008-12-25 15:29:59.5 +0000', '2023-11-11 14:29:59.5 +0000']
        ),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_sub(timestamp, 5000, 'MILLISECOND'),
        tf.constant(
            ['2008-12-25 15:29:55.0 +0000', '2023-11-11 14:29:55.0 +0000']
        ),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_sub(timestamp, 50, 'SECOND'),
        tf.constant(
            ['2008-12-25 15:29:10.0 +0000', '2023-11-11 14:29:10.0 +0000']
        ),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_sub(timestamp, 1, 'MINUTE'),
        tf.constant(
            ['2008-12-25 15:29:00.0 +0000', '2023-11-11 14:29:00.0 +0000']
        ),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_sub(timestamp, 2, 'HOUR'),
        tf.constant(
            ['2008-12-25 13:30:00.0 +0000', '2023-11-11 12:30:00.0 +0000']
        ),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_sub(timestamp, 3, 'DAY'),
        tf.constant(
            ['2008-12-22 15:30:00.0 +0000', '2023-11-08 14:30:00.0 +0000']
        ),
    )

  def test_timestamp_sub_invalid_part(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in TimestampSub: RandomPart',
    ):
      self.evaluate(
          timestamp_ops.timestamp_sub(timestamp, 500000, 'RandomPart')
      )

  def test_timestamp_sub_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 15:30:00 abc"',
    ):
      self.evaluate(timestamp_ops.timestamp_sub(timestamp, 500000, 'DAY'))


if __name__ == '__main__':
  tf.test.main()
