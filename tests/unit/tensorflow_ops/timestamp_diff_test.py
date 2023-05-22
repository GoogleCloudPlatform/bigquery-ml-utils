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

"""Tests for BQML TIMESTAMP_DIFF custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampDiffTest(tf.test.TestCase):

  def test_timestamp_diff(self):
    timestamp_a = tf.constant(
        ['2023-11-11 14:30:00+00', '2008-12-25 15:30:00+00']
    )
    timestamp_b = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'MICROSECOND'),
        tf.constant([469494000000000, -469494000000000]),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'MILLISECOND'),
        tf.constant([469494000000, -469494000000]),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'SECOND'),
        tf.constant([469494000, -469494000]),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'MINUTE'),
        tf.constant([7824900, -7824900]),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'HOUR'),
        tf.constant([130415, -130415]),
    )

    self.assertAllEqual(
        timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'DAY'),
        tf.constant([5433, -5433]),
    )

  def test_timestamp_diff_invalid_part(self):
    timestamp_a = tf.constant(
        ['2023-11-11 14:30:00+00', '2008-12-25 15:30:00+00']
    )
    timestamp_b = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in TimestampDiff: RandomPart',
    ):
      self.evaluate(
          timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'RandomPart')
      )

  def test_timestamp_diff_invalid_timestamp(self):
    timestamp_a = tf.constant(
        ['2023-11-11 14:30:00 abc', '2008-12-25 15:30:00+00']
    )
    timestamp_b = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-11-11 14:30:00 abc"',
    ):
      self.evaluate(
          timestamp_ops.timestamp_diff(timestamp_a, timestamp_b, 'DAY')
      )


if __name__ == '__main__':
  tf.test.main()
