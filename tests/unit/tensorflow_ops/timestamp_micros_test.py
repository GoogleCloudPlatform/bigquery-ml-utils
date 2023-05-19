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

"""Tests for BQML TIMESTAMP_MICROS custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampMicrosTest(tf.test.TestCase):

  def test_timestamp_micros(self):
    timestamp_int = tf.constant(
        [1230219000000000, 1699713000000000], dtype=tf.int64
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_micros(timestamp_int),
        tf.constant(
            ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
        ),
    )

  def test_timestamp_micros_out_range(self):
    timestamp_int = tf.constant(
        [9223372036854775801, 1699713000], dtype=tf.int64
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Timestamp value in TimestampMicros is out of allowed range',
    ):
      self.evaluate(timestamp_ops.timestamp_micros(timestamp_int))


if __name__ == '__main__':
  tf.test.main()
