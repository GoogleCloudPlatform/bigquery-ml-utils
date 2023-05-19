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

"""Tests for BQML UNIX_SECONDS custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class UnixSecondsTest(tf.test.TestCase):

  def test_unix_seconds(self):
    timestamp_int = tf.constant(
        ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
    )
    self.assertAllEqual(
        timestamp_ops.unix_seconds(timestamp_int),
        tf.constant([1230219000, 1699713000], dtype=tf.int64),
    )

  def test_unix_seconds_invalid_timestamp(self):
    timestamp_int = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00.0 +0000']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timestamp in UnixSeconds: 2008-12-25 15:30:00 abc',
    ):
      self.evaluate(timestamp_ops.unix_seconds(timestamp_int))


if __name__ == '__main__':
  tf.test.main()
