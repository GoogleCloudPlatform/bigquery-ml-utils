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

"""Tests for BigQuery ExtractFromTime custom op."""

from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class ExtractFromTimeTest(tf.test.TestCase):

  def test_extract_from_time(self):
    time = tf.constant(['07:31:15.123456', '06:22:23.123456'])
    self.assertAllEqual(
        time_ops.extract_from_time(time, 'MICROSECOND'),
        tf.constant([123456, 123456], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.extract_from_time(time, 'MILLISECOND'),
        tf.constant([123, 123], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.extract_from_time(time, 'SECOND'),
        tf.constant([15, 23], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.extract_from_time(time, 'MINUTE'),
        tf.constant([31, 22], dtype=tf.int64),
    )
    self.assertAllEqual(
        time_ops.extract_from_time(time, 'HOUR'),
        tf.constant([7, 6], dtype=tf.int64),
    )

  def test_extract_from_time_invalid_time(self):
    time = tf.constant(['07:30:00.00000a', '06:30:00.000000'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "07:30:00.00000a"',
    ):
      self.evaluate(time_ops.extract_from_time(time, 'HOUR'))

  def test_extract_from_time_invalid_part(self):
    time = tf.constant(['07:30:00.000000', '06:30:00.000000'])
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Unsupported part in ExtractFromTime: DAY',
    ):
      self.evaluate(time_ops.extract_from_time(time, 'DAY'))


if __name__ == '__main__':
  tf.test.main()
