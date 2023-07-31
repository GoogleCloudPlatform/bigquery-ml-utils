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

"""Tests for BigQuery CastToTimeFromString custom op."""


from bigquery_ml_utils.tensorflow_ops import time_ops
import tensorflow as tf


class CastToTimeFromStringTest(tf.test.TestCase):

  def test_cast_to_time_from_string_without_format(self):
    time_string = tf.constant(['07:31:15', '14:30:00'])
    self.assertAllEqual(
        time_ops.cast_to_time_from_string(time_string),
        tf.constant(['07:31:15', '14:30:00']),
    )

  def test_cast_to_time_from_string_with_format(self):
    self.assertAllEqual(
        time_ops.cast_to_time_from_string(
            tf.constant(['03:30 P.M.', '12:00 p.m.']), 'HH:MI P.M.'
        ),
        tf.constant(['15:30:00', '12:00:00']),
    )
    self.assertAllEqual(
        time_ops.cast_to_time_from_string(
            tf.constant(['03:30 A.M.', '03:30 a.m.']), 'HH12:MI A.M.'
        ),
        tf.constant(['03:30:00', '03:30:00']),
    )
    self.assertAllEqual(
        time_ops.cast_to_time_from_string(
            tf.constant(['17:00:53.110', '01:05:07.16']), 'HH24:MI:SS.FF3'
        ),
        tf.constant(['17:00:53.110', '01:05:07.160']),
    )

  def test_cast_to_time_from_string_invalid_string(self):
    time_string = tf.constant(['02:02:01.15290a', '15:30:00.152903'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Invalid time string "02:02:01.15290a"',
    ):
      self.evaluate(time_ops.cast_to_time_from_string(time_string))

  def test_cast_to_time_from_string_invalid_format(self):
    time_string = tf.constant(['07:31:15', '14:30:00'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Cannot find matched format element at 0',
    ):
      self.evaluate(time_ops.cast_to_time_from_string(time_string, 'abc'))


if __name__ == '__main__':
  tf.test.main()
