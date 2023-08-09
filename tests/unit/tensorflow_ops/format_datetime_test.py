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

"""Tests for BQML FORMAT_DATETIME custom ops."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


class FormatDatetimeTest(tf.test.TestCase):

  def test_format_datetime(self):
    datetime = tf.constant(['2008-12-25 15:30:00', '2023-11-11 14:30:00'])
    self.assertAllEqual(
        datetime_ops.format_datetime('%c', datetime),
        tf.constant(['Thu Dec 25 15:30:00 2008', 'Sat Nov 11 14:30:00 2023']),
    )
    self.assertAllEqual(
        datetime_ops.format_datetime('%b-%d-%Y', datetime),
        tf.constant(['Dec-25-2008', 'Nov-11-2023']),
    )
    self.assertAllEqual(
        datetime_ops.format_datetime('%b %Y', datetime),
        tf.constant(['Dec 2008', 'Nov 2023']),
    )

  def test_format_datetime_invalid_datetime(self):
    datetime = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2008-12-25 15:30:00 abc"',
    ):
      self.evaluate(datetime_ops.format_datetime('%c', datetime))


if __name__ == '__main__':
  tf.test.main()
