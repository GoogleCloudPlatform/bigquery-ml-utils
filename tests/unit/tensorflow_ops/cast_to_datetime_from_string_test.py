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

"""Tests for BigQuery CastToDatetimeFromString custom op."""


from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


class CastToDatetimeFromStringTest(tf.test.TestCase):

  def test_cast_to_datetime_from_string_without_format(self):
    datetime_string = tf.constant(
        ['2018-12-03 07:31:15', '2020-01-11 14:30:00']
    )
    self.assertAllEqual(
        datetime_ops.cast_to_datetime_from_string(datetime_string),
        tf.constant(['2018-12-03 07:31:15', '2020-01-11 14:30:00']),
    )

  def test_cast_to_datetime_from_string_with_format(self):
    datetime_string = tf.constant(
        ['2020.06.03 00:00:53', '2000.10.11 18:50:23']
    )
    self.assertAllEqual(
        datetime_ops.cast_to_datetime_from_string(
            datetime_string, 'YYYY.MM.DD HH24:MI:SS'
        ),
        tf.constant(['2020-06-03 00:00:53', '2000-10-11 18:50:23']),
    )

  def test_cast_to_datetime_from_string_invalid_string(self):
    datetime_string = tf.constant(
        ['2018-12-03 07:31:15a', '2020-01-11 14:30:00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Invalid datetime string "2018-12-03 07:31:15a"',
    ):
      self.evaluate(datetime_ops.cast_to_datetime_from_string(datetime_string))

  def test_cast_to_datetime_from_string_invalid_format(self):
    datetime_string = tf.constant(
        ['2018-12-03 07:31:15', '2020-01-11 14:30:00']
    )
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Cannot find matched format element at 0',
    ):
      self.evaluate(
          datetime_ops.cast_to_datetime_from_string(datetime_string, 'abc')
      )


if __name__ == '__main__':
  tf.test.main()
