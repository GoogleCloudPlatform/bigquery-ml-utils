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

"""Tests for BigQuery CastToDateFromString custom op."""


from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class CastToDateFromStringTest(tf.test.TestCase):

  def test_cast_to_date_from_string_without_format(self):
    date_string = tf.constant(['2018-12-03', '2020-01-11'])
    self.assertAllEqual(
        date_ops.cast_to_date_from_string(date_string),
        tf.constant(['2018-12-03', '2020-01-11']),
    )

  def test_cast_to_date_from_string_with_format(self):
    self.assertAllEqual(
        date_ops.cast_to_date_from_string(
            tf.constant(['18-12-03', '00-06-11']),
            'YY-MM-DD',
        ),
        tf.constant(['2018-12-03', '2000-06-11']),
    )

  def test_cast_to_date_from_string_invalid_string(self):
    date_string = tf.constant(['2018-12-03a', '2020-01-11'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        "Invalid date: '2018-12-03a'",
    ):
      self.evaluate(date_ops.cast_to_date_from_string(date_string))

  def test_cast_to_date_from_string_invalid_format(self):
    date_string = tf.constant(['2018-12-03', '2020-01-11'])
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Cannot find matched format element at 0',
    ):
      self.evaluate(date_ops.cast_to_date_from_string(date_string, 'abc'))


if __name__ == '__main__':
  tf.test.main()
