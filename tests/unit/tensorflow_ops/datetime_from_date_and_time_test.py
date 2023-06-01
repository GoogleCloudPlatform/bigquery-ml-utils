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

"""Tests for BigQuery datetime_from_date_and_time custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeFromDateAndTimeTest(tf.test.TestCase):

  def test_datetime_from_date_and_time(self):
    self.assertAllEqual(
        datetime_ops.datetime_from_date_and_time(
            tf.constant(['2012-01-01', '2023-04-14']),
            tf.constant(['01:00:00', '21:10:01']),
        ),
        tf.constant(['2012-01-01 01:00:00', '2023-04-14 21:10:01']),
    )

  def test_datetime_from_date_and_time_invalid(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "01-01-2021"',
    ):
      self.evaluate(
          datetime_ops.datetime_from_date_and_time(
              tf.constant(['2023-04-14', '01-01-2021']),
              tf.constant(['01:00:00', '21:10:01']),
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "25:10:00"',
    ):
      self.evaluate(
          datetime_ops.datetime_from_date_and_time(
              tf.constant(['2023-04-14', '2023-04-14']),
              tf.constant(['25:10:00', '21:10:01']),
          )
      )


if __name__ == '__main__':
  tf.test.main()
