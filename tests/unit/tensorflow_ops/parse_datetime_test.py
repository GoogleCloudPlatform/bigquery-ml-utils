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

"""Tests for BigQuery parse_datetime custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class ParseDatetimeTest(tf.test.TestCase):

  def test_parse_datetime_succeed(self):
    self.assertAllEqual(
        datetime_ops.parse_datetime(
            '%Y-%m-%d %H:%M:%S',
            tf.constant(['1998-10-18 13:45:55', '2023-01-01 13:45:55']),
        ),
        tf.constant(['1998-10-18 13:45:55', '2023-01-01 13:45:55']),
    )

    self.assertAllEqual(
        datetime_ops.parse_datetime(
            '%m/%d/%Y %I:%M:%S %p',
            tf.constant(['8/30/2018 2:23:38 pm', '03/01/2021 10:23:22 pm']),
        ),
        tf.constant(['2018-08-30 14:23:38', '2021-03-01 22:23:22']),
    )

    self.assertAllEqual(
        datetime_ops.parse_datetime(
            '%A, %B %e, %Y',
            tf.constant(
                ['Wednesday, December 19, 2018', 'Thursday, April 20, 2023']
            ),
        ),
        tf.constant(['2018-12-19 00:00:00', '2023-04-20 00:00:00']),
    )

  def test_parse_datetime_invalid_format_string(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        "Mismatch between format character 'a' and string character '1'",
    ):
      self.evaluate(
          datetime_ops.parse_datetime(
              'abcd',
              tf.constant(['1998-10-18 13:45:55', '2023-01-01 13:45:55']),
          )
      )

  def test_parse_datetime_invalid_datetime_string(self):
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Out-of-range datetime field in parsing function',
    ):
      self.evaluate(
          datetime_ops.parse_datetime(
              '%m/%d/%Y %I:%M:%S %p',
              tf.constant(['02/29/2018 2:23:38 pm', '03/01/2021 10:23:22 pm']),
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Failed to parse input string "2023-03-14 23:45:12.3"',
    ):
      self.evaluate(
          datetime_ops.parse_datetime(
              '%m/%d/%Y %I:%M:%S %p',
              tf.constant(['02/28/2018 2:23:38 pm', '2023-03-14 23:45:12.3']),
          )
      )


if __name__ == '__main__':
  tf.test.main()
