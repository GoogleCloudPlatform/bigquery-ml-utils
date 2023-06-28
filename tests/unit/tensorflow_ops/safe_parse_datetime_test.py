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

"""Tests for BigQuery SafeParseDatetime custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class SafeParseDatetimeTest(tf.test.TestCase):

  def test_safe_parse_datetime_succeed(self):
    self.assertAllEqual(
        datetime_ops.safe_parse_datetime(
            '%Y-%m-%d %H:%M:%S',
            tf.constant(['1998-10-18 13:45:55', '2023-01-01 13:45:55']),
        ),
        tf.constant(['1998-10-18 13:45:55', '2023-01-01 13:45:55']),
    )

    self.assertAllEqual(
        datetime_ops.safe_parse_datetime(
            '%m/%d/%Y %I:%M:%S %p',
            tf.constant(['8/30/2018 2:23:38 pm', '03/01/2021 10:23:22 pm']),
        ),
        tf.constant(['2018-08-30 14:23:38', '2021-03-01 22:23:22']),
    )

    self.assertAllEqual(
        datetime_ops.safe_parse_datetime(
            '%A, %B %e, %Y',
            tf.constant(
                ['Wednesday, December 19, 2018', 'Thursday, April 20, 2023']
            ),
        ),
        tf.constant(['2018-12-19 00:00:00', '2023-04-20 00:00:00']),
    )

  def test_safe_parse_datetime_invalid_format_string(self):
    self.assertAllEqual(
        datetime_ops.safe_parse_datetime(
            'invalid_format',
            tf.constant(['1998-10-18 13:45:55', '2023-01-01 13:45:55']),
        ),
        tf.constant(['1970-01-01 00:00:00', '1970-01-01 00:00:00']),
    )

  def test_safe_parse_datetime_invalid_datetime_string(self):
    self.assertAllEqual(
        datetime_ops.safe_parse_datetime(
            '%Y-%m-%d %H:%M:%S',
            tf.constant([
                '1998-10-18 13:45:55',
                '2023-01-01 13:45:55',
                'invalid_datetime',
            ]),
        ),
        tf.constant([
            '1998-10-18 13:45:55',
            '2023-01-01 13:45:55',
            '1970-01-01 00:00:00',
        ]),
    )


if __name__ == '__main__':
  tf.test.main()
