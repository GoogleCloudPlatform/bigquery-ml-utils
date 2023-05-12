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

"""Tests for BigQuery datetime_from_timestamp custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeFromTimestampTest(tf.test.TestCase):

  def test_datetime_from_timestamp(self):
    self.assertAllEqual(
        datetime_ops.datetime_from_timestamp(
            tf.constant(
                ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
            )
        ),
        tf.constant(
            ['2023-01-10 00:00:56.700000', '2023-03-14 11:11:12.300000']
        ),
    )

  def test_datetime_from_timestamp_invalid(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        "Invalid date input 'aaaa' in DatetimeFromTimestamp.",
    ):
      self.evaluate(
          datetime_ops.datetime_from_timestamp(
              tf.constant(['2023-01-10 12:34:56.7 +1234', 'aaaa'])
          )
      )

    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timezone in DatetimeFromTimestamp: aaaa',
    ):
      self.evaluate(
          datetime_ops.datetime_from_timestamp(
              tf.constant(['2023-01-10 12:34:56.7 +1234']), 'aaaa'
          )
      )


if __name__ == '__main__':
  tf.test.main()
