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

"""Tests for BigQuery EXTRACT time from datetime custom ops."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class ExtractTimeFromDatetimeTest(tf.test.TestCase):

  def test_extract_time_from_datetime(self):
    self.assertAllEqual(
        datetime_ops.extract_time_from_datetime(
            tf.constant(['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3'])
        ),
        tf.constant(['12:34:56.700', '23:45:12.300']),
    )

  def test_extract_time_from_datetime_invalid_datetime(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid datetime in ExtractTimeFromDatetime: 2023-01-10',
    ):
      self.evaluate(
          datetime_ops.extract_time_from_datetime(
              tf.constant(['2023-01-10', '2023-03-14'])
          )
      )


if __name__ == '__main__':
  tf.test.main()
