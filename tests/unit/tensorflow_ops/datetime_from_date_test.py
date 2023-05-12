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

"""Tests for BigQuery datetime_from_date custom op."""

from bigquery_ml_utils.tensorflow_ops import datetime_ops
import tensorflow as tf


@tf.test.with_eager_op_as_function
class DatetimeFromDateTest(tf.test.TestCase):

  def test_datetime_from_date(self):
    self.assertAllEqual(
        datetime_ops.datetime_from_date(
            tf.constant(['2012-01-01', '2023-04-14'])
        ),
        tf.constant(
            ['2012-01-01 00:00:00.000000', '2023-04-14 00:00:00.000000']
        ),
    )

  def test_datetime_from_date_invalid(self):
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        "Invalid date input '01-01-2021' in DatetimeFromDate.",
    ):
      self.evaluate(
          datetime_ops.datetime_from_date(
              tf.constant(['2023-04-14', '01-01-2021'])
          )
      )


if __name__ == '__main__':
  tf.test.main()
