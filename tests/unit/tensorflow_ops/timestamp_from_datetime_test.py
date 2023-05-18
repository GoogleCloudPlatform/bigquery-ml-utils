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

"""Tests for BQML TIMESTAMP from datetime custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampFromDatetimeTest(tf.test.TestCase):

  def test_timestamp_from_datetime(self):
    datetime = tf.constant(
        ['2023-02-02 02:02:01.152903', '2008-12-25 15:30:00.152903']
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_from_datetime(datetime),
        tf.constant(
            ['2023-02-02 02:02:01.1 +0000', '2008-12-25 15:30:00.1 +0000']
        ),
    )

  def test_timestamp_from_datetime_tz(self):
    datetime = tf.constant(
        ['2023-02-02 02:02:01.152903', '2008-12-25 15:30:00.152903']
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_from_datetime(datetime, 'America/Los_Angeles'),
        tf.constant(
            ['2023-02-02 10:02:01.1 +0000', '2008-12-25 23:30:00.1 +0000']
        ),
    )

  def test_timestamp_from_datetime_invalid_timezone(self):
    datetime = tf.constant(
        ['2023-02-02 02:02:01.152903', '2008-12-25 15:30:00.152903']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid time zone in TimestampFromDatetime: UtC',
    ):
      self.evaluate(timestamp_ops.timestamp_from_datetime(datetime, 'UtC'))

  def test_timestamp_from_datetime_invalid_date(self):
    datetime = tf.constant(
        ['2023-02-02T02:02:01.152903', '2008-12-25 15:30:00.152903']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid datetime in TimestampFromDatetime: 2023-02-02T02:02:01.152903',
    ):
      self.evaluate(timestamp_ops.timestamp_from_datetime(datetime))


if __name__ == '__main__':
  tf.test.main()
