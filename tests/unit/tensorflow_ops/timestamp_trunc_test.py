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

"""Tests for BQML TIMESTAMP_SUB custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class TimestampTruncTest(tf.test.TestCase):

  def test_timestamp_trunc(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'MICROSECOND'),
        tf.constant(
            ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'MILLISECOND'),
        tf.constant(
            ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'SECOND'),
        tf.constant(
            ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'MINUTE'),
        tf.constant(
            ['2008-12-25 15:30:00.0 +0000', '2023-11-11 14:30:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'HOUR'),
        tf.constant(
            ['2008-12-25 15:00:00.0 +0000', '2023-11-11 14:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'DAY', 'America/Los_Angeles'),
        tf.constant(
            ['2008-12-25 08:00:00.0 +0000', '2023-11-11 08:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'WEEK'),
        tf.constant(
            ['2008-12-21 00:00:00.0 +0000', '2023-11-05 00:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'WEEK_TUESDAY'),
        tf.constant(
            ['2008-12-23 00:00:00.0 +0000', '2023-11-07 00:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'ISOWEEK'),
        tf.constant(
            ['2008-12-22 00:00:00.0 +0000', '2023-11-06 00:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'MONTH'),
        tf.constant(
            ['2008-12-01 00:00:00.0 +0000', '2023-11-01 00:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'QUARTER'),
        tf.constant(
            ['2008-10-01 00:00:00.0 +0000', '2023-10-01 00:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'YEAR'),
        tf.constant(
            ['2008-01-01 00:00:00.0 +0000', '2023-01-01 00:00:00.0 +0000']
        ),
    )
    self.assertAllEqual(
        timestamp_ops.timestamp_trunc(timestamp, 'ISOYEAR'),
        tf.constant(
            ['2007-12-31 00:00:00.0 +0000', '2023-01-02 00:00:00.0 +0000']
        ),
    )

  def test_timestamp_trunc_invalid_part(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in TimestampTrunc: RandomPart',
    ):
      self.evaluate(timestamp_ops.timestamp_trunc(timestamp, 'RandomPart'))

  def test_timestamp_trunc_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00 abc', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timestamp in TimestampTrunc: 2008-12-25 15:30:00 abc',
    ):
      self.evaluate(timestamp_ops.timestamp_trunc(timestamp, 'YEAR'))

  def test_timestamp_trunc_invalid_time_zone(self):
    timestamp = tf.constant(
        ['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid time zone in TimestampTrunc: UtC',
    ):
      self.evaluate(timestamp_ops.timestamp_trunc(timestamp, 'YEAR', 'UtC'))


if __name__ == '__main__':
  tf.test.main()
