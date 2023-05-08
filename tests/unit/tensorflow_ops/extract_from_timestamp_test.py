# Copyright 2022 Google LLC
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

"""Tests for BQML EXTRACT from timestamp custom ops."""

from bigquery_ml_utils.tensorflow_ops import timestamp_ops
import tensorflow as tf


class ExtractFromTimestampTest(tf.test.TestCase):

  def test_extract_from_timestamp(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
    )

    expect = tf.constant([700000, 300000])
    result = timestamp_ops.extract_from_timestamp(
        'MICROSECOND', timestamp, 'UTC'
    )
    self.assertAllEqual(result, expect)

    expect = tf.constant([700, 300])
    result = timestamp_ops.extract_from_timestamp(
        'MILLISECOND', timestamp, 'UTC'
    )
    self.assertAllEqual(result, expect)

    expect = tf.constant([56, 12])
    result = timestamp_ops.extract_from_timestamp('SECOND', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([0, 11])
    result = timestamp_ops.extract_from_timestamp('MINUTE', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([0, 11])
    result = timestamp_ops.extract_from_timestamp('HOUR', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([3, 3])
    result = timestamp_ops.extract_from_timestamp('DAYOFWEEK', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([10, 14])
    result = timestamp_ops.extract_from_timestamp('DAY', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([10, 73])
    result = timestamp_ops.extract_from_timestamp('DAYOFYEAR', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([2, 11])
    result = timestamp_ops.extract_from_timestamp('WeeK', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([1, 11])
    result = timestamp_ops.extract_from_timestamp(
        'WeeK_TuEsDaY', timestamp, 'America/Los_Angeles'
    )
    self.assertAllEqual(result, expect)

    expect = tf.constant([2, 11])
    result = timestamp_ops.extract_from_timestamp('ISOWEEK', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([1, 3])
    result = timestamp_ops.extract_from_timestamp('MONTH', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([1, 1])
    result = timestamp_ops.extract_from_timestamp('QUARTER', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([2023, 2023])
    result = timestamp_ops.extract_from_timestamp('YEAR', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

    expect = tf.constant([2023, 2023])
    result = timestamp_ops.extract_from_timestamp('ISOYEAR', timestamp, 'UTC')
    self.assertAllEqual(result, expect)

  def test_extract_from_timestamp_invalid_timezone(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timezone in ExtractFromTimestamp: UtC',
    ):
      self.evaluate(
          timestamp_ops.extract_from_timestamp('MICROSECOND', timestamp, 'UtC')
      )

  def test_extract_from_timestamp_invalid_timestamp(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7', '2023-03-14 23:45:12.3 +1234']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid timestamp in ExtractFromTimestamp: 2023-01-10 12:34:56.7',
    ):
      self.evaluate(
          timestamp_ops.extract_from_timestamp('MICROSECOND', timestamp, 'UTC')
      )

  def test_extract_from_timestamp_invalid_part(self):
    timestamp = tf.constant(
        ['2023-01-10 12:34:56.7 +1234', '2023-03-14 23:45:12.3 +1234']
    )
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        'Invalid part in ExtractFromTimestamp: micro',
    ):
      self.evaluate(
          timestamp_ops.extract_from_timestamp('MICRO', timestamp, 'UTC')
      )


if __name__ == '__main__':
  tf.test.main()
