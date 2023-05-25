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

"""Tests for BigQuery DateFromComponents custom op."""

from bigquery_ml_utils.tensorflow_ops import date_ops
import tensorflow as tf


class DateFromComponentsTest(tf.test.TestCase):

  def test_date_from_components(self):
    year = tf.constant([2008, 2023], dtype=tf.int64)
    month = tf.constant([5, 12], dtype=tf.int64)
    day = tf.constant([1, 28], dtype=tf.int64)
    self.assertAllEqual(
        date_ops.date_from_components(year, month, day),
        tf.constant(['2008-05-01', '2023-12-28']),
    )

  def test_date_from_components_invalid_input(self):
    year = tf.constant([2008, 2023], dtype=tf.int64)
    month = tf.constant([15, 12], dtype=tf.int64)
    day = tf.constant([1, 28], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.OutOfRangeError, ValueError),
        'Input calculates to invalid date: 2008-15-01',
    ):
      self.evaluate(date_ops.date_from_components(year, month, day))

  def test_date_from_components_differenet_shapes(self):
    year = tf.constant([2008, 2023], dtype=tf.int64)
    month = tf.constant([5, 12], dtype=tf.int64)
    day = tf.constant([1, 28, 1], dtype=tf.int64)
    with self.assertRaisesRegex(
        (tf.errors.InvalidArgumentError, ValueError),
        (
            'Errors in DateFromComponents: Inputs must have the same shape, but'
            ' are: 2, 2, 3'
        ),
    ):
      self.evaluate(date_ops.date_from_components(year, month, day))


if __name__ == '__main__':
  tf.test.main()
