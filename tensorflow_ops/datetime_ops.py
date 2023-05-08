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

"""Python wrapper for BigQuery datetime custom ops."""

import tensorflow as tf

gen_datetime_ops = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile('datetime_ops.so')
)


def extract_from_datetime(datetime, part, name=None):
  """Returns the specified part from a supplied datetime.

  Equivalent SQL: EXTRACT(part FROM datetime)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAYOFWEEK, DAY, DAYOFYEAR, WEEK,
      WEEK(WEEKDAY), ISOWEEK, MONTH, QUARTER, YEAR, ISOYEAR. It is case
      insensitive.
    name: An optional name for the op.
  """
  return gen_datetime_ops.extract_from_datetime(
      datetime=datetime, part=part, name=name
  )


def extract_date_from_datetime(datetime, name=None):
  """Returns the DATE part from a supplied datetime.

  Equivalent SQL: EXTRACT(DATE FROM datetime)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    name: An optional name for the op.
  """
  return gen_datetime_ops.extract_date_from_datetime(
      datetime=datetime, name=name
  )


def extract_time_from_datetime(datetime, name=None):
  """Returns the TIME part from a supplied datetime.

  Equivalent SQL: EXTRACT(TIME FROM datetime)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    name: An optional name for the op.
  """
  return gen_datetime_ops.extract_time_from_datetime(
      datetime=datetime, name=name
  )
