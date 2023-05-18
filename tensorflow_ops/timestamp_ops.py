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

"""Python wrapper for BQML timestamp custom ops."""

from bigquery_ml_utils.tensorflow_ops.load_module import load_module

gen_timestamp_ops = load_module("_timestamp_ops.so")


def extract_from_timestamp(part, timestamp, time_zone="UTC", name=None):
  """Returns the specified part from a supplied timestamp at a given timezone.

  Equivalent SQL: EXTRACT(part FROM timestamp AT TIME ZONE time_zone)

  Args:
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAYOFWEEK, DAY, DAYOFYEAR, WEEK,
      WEEK(<WEEKDAY>), ISOWEEK, MONTH, QUARTER, YEAR, ISOYEAR. Case insensitive.
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.extract_from_timestamp(
      part=part, timestamp=timestamp, time_zone=time_zone, name=name
  )


def string_from_timestamp(timestamp, time_zone="UTC", name=None):
  """Returns a string from a timestamp at a given timezone.

  Equivalent SQL: STRING(timestamp_expression[, time_zone])

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.string_from_timestamp(
      timestamp=timestamp, time_zone=time_zone, name=name
  )


def timestamp_from_string(timestamp, time_zone=None, name=None):
  """Returns a timestamp from a string at a given timezone.

  Equivalent SQL: TIMESTAMP(string_expression[, time_zone])

  Args:
    timestamp: tf.Tensor of type string. Must include a timestamp literal. If
      timestamp includes a time_zone in the string, do not include an explicit
      time_zone argument.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_from_string(
      timestamp=timestamp,
      time_zone="UTC" if time_zone is None else time_zone,
      allow_tz_in_str=time_zone is None,
      name=name,
  )


def timestamp_from_date(date, time_zone="UTC", name=None):
  """Returns a timestamp from a date at a given timezone.

  Equivalent SQL: TIMESTAMP(date_expression[, time_zone])

  Args:
    date: tf.Tensor of type string. Date in "%F" format. Returned is the
      earliest timestamp that falls within the given date.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_from_date(
      date=date,
      time_zone=time_zone,
      name=name,
  )


def timestamp_from_datetime(datetime, time_zone="UTC", name=None):
  """Returns a timestamp from a datetime at a given timezone.

  Equivalent SQL: TIMESTAMP(datetime_expression[, time_zone])

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_from_datetime(
      datetime=datetime,
      time_zone=time_zone,
      name=name,
  )
