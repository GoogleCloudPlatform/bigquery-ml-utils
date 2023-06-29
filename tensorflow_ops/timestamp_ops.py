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


def timestamp_from_string(timestamp_string, time_zone=None, name=None):
  """Returns a timestamp from a string at a given timezone.

  Equivalent SQL: TIMESTAMP(string_expression[, time_zone])

  Args:
    timestamp_string: tf.Tensor of type string. Must include a timestamp
      literal. If timestamp includes a time_zone in the string, do not include
      an explicit time_zone argument.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_from_string(
      timestamp_string=timestamp_string,
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


def timestamp_add(timestamp, interval, part, name=None):
  """Returns a timestamp by adding interval to the timestamp.

  Equivalent SQL: TIMESTAMP_ADD(timestamp_expression, INTERVAL int64_expression
  date_part)

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    interval: tf.Tensor of type int64. Integer represents the unit of part.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY. Case insensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_add(
      timestamp=timestamp,
      interval=interval,
      part=part,
      name=name,
  )


def timestamp_sub(timestamp, interval, part, name=None):
  """Returns a timestamp by subtracting interval to the timestamp.

  Equivalent SQL: TIMESTAMP_SUB(timestamp_expression, INTERVAL int64_expression
  date_part)

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    interval: tf.Tensor of type int64. Integer represents the unit of part.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY. Case insensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_sub(
      timestamp=timestamp,
      interval=interval,
      part=part,
      name=name,
  )


def timestamp_diff(timestamp_a, timestamp_b, part, name=None):
  """Returns the whole number of specified date_part intervals between timestamp_a and timestamp_b.

  Equivalent SQL: TIMESTAMP_DIFF(timestamp_expression_a, timestamp_expression_b,
  date_part)

  Args:
    timestamp_a: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z"
      format.
    timestamp_b: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z"
      format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY. Case insensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_diff(
      timestamp_a=timestamp_a,
      timestamp_b=timestamp_b,
      part=part,
      name=name,
  )


def timestamp_trunc(timestamp, part, time_zone="UTC", name=None):
  """Returns a timestamp which by truncating the original timestamp to the granularity of part.

  Equivalent SQL: TIMESTAMP_TRUNC(timestamp_expression, date_time_part[,
  time_zone])

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND,MINUTE, HOUR,  DAY, WEEK,  WEEK_MONDAY, WEEK_TUESDAY,
      WEEK_WEDNESDAY, WEEK_THURSDAY, WEEK_FRIDAY, WEEK_SATURDAY, ISOWEEK, MONTH,
      QUARTER, YEAR,  ISOYEAR. Case insensitive.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_trunc(
      timestamp=timestamp,
      part=part,
      time_zone=time_zone,
      name=name,
  )


def format_timestamp(format_string, timestamp, time_zone="UTC", name=None):
  """Returns a timestamp string based on format_string.

  Equivalent SQL: FORMAT_TIMESTAMP(format_string, timestamp[, time_zone])

  Args:
    format_string: tf.Tensor of type string. Format of the output string.
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.format_timestamp(
      format_string=format_string,
      timestamp=timestamp,
      time_zone=time_zone,
      name=name,
  )


def parse_timestamp(
    format_string, timestamp_string, time_zone="UTC", name=None
):
  """Returns a timestamp by parsing a string.

  Equivalent SQL: PARSE_TIMESTAMP(format_string, timestamp_string[, time_zone])

  Args:
    format_string: tf.Tensor of type string. Format of the string timestamp.
    timestamp_string: tf.Tensor of type string. Timestamp in any supported
      format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.parse_timestamp(
      format_string=format_string,
      timestamp_string=timestamp_string,
      time_zone=time_zone,
      name=name,
  )


def safe_parse_timestamp(
    format_string, timestamp_string, time_zone="UTC", name=None
):
  """Returns a timestamp by safely parsing a string.

  Equivalent SQL:
  SAFE.PARSE_TIMESTAMP(format_string, timestamp_string[, time_zone]).
  Returns '1970-01-01 00:00:00.0 +0000' for unsuccessful parsing.

  Args:
    format_string: tf.Tensor of type string. Format of the string timestamp.
    timestamp_string: tf.Tensor of type string. Timestamp in any supported
      format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.safe_parse_timestamp(
      format_string=format_string,
      timestamp_string=timestamp_string,
      time_zone=time_zone,
      name=name,
  )


def timestamp_micros(timestamp_micro, name=None):
  """Returns a timestamp by interpreting timestamp_micro as the number of microseconds since 1970-01-01 00:00:00 UTC.

  Equivalent SQL: TIMESTAMP_MICROS(int64_expression)

  Args:
    timestamp_micro: tf.Tensor of type int64.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_micros(
      timestamp_micro=timestamp_micro,
      name=name,
  )


def timestamp_millis(timestamp_milli, name=None):
  """Returns a timestamp by interpreting timestamp_milli as the number of milliseconds since 1970-01-01 00:00:00 UTC.

  Equivalent SQL: TIMESTAMP_MILLIS(int64_expression)

  Args:
    timestamp_milli: tf.Tensor of type int64.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_millis(
      timestamp_milli=timestamp_milli,
      name=name,
  )


def timestamp_seconds(timestamp_sec, name=None):
  """Returns a timestamp by interpreting timestamp_sec as the number of seconds since 1970-01-01 00:00:00 UTC.

  Equivalent SQL: TIMESTAMP_SECONDS(int64_expression)

  Args:
    timestamp_sec: tf.Tensor of type int64.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.timestamp_seconds(
      timestamp_sec=timestamp_sec,
      name=name,
  )


def unix_micros(timestamp, name=None):
  """Returns the number of microseconds since 1970-01-01 00:00:00 UTC.

  Equivalent SQL: UNIX_MICROS(timestamp_expression)

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.unix_micros(
      timestamp=timestamp,
      name=name,
  )


def unix_millis(timestamp, name=None):
  """Returns number of milliseconds since 1970-01-01 00:00:00 UTC.

  Equivalent SQL: UNIX_MILLIS(timestamp_expression)

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.unix_millis(
      timestamp=timestamp,
      name=name,
  )


def unix_seconds(timestamp, name=None):
  """Returns number of seconds since 1970-01-01 00:00:00 UTC.

  Equivalent SQL: UNIX_SECONDS(timestamp_expression)

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    name: An optional name for the op.
  """
  return gen_timestamp_ops.unix_seconds(
      timestamp=timestamp,
      name=name,
  )
