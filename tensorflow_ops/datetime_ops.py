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

from bigquery_ml_utils.tensorflow_ops.load_module import load_module

gen_datetime_ops = load_module("_datetime_ops.so")


def datetime_from_components(year, month, day, hour, minute, second, name=None):
  """Returns datetime using INT64 values representing the components of it.

  Equivalent SQL: DATETIME(year, month, day, hour, minute, second)

  Args:
    year: tf.Tensor of type int64 representing the year.
    month: tf.Tensor of type int64 representing the month.
    day: tf.Tensor of type int64 representing the day.
    hour: tf.Tensor of type int64 representing the hour.
    minute: tf.Tensor of type int64 representing the minute.
    second: tf.Tensor of type int64 representing the second.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_from_components(
      year=year,
      month=month,
      day=day,
      hour=hour,
      minute=minute,
      second=second,
      name=name,
  )


def datetime_from_date(date, name=None):
  """Returns datetime using DATE value.

  Equivalent SQL: DATETIME(date_expression)

  Args:
    date: tf.Tensor of type string in "%F" format.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_from_date(date=date, name=name)


def datetime_from_date_and_time(date, time, name=None):
  """Returns datetime using DATE value and TIME value.

  Equivalent SQL: DATETIME(date_expression, time_expression)

  Args:
    date: tf.Tensor of type string in "%F" format.
    time: tf.Tensor of type string in "%H:%M:%E6S" format.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_from_date_and_time(
      date=date, time=time, name=name
  )


def datetime_from_timestamp(timestamp, time_zone="UTC", name=None):
  """Returns datetime using TIMESTAMP value and optional time zone.

  Equivalent SQL: DATETIME(timestamp_expression [, time_zone])

  Args:
    timestamp: tf.Tensor of type string in "%F %H:%M:%E1S %z" format.
    time_zone: Optional. A string represents the time zone.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_from_timestamp(
      timestamp=timestamp, time_zone=time_zone, name=name
  )


def datetime_add(datetime, interval, part, name=None):
  """Returns the added DATETIME with the interval of part.

  Equivalent SQL:
    DATETIME_ADD(datetime_expression, INTERVAL int64_expression part)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    interval: tf.Tensor of type int64. It has the same shape of datetime input.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR. It is
      case insensitive.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_add(
      datetime=datetime, interval=interval, part=part, name=name
  )


def datetime_diff(datetime_a, datetime_b, part, name=None):
  """Returns the number of specified part intervals between two DATETIME.

  Equivalent SQL:
    DATETIME_DIFF(datetime_a, datetime_b, part)

  Args:
    datetime_a: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    datetime_b: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, WEEK(<WEEKDAY>), ISOWEEK,
      MONTH, QUARTER, YEAR, ISOYEAR. It is case insensitive.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_diff(
      datetime_a=datetime_a, datetime_b=datetime_b, part=part, name=name
  )


def datetime_sub(datetime, interval, part, name=None):
  """Returns the subtracted DATETIME with the interval of part.

  Equivalent SQL:
    DATETIME_SUB(datetime_expression, INTERVAL int64_expression part)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    interval: tf.Tensor of type int64. It has the same shape of datetime input.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR. It is
      case insensitive.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_sub(
      datetime=datetime, interval=interval, part=part, name=name
  )


def datetime_trunc(datetime, part, name=None):
  """Returns the truncated DATETIME value to the granularity of date_time_part.

  Equivalent SQL:
    DATETIME_TRUNC(datetime_expression, part)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR. It is
      case insensitive.
    name: An optional name for the op.
  """
  return gen_datetime_ops.datetime_trunc(
      datetime=datetime, part=part, name=name
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


def last_day_from_datetime(datetime, part="MONTH", name=None):
  """Returns the last day from a datetime that contains the date.

  Equivalent SQL: LAST_DAY(datetime[, part])

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    part: A string represents the datetime part. Can be WEEK, WEEK(WEEKDAY),
      ISOWEEK, MONTH, QUARTER, YEAR, ISOYEAR. It is case insensitive.
    name: An optional name for the op.
  """
  return gen_datetime_ops.last_day_from_datetime(
      datetime=datetime, part=part, name=name
  )


def parse_datetime(format_string, datetime_string, name=None):
  """Returns the parsed DATETIME value based on the format_string.

  Equivalent SQL:

    PARSE_DATETIME(format_string, datetime_string)
  Args:
    format_string: A string represents the format of the datetime value.
    datetime_string: tf.Tensor of type string.
    name: An optional name for the op.
  """
  return gen_datetime_ops.parse_datetime(
      format_string=format_string, datetime_string=datetime_string, name=name
  )


def safe_parse_datetime(format_string, datetime_string, name=None):
  """Returns the safely parsed DATETIME value based on the format_string.

  Equivalent SQL: SAFE.PARSE_DATETIME(format_string, datetime_string). Returns
  "1970-01-01 00:00:00.000000" for unsuccessful parsing.

  Args:
    format_string: A string represents the format of the datetime value.
    datetime_string: tf.Tensor of type string.
    name: An optional name for the op.
  """
  return gen_datetime_ops.safe_parse_datetime(
      format_string=format_string, datetime_string=datetime_string, name=name
  )
