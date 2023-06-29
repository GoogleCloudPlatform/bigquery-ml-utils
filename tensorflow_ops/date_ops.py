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

"""Python wrapper for BigQuery date custom ops."""

from bigquery_ml_utils.tensorflow_ops.load_module import load_module

gen_date_ops = load_module("_date_ops.so")


def extract_from_date(date, part, name=None):
  """Returns the specified part from a supplied date.

  Equivalent SQL: EXTRACT(part FROM date)

  Args:
    date: tf.Tensor of type string. Date in "%F" format.
    part: A string represents the date part. Can be DAYOFWEEK, DAY, DAYOFYEAR,
      WEEK, WEEK(WEEKDAY), ISOWEEK, MONTH, QUARTER, YEAR, ISOYEAR. It is case
      insensitive.
    name: An optional name for the op.
  """
  return gen_date_ops.extract_from_date(date=date, part=part, name=name)


def date_from_components(year, month, day, name=None):
  """Returns a date using INT64 values representing the year, month and day.

  Equivalent SQL: DATE(year, month, day)

  Args:
    year: tf.Tensor of type int64 representing the year.
    month: tf.Tensor of type int64 representing the month.
    day: tf.Tensor of type int64 representing the day.
    name: An optional name for the op.
  """
  return gen_date_ops.date_from_components(
      year=year, month=month, day=day, name=name
  )


def date_from_timestamp(timestamp, time_zone="UTC", name=None):
  """Returns a date from a timestamp.

  Equivalent SQL: DATE(timestamp_expression, [time_zone_expression])

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_date_ops.date_from_timestamp(
      timestamp=timestamp, time_zone=time_zone, name=name
  )


def date_from_datetime(datetime, name=None):
  """Returns a date from a datetime.

  Equivalent SQL: DATE(datetime_expression)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    name: An optional name for the op.
  """
  return gen_date_ops.date_from_datetime(datetime=datetime, name=name)


def date_add(date, interval, part, name=None):
  """Returns a date by adding interval to the date.

  Equivalent SQL: DATE_ADD(date_expression, INTERVAL int64_expression date_part)

  Args:
    date: tf.Tensor of type string. Date in "%F" format.
    interval: tf.Tensor of type int64. Integer represents the unit of part.
    part: A string represents the date part. Can be DAY, WEEK, MONTH, QUARTER,
      YEAR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_date_ops.date_add(
      date=date,
      interval=interval,
      part=part,
      name=name,
  )


def date_sub(date, interval, part, name=None):
  """Returns a date by subtracting interval to the timestamp.

  Equivalent SQL: DATE_SUB(date_expression, INTERVAL int64_expression date_part)

  Args:
    date: tf.Tensor of type string. Date in "%F" format.
    interval: tf.Tensor of type int64. Integer represents the unit of part.
    part: A string represents the date part. Can be DAY, WEEK, MONTH, QUARTER,
      YEAR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_date_ops.date_sub(
      date=date,
      interval=interval,
      part=part,
      name=name,
  )


def date_diff(date_a, date_b, part, name=None):
  """Returns the whole number of specified part intervals between date_a and date_b.

  Equivalent SQL: DATE_DIFF(date_expression_a, date_expression_b, date_part)

  Args:
    date_a: tf.Tensor of type string. Date in "%F" format.
    date_b: tf.Tensor of type string. Date in "%F" format.
    part: A string represents the date part. Can be DAY, WEEK,  WEEK_MONDAY,
      WEEK_TUESDAY, WEEK_WEDNESDAY, WEEK_THURSDAY, WEEK_FRIDAY, WEEK_SATURDAY,
      ISOWEEK, MONTH, QUARTER, YEAR,  ISOYEAR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_date_ops.date_diff(
      date_a=date_a,
      date_b=date_b,
      part=part,
      name=name,
  )


def date_trunc(date, part, name=None):
  """Returns the whole number of specified part intervals between date_a and date_b.

  Equivalent SQL: DATE_DIFF(date_expression_a, date_expression_b, date_part)

  Args:
    date: tf.Tensor of type string. Date in "%F" format.
    part: A string represents the date part. Can be DAY, WEEK,  WEEK_MONDAY,
      WEEK_TUESDAY, WEEK_WEDNESDAY, WEEK_THURSDAY, WEEK_FRIDAY, WEEK_SATURDAY,
      ISOWEEK, MONTH, QUARTER, YEAR,  ISOYEAR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_date_ops.date_trunc(
      date=date,
      part=part,
      name=name,
  )


def format_date(format_string, date, name=None):
  """Returns a date string based on format_string.

  Equivalent SQL: FORMAT_DATE(format_string, date_expr)

  Args:
    format_string: tf.Tensor of type string. Format of the output string.
    date: tf.Tensor of type string. Timestamp in "%F" format.
    name: An optional name for the op.
  """
  return gen_date_ops.format_date(
      format_string=format_string,
      date=date,
      name=name,
  )


def parse_date(format_string, date_string, name=None):
  """Returns a date by parsing a string.

  Equivalent SQL: PARSE_TIMESTAMP(format_string, date_string[, time_zone])

  Args:
    format_string: tf.Tensor of type string. Format of the string date.
    date_string: tf.Tensor of type string. Date in any supported format.
    name: An optional name for the op.
  """
  return gen_date_ops.parse_date(
      format_string=format_string,
      date_string=date_string,
      name=name,
  )


def safe_parse_date(format_string, date_string, name=None):
  """Returns a date by safely parsing a string.

  Equivalent SQL: SAFE.PARSE_TIMESTAMP(format_string, date_string[, time_zone]).
  Returns "1970-01-01" for unsuccessful parsing.

  Args:
    format_string: tf.Tensor of type string. Format of the string date.
    date_string: tf.Tensor of type string. Date in any supported format.
    name: An optional name for the op.
  """
  return gen_date_ops.safe_parse_date(
      format_string=format_string,
      date_string=date_string,
      name=name,
  )
