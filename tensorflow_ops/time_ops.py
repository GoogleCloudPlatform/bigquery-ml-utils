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

"""Python wrapper for BQML time custom ops."""

from bigquery_ml_utils.tensorflow_ops.load_module import load_module

gen_time_ops = load_module("_time_ops.so")


def time_from_components(hour, minute, second, name=None):
  """Returns a time using INT64 values representing the hour, minute, and second.

  Equivalent SQL: TIME(hour, minute, second)

  Args:
    hour: tf.Tensor of type int64 representing the hour.
    minute: tf.Tensor of type int64 representing the minute.
    second: tf.Tensor of type int64 representing the second.
    name: An optional name for the op.
  """
  return gen_time_ops.time_from_components(
      hour=hour, minute=minute, second=second, name=name
  )


def time_from_timestamp(timestamp, time_zone="UTC", name=None):
  """Returns a time from a timestamp.

  Equivalent SQL: TIME(timestamp, [time_zone])

  Args:
    timestamp: tf.Tensor of type string. Timestamp in "%F %H:%M:%E1S %z" format.
    time_zone: A string represents the timezone. Case sensitive.
    name: An optional name for the op.
  """
  return gen_time_ops.time_from_timestamp(
      timestamp=timestamp, time_zone=time_zone, name=name
  )


def time_from_datetime(datetime, name=None):
  """Returns a time from a datetime.

  Equivalent SQL: TIME(datetime)

  Args:
    datetime: tf.Tensor of type string. Datetime in "%F %H:%M:%E6S" format.
    name: An optional name for the op.
  """
  return gen_time_ops.time_from_datetime(datetime=datetime, name=name)


def time_add(time, interval, part, name=None):
  """Returns a time by adding interval to the time..

  Equivalent SQL: TIME(datetime)

  Args:
    time: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    interval: tf.Tensor of type int64. Integer represents the unit of part.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_time_ops.time_add(
      time=time, interval=interval, part=part, name=name
  )


def time_sub(time, interval, part, name=None):
  """Returns a time by subtracting interval to the time.

  Equivalent SQL: TIME(datetime)

  Args:
    time: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    interval: tf.Tensor of type int64. Integer represents the unit of part.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_time_ops.time_sub(
      time=time, interval=interval, part=part, name=name
  )


def time_diff(time_a, time_b, part, name=None):
  """Returns the whole number of specified part intervals between two times.

  Equivalent SQL: TIME_DIFF(time_expression_a, time_expression_b, part)

  Args:
    time_a: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    time_b: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_time_ops.time_diff(
      time_a=time_a, time_b=time_b, part=part, name=name
  )


def time_trunc(time, part, name=None):
  """Returns a time by truncating a time to the granularity of part.

  Equivalent SQL: TIME_TRUNC(time_expression, time_part)

  Args:
    time: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_time_ops.time_trunc(time=time, part=part, name=name)


def extract_from_time(time, part, name=None):
  """Returns a value that corresponds to the specified part from a supplied time.

  Equivalent SQL: EXTRACT(part FROM time_expression)

  Args:
    time: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    part: A string represents the datetime part. Can be MICROSECOND,
      MILLISECOND, SECOND, MINUTE, HOUR. Case insensitive.
    name: An optional name for the op.
  """
  return gen_time_ops.extract_from_time(time=time, part=part, name=name)


def parse_time(format_string, time_string, name=None):
  """Returns a time by parsing a string representation of time.

  Equivalent SQL: PARSE_TIME(format_string, time_string)

  Args:
    format_string: tf.Tensor of type string. Format of the string time.
    time_string: tf.Tensor of type string. Time in any supported format.
    name: An optional name for the op.
  """
  return gen_time_ops.parse_time(
      format_string=format_string, time_string=time_string, name=name
  )


def safe_parse_time(format_string, time_string, name=None):
  """Returns a time by safely parsing a string representation of time.

  Equivalent SQL: SAFE.PARSE_TIME(format_string, time_string).
  Returns "12:34:56.123456" for unsuccessful parsing.

  Args:
    format_string: tf.Tensor of type string. Format of the string time.
    time_string: tf.Tensor of type string. Time in any supported format.
    name: An optional name for the op.
  """
  return gen_time_ops.safe_parse_time(
      format_string=format_string, time_string=time_string, name=name
  )


def format_time(format_string, time, name=None):
  """Returns a time by parsing a string representation of time.

  Equivalent SQL: TIME_TRUNC(time_expression, time_part)

  Args:
    format_string: tf.Tensor of type string. Format of the string time.
    time: tf.Tensor of type string. Time in "%H:%M:%E6S" format.
    name: An optional name for the op.
  """
  return gen_time_ops.format_time(
      format_string=format_string, time=time, name=name
  )
