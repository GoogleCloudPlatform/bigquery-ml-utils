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

import tensorflow as tf

gen_timestamp_ops = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile('timestamp_ops.so')
)

def extract_from_timestamp(part, timestamp, time_zone, name=None):
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
