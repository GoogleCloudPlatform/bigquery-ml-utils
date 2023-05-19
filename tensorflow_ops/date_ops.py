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
