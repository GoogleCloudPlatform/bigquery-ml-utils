/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_ops/utils.h"

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/parse_date_time.h"
#include "tensorflow_ops/constants.h"
#include "tensorflow/core/framework/op_kernel.h"

using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

namespace bigquery_ml_utils {

::tsl::Status ParseInputDateTimestampPart(
    absl::string_view part, absl::string_view function_name,
    functions::DateTimestampPart* out,
    absl::flat_hash_set<functions::DateTimestampPart> supported_parts) {
  int part_int = functions::DateTimestampPart_FromName(part);
  if (part_int == -1) {
    return InvalidArgument(
        absl::Substitute("Invalid part in $0: $1", function_name, part));
  }

  *out = static_cast<functions::DateTimestampPart>(part_int);
  if (!supported_parts.empty() && !supported_parts.contains(*out)) {
    return InvalidArgument(
        absl::Substitute("Unsupported part in $0: $1", function_name, part));
  }
  return ::tsl::OkStatus();
}

::tsl::Status ParseInputTimeZone(absl::string_view time_zone,
                                 absl::string_view function_name,
                                 absl::TimeZone* out) {
  absl::Status status = functions::MakeTimeZone(time_zone, out);
  if (!status.ok()) {
    return InvalidArgument(absl::Substitute("Invalid time zone in $0: $1",
                                            function_name, time_zone));
  }

  return ::tsl::OkStatus();
}

::tsl::Status ParseInputDate(absl::string_view date,
                             absl::string_view function_name, int32_t* out) {
  absl::Status status = functions::ParseStringToDate(
      kDateFormatString, date, /*parse_version2=*/true, out);
  if (!status.ok()) {
    return InvalidArgument(
        absl::Substitute("Invalid date in $0: $1", function_name, date));
  }

  return ::tsl::OkStatus();
}

::tsl::Status ParseInputDatetime(absl::string_view datetime,
                                 absl::string_view function_name,
                                 DatetimeValue* out) {
  absl::Status status = functions::ParseStringToDatetime(
      kDatetimeFormatString, datetime, functions::kMicroseconds,
      /*parse_version2=*/true, out);
  if (!status.ok()) {
    return InvalidArgument(absl::Substitute("Invalid datetime in $0: $1",
                                            function_name, datetime));
  }

  return ::tsl::OkStatus();
}

::tsl::Status ParseInputTimestamp(absl::string_view timestamp,
                                  absl::TimeZone time_zone,
                                  absl::string_view function_name,
                                  int64_t* out) {
  absl::Status status = functions::ParseStringToTimestamp(
      kTimestampFormatString, timestamp, time_zone,
      /*parse_version2=*/true, out);
  if (!status.ok()) {
    return InvalidArgument(absl::Substitute("Invalid timestamp in $0: $1",
                                            function_name, timestamp));
  }

  return ::tsl::OkStatus();
}

::tsl::Status FormatOutputTimestamp(int64_t ts, absl::string_view function_name,
                                    std::string* out) {
  functions::FormatDateTimestampOptions format_options = {
      .expand_Q = false,
      .expand_J = false,
  };
  // Output at the UTC time zone.
  absl::Status status = functions::FormatTimestampToString(
      kTimestampFormatString, ts, absl::UTCTimeZone(), format_options, out);
  if (!status.ok()) {
    return Internal(absl::Substitute(
        "Error to format output timestamp in $0 with status: $1", function_name,
        status.ToString()));
  }

  return ::tsl::OkStatus();
}

}  // namespace bigquery_ml_utils
