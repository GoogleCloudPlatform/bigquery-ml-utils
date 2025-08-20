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
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/parse_date_time.h"
#include "sql_utils/public/interval_value.h"
#include "tensorflow_ops/constants.h"
#include "tensorflow/core/framework/op_kernel.h"

using ::tensorflow::errors::InvalidArgument;

namespace bigquery_ml_utils {

::tsl::Status ParseInputDateTimestampPart(
    absl::string_view part, absl::string_view function_name,
    functions::DateTimestampPart* out,
    const absl::flat_hash_set<functions::DateTimestampPart>& supported_parts) {
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
  return absl::OkStatus();
}

::tsl::Status ParseInputDate(absl::string_view date,
                             absl::string_view function_name, int32_t* out) {
  return ToTslStatus(function_name, functions::ParseStringToDate(
                                        kDateFormatString, date,
                                        /*parse_version2=*/true, out));
}

::tsl::Status ParseInputDatetime(absl::string_view datetime,
                                 absl::string_view function_name,
                                 DatetimeValue* out) {
  return ToTslStatus(function_name, functions::ParseStringToDatetime(
                                        kDatetimeFormatString, datetime,
                                        functions::kMicroseconds,
                                        /*parse_version2=*/true, out));
}

::tsl::Status ParseInputTime(absl::string_view time,
                             absl::string_view function_name, TimeValue* out) {
  return ToTslStatus(function_name, functions::ParseStringToTime(
                                        kTimeFormatString, time,
                                        functions::kMicroseconds, out));
}

::tsl::Status ParseInputTimestamp(absl::string_view timestamp,
                                  const absl::TimeZone& time_zone,
                                  absl::string_view function_name,
                                  int64_t* out) {
  return ToTslStatus(function_name,
                     functions::ParseStringToTimestamp(
                         kTimestampFormatString, timestamp, time_zone,
                         /*parse_version2=*/true, out));
}

::tsl::Status FormatOutputDatetime(const DatetimeValue& dt,
                                   absl::string_view function_name,
                                   std::string* out) {
  // Output 3 formats dynamically to align with CAST AS STRING in BQML.
  return ToTslStatus(function_name, functions::ConvertDatetimeToString(
                                        dt, functions::kMicroseconds, out));
}

::tsl::Status FormatOutputDate(int32_t d, absl::string_view function_name,
                               std::string* out) {
  return ToTslStatus(function_name,
                     functions::FormatDateToString(kDateFormatString, d, out));
}

::tsl::Status FormatOutputTime(const TimeValue& time,
                               absl::string_view function_name,
                               std::string* out) {
  // Output 3 formats dynamically to align with CAST AS STRING in BQML.
  return ToTslStatus(function_name, functions::ConvertTimeToString(
                                        time, functions::kMicroseconds, out));
}

::tsl::Status FormatOutputTimestamp(int64_t ts, absl::string_view function_name,
                                    std::string* out) {
  functions::FormatDateTimestampOptions format_options = {
      .expand_Q = true,
      .expand_J = true,
  };
  // Output at the UTC time zone.
  return ToTslStatus(function_name,
                     functions::FormatTimestampToString(kTimestampFormatString,
                                                        ts, absl::UTCTimeZone(),
                                                        format_options, out));
}

::tsl::Status ToTslStatus(absl::string_view function_name,
                          const absl::Status& status) {
  if (status.ok()) {
    return absl::OkStatus();
  }

  return ::tsl::Status(static_cast<tensorflow::errors::Code>(status.code()),
                       absl::Substitute("Error in $0 with status: $1",
                                        function_name, status.ToString()));
}

absl::StatusOr<IntervalValue> GetIntervalValue(
    int64_t diff, functions::DateTimestampPart part_enum) {
  switch (part_enum) {
    case functions::MILLISECOND:
      diff = IntervalValue::kMicrosInMilli * diff;
      return IntervalValue::FromMicros(diff);
    case functions::MICROSECOND:
      return IntervalValue::FromMicros(diff);
    default:
      return IntervalValue::FromInteger(diff, part_enum);
  }
}

}  // namespace bigquery_ml_utils
