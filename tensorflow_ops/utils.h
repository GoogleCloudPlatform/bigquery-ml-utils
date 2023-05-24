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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_TENSORFLOW_OPS_UTILS_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_TENSORFLOW_OPS_UTILS_H_

#include "absl/container/flat_hash_set.h"
#include "sql_utils/public/civil_time.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace bigquery_ml_utils {

::tsl::Status ParseInputDateTimestampPart(
    absl::string_view part, absl::string_view function_name,
    functions::DateTimestampPart* out,
    const absl::flat_hash_set<functions::DateTimestampPart>& supported_parts =
        {});

::tsl::Status ParseInputDate(absl::string_view date,
                             absl::string_view function_name, int32_t* out);

::tsl::Status ParseInputDatetime(absl::string_view datetime,
                                 absl::string_view function_name,
                                 DatetimeValue* out);

::tsl::Status ParseInputTime(absl::string_view time,
                             absl::string_view function_name, TimeValue* out);

::tsl::Status ParseInputTimestamp(absl::string_view timestamp,
                                  const absl::TimeZone& time_zone,
                                  absl::string_view function_name,
                                  int64_t* out);

::tsl::Status FormatOutputDatetime(const DatetimeValue& dt,
                                   absl::string_view function_name,
                                   std::string* out);

::tsl::Status FormatOutputDate(int32_t d, absl::string_view function_name,
                               std::string* out);

::tsl::Status FormatOutputTime(const TimeValue& time,
                               absl::string_view function_name,
                               std::string* out);

::tsl::Status FormatOutputTimestamp(int64_t ts, absl::string_view function_name,
                                    std::string* out);

::tsl::Status ToTslStatus(absl::string_view function_name,
                          const absl::Status& status);

}  // namespace bigquery_ml_utils

#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_TENSORFLOW_OPS_UTILS_H_
