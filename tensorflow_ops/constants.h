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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_TENSORFLOW_OPS_CONSTANTS_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_TENSORFLOW_OPS_CONSTANTS_H_

#include "absl/strings/string_view.h"

namespace bigquery_ml_utils {

// Date string format for SavedModel input.
inline constexpr absl::string_view kDateFormatString = "%F";
// Datetime string format for SavedModel input.
inline constexpr absl::string_view kDatetimeFormatString = "%F %H:%M:%E6S";
// Time string format for SavedModel input.
inline constexpr absl::string_view kTimeFormatString = "%H:%M:%E6S";
// Timestamp string format for SavedModel input.
inline constexpr absl::string_view kTimestampFormatString = "%F %H:%M:%E1S %z";

inline constexpr absl::string_view kNullDate = "1970-01-01";
inline constexpr absl::string_view kNullDatetime = "1970-01-01 00:00:00.000000";
inline constexpr absl::string_view kNullTime = "12:34:56.123456";
inline constexpr absl::string_view kNullTimestamp =
    "1970-01-01 00:00:00.0 +0000";

}  // namespace bigquery_ml_utils

#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_TENSORFLOW_OPS_CONSTANTS_H_
