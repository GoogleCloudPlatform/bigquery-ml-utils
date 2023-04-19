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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_COMMON_STATUS_PAYLOAD_UTILS_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_COMMON_STATUS_PAYLOAD_UTILS_H_

#include <string>

#include "sql_utils/base/logging.h"
#include "google/protobuf/message.h"
#include "absl/memory/memory.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "sql_utils/base/status.h"
#include "sql_utils/base/status_payload.h"

namespace bigquery_ml_utils {
namespace internal {

// Whether the given status has any payload at all.
bool HasPayload(const absl::Status& status);

// Gets the number payloads.
int GetPayloadCount(const absl::Status& status);

template <class T>
std::string GetTypeUrl() {
  return bigquery_ml_utils_base::GetTypeUrl<T>();
}

// Whether the given status has exactly one payload of type T.
template <class T>
bool HasPayloadWithType(const absl::Status& status) {
  return status.GetPayload(bigquery_ml_utils_base::GetTypeUrl<T>()).has_value();
}

// Gets the payload of type T from the status proto. Results undefined if
// the status does not contain a payload of the given type, but will not crash.
template <class T>
T GetPayload(const absl::Status& status) {
  std::optional<absl::Cord> payload = status.GetPayload(GetTypeUrl<T>());
  if (!payload.has_value()) {
    return T();
  }
  T proto;
  if (!proto.ParseFromString(std::string(*payload))) {
    proto.Clear();  // Prefer empty over partial??
  }
  return proto;
}

// Mutates `status` by removing any attached payload of type T.
template <class T>
void ErasePayloadTyped(absl::Status* status) {
  status->ErasePayload(GetTypeUrl<T>());
}

// Attaches the given payload. This will overwrite any previous payload with
// the same type.
template <class T>
void AttachPayload(absl::Status* status, const T& payload) {
  bigquery_ml_utils_base::AttachPayload<T>(status, payload);
}

// Creates a human readable string from the status payload (or empty if there
// is no payload). Exact form is not defined.
std::string PayloadToString(const absl::Status& status);

// Creates a human readable string from the status, including its payload.
// Exact form is not defined.
std::string StatusToString(const absl::Status& status);

inline absl::Status AppendMessage(const absl::Status& status,
                                  absl::string_view msg) {
  absl::string_view merged_msg_view = msg;
  std::string merged_msg;
  if (!status.message().empty()) {
    absl::StrAppend(&merged_msg, status.message(), "; ", msg);
    merged_msg_view = merged_msg;
  }

  absl::Status merged_status(status.code(), merged_msg_view);
  return merged_status;
}

// Update the current status with a new status. To ensure no error is ignored,
// requires that if neither status is OK, then both should share the same code.
// 1. If the new status is OK: NOOP
// 2. Otherwise:
//    a. If the current status is OK, the new one takes over.
//    b. Both statuses are not OK. The message of the new status is appended to
//       the current one.
inline void UpdateStatus(absl::Status* status, const absl::Status& new_status) {
  SQL_DCHECK_NE(status, nullptr);
  if (new_status.ok()) {
    return;
  }
  if (status->ok()) {
    *status = new_status;
    return;
  }

  SQL_DCHECK_EQ(status->code(), new_status.code());
  *status = AppendMessage(*status, new_status.message());
}

}  // namespace internal
}  // namespace bigquery_ml_utils


#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_COMMON_STATUS_PAYLOAD_UTILS_H_
