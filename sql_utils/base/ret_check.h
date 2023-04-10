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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_RET_CHECK_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_RET_CHECK_H_

#include <string>

#include "absl/status/status.h"
#include "sql_utils/base/logging.h"
#include "sql_utils/base/source_location.h"
#include "sql_utils/base/status_builder.h"
#include "sql_utils/base/status_macros.h"

namespace bigquery_ml_utils_base {
namespace internal_ret_check {

// Returns a StatusBuilder that corresponds to a `SQL_RET_CHECK` failure.
StatusBuilder RetCheckFailSlowPath(SourceLocation location);
StatusBuilder RetCheckFailSlowPath(SourceLocation location,
                                   const char* condition);
StatusBuilder RetCheckFailSlowPath(SourceLocation location,
                                   const char* condition,
                                   const absl::Status& s);

// Takes ownership of `condition`.  This API is a little quirky because it is
// designed to make use of the `::Check_*Impl` methods that implement `CHECK_*`
// and `DCHECK_*`.
StatusBuilder RetCheckFailSlowPath(SourceLocation location,
                                   std::string* condition);

inline StatusBuilder RetCheckImpl(const absl::Status& status,
                                  const char* condition,
                                  SourceLocation location) {
  if (ABSL_PREDICT_TRUE(status.ok()))
    return StatusBuilder(absl::OkStatus(), location);
  return RetCheckFailSlowPath(location, condition, status);
}

}  // namespace internal_ret_check
}  // namespace bigquery_ml_utils_base

#define SQL_RET_CHECK(cond)                                                \
  while (ABSL_PREDICT_FALSE(!(cond)))                                          \
  return ::bigquery_ml_utils_base::internal_ret_check::RetCheckFailSlowPath(SQL_LOC, \
                                                                  #cond)

#define SQL_RET_CHECK_FAIL() \
  return ::bigquery_ml_utils_base::internal_ret_check::RetCheckFailSlowPath(SQL_LOC)

// Takes an expression returning absl::Status and asserts that the
// status is `ok()`.  If not, it returns an internal error.
//
// This is similar to `SQL_RETURN_IF_ERROR` in that it propagates errors.
// The difference is that it follows the behavior of `SQL_RET_CHECK`,
// returning an internal error (wrapping the original error text), including the
// filename and line number, and logging a stack trace.
//
// This is appropriate to use to write an assertion that a function that returns
// `absl::Status` cannot fail, particularly when the error code itself
// should not be surfaced.
#define SQL_RET_CHECK_OK(status)                                        \
  SQL_RETURN_IF_ERROR(::bigquery_ml_utils_base::internal_ret_check::RetCheckImpl( \
      (status), #status, SQL_LOC))

#if defined(STATIC_ANALYSIS)
#define SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(name, op, lhs, rhs) \
  SQL_RET_CHECK((lhs)op(rhs))
#else
#define SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(name, op, lhs, rhs)        \
  while (std::string* _result = bigquery_ml_utils_base::Check_##name##Impl(              \
             ::bigquery_ml_utils_base::GetReferenceableValue(lhs),                       \
             ::bigquery_ml_utils_base::GetReferenceableValue(rhs),                       \
             #lhs " " #op " " #rhs))                                           \
  return ::bigquery_ml_utils_base::internal_ret_check::RetCheckFailSlowPath(SQL_LOC, \
                                                                  _result)
#endif

#define SQL_RET_CHECK_EQ(lhs, rhs) \
  SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(EQ, ==, lhs, rhs)
#define SQL_RET_CHECK_NE(lhs, rhs) \
  SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(NE, !=, lhs, rhs)
#define SQL_RET_CHECK_LE(lhs, rhs) \
  SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(LE, <=, lhs, rhs)
#define SQL_RET_CHECK_LT(lhs, rhs) \
  SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(LT, <, lhs, rhs)
#define SQL_RET_CHECK_GE(lhs, rhs) \
  SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(GE, >=, lhs, rhs)
#define SQL_RET_CHECK_GT(lhs, rhs) \
  SQL_STATUS_MACROS_INTERNAL_RET_CHECK_OP(GT, >, lhs, rhs)

#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_RET_CHECK_H_
