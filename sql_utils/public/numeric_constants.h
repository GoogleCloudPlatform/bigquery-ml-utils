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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_PUBLIC_NUMERIC_CONSTANTS_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_PUBLIC_NUMERIC_CONSTANTS_H_

#include <cstdint>

#include <cstdint>  

// Common constants used by numeric_parser and numeric_value libraries.
namespace bigquery_ml_utils {
namespace internal {
constexpr uint32_t k1e9 = 1000 * 1000 * 1000;
constexpr uint64_t k1e10 = static_cast<uint64_t>(k1e9) * 10;
constexpr uint64_t k1e16 = 10000000000000000ULL;
constexpr uint64_t k1e17 = k1e16 * 10;
constexpr uint64_t k1e18 = k1e17 * 10;
constexpr uint64_t k1e19 = k1e18 * 10;
constexpr __int128 k1e38 = static_cast<__int128>(k1e19) * k1e19;
}  // namespace internal
}  // namespace bigquery_ml_utils

#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_PUBLIC_NUMERIC_CONSTANTS_H_
