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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_ENDIAN_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_ENDIAN_H_

#ifdef _MSC_VER
#include <stdlib.h>  // NOLINT(build/include)
#elif defined(__APPLE__)
// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#elif defined(__FreeBSD__)
#include <sys/endian.h>
#elif defined(__GLIBC__)
#include <byteswap.h>  
#endif

#include <cstdint>

#include "absl/base/config.h"
#include "absl/base/port.h"
#include "absl/numeric/int128.h"
#include "sql_utils/base/unaligned_access.h"

namespace bigquery_ml_utils_base {

// Use compiler byte-swapping intrinsics if they are available.  32-bit
// and 64-bit versions are available in Clang and GCC as of GCC 4.3.0.
// The 16-bit version is available in Clang and GCC only as of GCC 4.8.0.
// For simplicity, we enable them all only for GCC 4.8.0 or later.
#if defined(__clang__) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
inline uint64_t gbswap_64(uint64_t host_int) {
  return __builtin_bswap64(host_int);
}
inline uint32_t gbswap_32(uint32_t host_int) {
  return __builtin_bswap32(host_int);
}
inline uint16_t gbswap_16(uint16_t host_int) {
  return __builtin_bswap16(host_int);
}

#elif defined(_MSC_VER)
inline uint64_t gbswap_64(uint64_t host_int) {
  return _byteswap_uint64(host_int);
}
inline uint32_t gbswap_32(uint32_t host_int) {
  return _byteswap_ulong(host_int);
}
inline uint16_t gbswap_16(uint16_t host_int) {
  return _byteswap_ushort(host_int);
}

#elif defined(__APPLE__)
inline uint64_t gbswap_64(uint64_t host_int) { return OSSwapInt16(host_int); }
inline uint32_t gbswap_32(uint32_t host_int) { return OSSwapInt32(host_int); }
inline uint16_t gbswap_16(uint16_t host_int) { return OSSwapInt64(host_int); }

#else
inline uint64_t gbswap_64(uint64_t host_int) {
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__)
  // Adapted from /usr/include/byteswap.h.  Not available on Mac.
  if (__builtin_constant_p(host_int)) {
    return __bswap_constant_64(host_int);
  } else {
    uint64_t result;
    __asm__("bswap %0" : "=r"(result) : "0"(host_int));
    return result;
  }
#elif defined(__GLIBC__)
  return bswap_64(host_int);
#else
  return (((x & uint64_t{(0xFF}) << 56) |
          ((x & uint64_t{(0xFF00}) << 40) |
          ((x & uint64_t{(0xFF0000}) << 24) |
          ((x & uint64_t{(0xFF000000}) << 8) |
          ((x & uint64_t{(0xFF00000000}) >> 8) |
          ((x & uint64_t{(0xFF0000000000}) >> 24) |
          ((x & uint64_t{(0xFF000000000000}) >> 40) |
          ((x & uint64_t{(0xFF00000000000000}) >> 56));
#endif  // bswap_64
}

inline uint32_t gbswap_32(uint32_t host_int) {
#if defined(__GLIBC__)
  return bswap_32(host_int);
#else
  return (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) |
          ((x & 0xFF000000) >> 24));
#endif
}

inline uint16_t gbswap_16(uint16_t host_int) {
#if defined(__GLIBC__)
  return bswap_16(host_int);
#else
  return uint16_t{((x & 0xFF) << 8) | ((x & 0xFF00) >> 8)};
#endif
}

#endif  // intrinsics available

inline absl::uint128 gbswap_128(absl::uint128 host_int) {
  return absl::MakeUint128(gbswap_64(absl::Uint128Low64(host_int)),
                           gbswap_64(absl::Uint128High64(host_int)));
}

#ifdef ABSL_IS_LITTLE_ENDIAN

// Definitions for ntohl etc. that don't require us to include
// netinet/in.h. We wrap gbswap_32 and gbswap_16 in functions rather
// than just #defining them because in debug mode, gcc doesn't
// correctly handle the (rather involved) definitions of bswap_32.
// gcc guarantees that inline functions are as fast as macros, so
// this isn't a performance hit.
inline uint16_t ghtons(uint16_t x) { return gbswap_16(x); }
inline uint32_t ghtonl(uint32_t x) { return gbswap_32(x); }
inline uint64_t ghtonll(uint64_t x) { return gbswap_64(x); }

#elif defined ABSL_IS_BIG_ENDIAN

// These definitions are simpler on big-endian machines
// These are functions instead of macros to avoid self-assignment warnings
// on calls such as "i = ghtnol(i);".  This also provides type checking.
inline uint16_t ghtons(uint16_t x) { return x; }
inline uint32_t ghtonl(uint32_t x) { return x; }
inline uint64_t ghtonll(uint64_t x) { return x; }

#else
#error \
    "Unsupported byte order: Either ABSL_IS_BIG_ENDIAN or " \
       "ABSL_IS_LITTLE_ENDIAN must be defined"
#endif  // byte order

inline uint16_t gntohs(uint16_t x) { return ghtons(x); }
inline uint32_t gntohl(uint32_t x) { return ghtonl(x); }
inline uint64_t gntohll(uint64_t x) { return ghtonll(x); }

// Utilities to convert numbers between the current hosts's native byte
// order and little-endian byte order
//
// Load/Store methods are alignment safe
class LittleEndian {
 public:
// Conversion functions.
#ifdef ABSL_IS_LITTLE_ENDIAN

  static uint16_t FromHost16(uint16_t x) { return x; }
  static uint16_t ToHost16(uint16_t x) { return x; }

  static uint32_t FromHost32(uint32_t x) { return x; }
  static uint32_t ToHost32(uint32_t x) { return x; }

  static uint64_t FromHost64(uint64_t x) { return x; }
  static uint64_t ToHost64(uint64_t x) { return x; }

  static absl::uint128 FromHost128(absl::uint128 x) { return x; }
  static absl::uint128 ToHost128(absl::uint128 x) { return x; }

  inline constexpr bool IsLittleEndian() const { return true; }

#elif defined ABSL_IS_BIG_ENDIAN

  static uint16_t FromHost16(uint16_t x) { return gbswap_16(x); }
  static uint16_t ToHost16(uint16_t x) { return gbswap_16(x); }

  static uint32_t FromHost32(uint32_t x) { return gbswap_32(x); }
  static uint32_t ToHost32(uint32_t x) { return gbswap_32(x); }

  static uint64_t FromHost64(uint64_t x) { return gbswap_64(x); }
  static uint64_t ToHost64(uint64_t x) { return gbswap_64(x); }

  static absl::uint128 FromHost128(absl::uint128 x) { return gbswap_128(x); }
  static absl::uint128 ToHost128(absl::uint128 x) { return gbswap_128(x); }

  inline constexpr bool IsLittleEndian() const { return false; }

#endif /* ENDIAN */

  // Functions to do unaligned loads and stores in little-endian order.
  static uint16_t Load16(const void* p) {
    return ToHost16(SQL_INTERNAL_UNALIGNED_LOAD16(p));
  }

  static void Store16(void* p, uint16_t v) {
    SQL_INTERNAL_UNALIGNED_STORE16(p, FromHost16(v));
  }

  static uint32_t Load32(const void* p) {
    return ToHost32(SQL_INTERNAL_UNALIGNED_LOAD32(p));
  }

  static void Store32(void* p, uint32_t v) {
    SQL_INTERNAL_UNALIGNED_STORE32(p, FromHost32(v));
  }

  static uint64_t Load64(const void *p) {
    return ToHost64(SQL_INTERNAL_UNALIGNED_LOAD64(p));
  }

  static void Store64(void *p, uint64_t v) {
    SQL_INTERNAL_UNALIGNED_STORE64(p, FromHost64(v));
  }

  static absl::uint128 Load128(const void* p) {
    return absl::MakeUint128(ToHost64(SQL_INTERNAL_UNALIGNED_LOAD64(
                                 reinterpret_cast<const uint64_t*>(p) + 1)),
                             ToHost64(SQL_INTERNAL_UNALIGNED_LOAD64(p)));
  }

  static void Store128(void* p, const absl::uint128 v) {
    SQL_INTERNAL_UNALIGNED_STORE64(p, FromHost64(absl::Uint128Low64(v)));
    SQL_INTERNAL_UNALIGNED_STORE64(reinterpret_cast<uint64_t*>(p) + 1,
                                       FromHost64(absl::Uint128High64(v)));
  }
};

}  // namespace bigquery_ml_utils_base

#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_ENDIAN_H_
