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

#ifndef THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_MATHUTIL_H_
#define THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_MATHUTIL_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/base/attributes.h"
#include "sql_utils/base/logging.h"

namespace bigquery_ml_utils_base {

class MathUtil {
 public:
  template<typename IntegralType>
  static IntegralType FloorOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
  }

  template<typename IntegralType, bool ceil>
  static IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                         IntegralType denominator);

  // Returns the nonnegative remainder when one integer is divided by another.
  // The modulus must be positive.  Use integral types only (no float or
  // double).
  template <class T>
  static T NonnegativeMod(T a, T b) {
    static_assert(std::is_integral<T>::value, "Integral types only.");
    SQL_DCHECK_GT(b, 0);
    // As of C++11 (per [expr.mul]/4), a%b is in (-b,0] for a<0, b>0.
    T c = a % b;
    return c + (c < 0) * b;
  }

  // Returns the minimum integer value which is a multiple of rounding_value,
  // and greater than or equal to input_value.
  // The input_value must be greater than or equal to zero, and the
  // rounding_value must be greater than zero.
  template <typename IntType>
  static IntType RoundUpTo(IntType input_value, IntType rounding_value) {
    static_assert(std::numeric_limits<IntType>::is_integer,
                  "RoundUpTo() operation type is not integer");
    SQL_DCHECK_GE(input_value, 0);
    SQL_DCHECK_GT(rounding_value, 0);
    const IntType remainder = input_value % rounding_value;
    return (remainder == 0) ? input_value
                            : (input_value - remainder + rounding_value);
  }

  // Decomposes `value` to the form `mantissa * pow(2, exponent)`.  Similar to
  // `std::frexp`, but returns `mantissa` as an integer without normalization.
  //
  // The returned `mantissa` might be a power of 2; this method does not shift
  // the trailing 0 bits out.
  //
  // If `value` is inf, then `mantissa = kint64max` and `exponent = intmax`.
  // If `value` is -inf, then `mantissa = -kint64max` and `exponent = intmax`.
  // If `value` is NaN, then `mantissa = 0` and `exponent = intmax`.
  // If `value` is 0, then `mantissa = 0` and `exponent < 0`.
  //
  // For all cases, `value` is equivalent to
  // `static_cast<double>(mantissa) * std::ldexp(1.0, exponent)`, though the
  // bits might differ (e.g., `-0.0` vs `0.0`, signaling NaN vs quiet NaN).
  //
  // For all cases except NaN,
  // `value = std::ldexp(static_cast<double>(mantissa), exponent)`.
  struct FloatParts {
    int32_t mantissa;
    int exponent;
  };
  static FloatParts Decompose(float value);
  struct DoubleParts {
    int64_t mantissa;
    int exponent;
  };
  static DoubleParts Decompose(double value);

 private:
  // Wraps `x` to the periodic range `[low, high)`
  static double Wrap(double x, double low, double high);
};

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64_t into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template<typename IntegralType, bool ceil>
IntegralType MathUtil::CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  static_assert(std::numeric_limits<IntegralType>::is_integer,
                "CeilOfRatio is only defined for integral types");
  SQL_DCHECK_NE(0, denominator) << "Division by zero is not supported.";
  SQL_DCHECK(!std::numeric_limits<IntegralType>::is_signed ||
             numerator != std::numeric_limits<IntegralType>::lowest() ||
             denominator != -1)
      << "Dividing " << numerator << "by -1 is not supported: it would SIGFPE";

  const IntegralType rounded_toward_zero = numerator / denominator;
  const bool needs_round = (numerator % denominator) != 0;
  // It is important to use >= here, even for the denominator, to ensure that
  // this value is a compile-time constant for unsigned types.
  const bool same_sign = (numerator >= 0) == (denominator >= 0);

  if (ceil) {  // Compile-time condition: not an actual branching
    return rounded_toward_zero +
           static_cast<IntegralType>(same_sign && needs_round);
  } else {
    return rounded_toward_zero -
           static_cast<IntegralType>(!same_sign && needs_round);
  }
}

}  // namespace bigquery_ml_utils_base


#endif  // THIRD_PARTY_PY_BIGQUERY_ML_UTILS_SQL_UTILS_BASE_MATHUTIL_H_
