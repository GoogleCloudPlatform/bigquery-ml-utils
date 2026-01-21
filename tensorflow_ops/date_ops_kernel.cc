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
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "sql_utils/public/civil_time.h"
#include "sql_utils/public/functions/cast_date_time.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "sql_utils/public/functions/parse_date_time.h"
#include "sql_utils/public/types/timestamp_util.h"
#include "tensorflow_ops/constants.h"
#include "tensorflow_ops/utils.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::tstring;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::errors::OutOfRange;

namespace bigquery_ml_utils {

class ExtractFromDate : public OpKernel {
 public:
  explicit ExtractFromDate(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor.
    const Tensor& date_tensor = context->input(0);
    auto date = date_tensor.flat<tstring>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(1);
    std::string part = absl::AsciiStrToLower(part_tensor.flat<tstring>()(0));
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::DAY, functions::DAYOFWEEK, functions::DAYOFYEAR,
             functions::WEEK, functions::WEEK_MONDAY, functions::WEEK_TUESDAY,
             functions::WEEK_WEDNESDAY, functions::WEEK_THURSDAY,
             functions::WEEK_FRIDAY, functions::WEEK_SATURDAY,
             functions::ISOWEEK, functions::MONTH, functions::QUARTER,
             functions::YEAR, functions::ISOYEAR});
    functions::DateTimestampPart part_enum;
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the date tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_value;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_value));

      // Extract part from the date.
      int32_t out;
      OP_REQUIRES_OK(context,
                     ToStatus(name(), functions::ExtractFromDate(
                                          part_enum, date_value, &out)));

      // Set the output value.
      output_flat(i) = static_cast<int64_t>(out);
    }
  }
};

class DateFromComponents : public OpKernel {
 public:
  explicit DateFromComponents(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the year tensor
    const Tensor& year_tensor = context->input(0);
    auto year = year_tensor.flat<int64_t>();
    // Grab the month tensor
    const Tensor& month_tensor = context->input(1);
    auto month = month_tensor.flat<int64_t>();
    // Grab the day tensor
    const Tensor& day_tensor = context->input(2);
    auto day = day_tensor.flat<int64_t>();

    OP_REQUIRES(
        context, year.size() == month.size() && year.size() == day.size(),
        InvalidArgument(absl::Substitute("Errors in $0: Inputs must have the "
                                         "same shape, but are: $1, $2, $3",
                                         name(), year.size(), month.size(),
                                         day.size())));

    // Create an output tensor with the shape of the year tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, year_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = year.size();
    for (int i = 0; i < N; i++) {
      // Construct the date.
      int32_t date;
      OP_REQUIRES_OK(
          context, ToStatus(name(), functions::ConstructDate(year(i), month(i),
                                                             day(i), &date)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class DateFromTimestamp : public OpKernel {
 public:
  explicit DateFromTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time zone tensor
    const Tensor& time_zone_tensor = context->input(1);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(
          context,
          ParseInputTimestamp(timestamp(i), absl::UTCTimeZone(), name(), &ts));

      // Extract date from timestamp.
      int32_t date;
      OP_REQUIRES_OK(context, ToStatus(name(), functions::ExtractFromTimestamp(
                                                   functions::DATE, ts,
                                                   functions::kMicroseconds,
                                                   time_zone, &date)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class DateFromDatetime : public OpKernel {
 public:
  explicit DateFromDatetime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor
    const Tensor& datetime_tensor = context->input(0);
    auto datetime = datetime_tensor.flat<tstring>();

    // Create an output tensor with the shape of the datetime tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue dt;
      OP_REQUIRES_OK(context, ParseInputDatetime(datetime(i), name(), &dt));

      // Extract date from datetime.
      int32_t date;
      OP_REQUIRES_OK(context,
                     ToStatus(name(), functions::ExtractFromDatetime(
                                          functions::DATE, dt, &date)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class CastToDateFromString : public OpKernel {
 public:
  explicit CastToDateFromString(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date_string tensor
    const Tensor& date_string_tensor = context->input(0);
    auto date_string = date_string_tensor.flat<tstring>();
    // Grab the format tensor
    const Tensor& format_tensor = context->input(1);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the with_format tensor
    const Tensor& with_format_tensor = context->input(2);
    bool with_format = with_format_tensor.flat<bool>()(0);

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, date_string_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date_string.size();
    for (int i = 0; i < N; i++) {
      // Convert string to date.
      int32_t date;
      if (with_format) {
        // Convert string with format
        int32_t current_date = functions::CurrentDate(absl::UTCTimeZone());
        OP_REQUIRES_OK(context, ToStatus(name(), functions::CastStringToDate(
                                                     format, date_string(i),
                                                     current_date, &date)));
      } else {
        // Convert string without format
        OP_REQUIRES_OK(context, ToStatus(name(), functions::ConvertStringToDate(
                                                     date_string(i), &date)));
      }
      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

::tsl::Status DateFromIntOperator(int64_t in, int32_t* out) {
  if (in > types::kDateMax || in < types::kDateMin) {
    std::string date_min;
    functions::ConvertDateToString(types::kDateMin, &date_min).IgnoreError();
    std::string date_max;
    functions::ConvertDateToString(types::kDateMax, &date_max).IgnoreError();
    return OutOfRange(
        absl::Substitute("DATE value is out of allowed range: from $0 to $1.",
                         date_min, date_max));
  }

  *out = static_cast<int32_t>(in);
  return absl::OkStatus();
}

class DateFromUnixDate : public OpKernel {
 public:
  explicit DateFromUnixDate(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the num_days tensor
    const Tensor& num_days_tensor = context->input(0);
    auto num_days = num_days_tensor.flat<int64_t>();

    // Create an output tensor with the shape of the num_days tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, num_days_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = num_days.size();
    for (int i = 0; i < N; i++) {
      // Convert num_days to date
      int32_t date;
      OP_REQUIRES_OK(context, DateFromIntOperator(num_days(i), &date));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class DateAdd : public OpKernel {
 public:
  explicit DateAdd(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor
    const Tensor& date_tensor = context->input(0);
    auto date = date_tensor.flat<tstring>();
    // Grab the interval tensor
    const Tensor& interval_tensor = context->input(1);
    auto interval_int = interval_tensor.flat<int64_t>();
    OP_REQUIRES(context, date.size() == interval_int.size(),
                InvalidArgument(absl::Substitute(
                    "Error in $0: date and interval must have the same shape, "
                    "but are $1, $2",
                    name(), date.size(), interval_int.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::DAY, functions::WEEK, functions::MONTH,
             functions::QUARTER, functions::YEAR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_in;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_in));

      // Add interval.
      int32_t date_out;
      OP_REQUIRES_OK(
          context,
          ToStatus(name(), functions::AddDate(date_in, part_enum,
                                              interval_int(i), &date_out)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date_out, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class DateSub : public OpKernel {
 public:
  explicit DateSub(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor
    const Tensor& date_tensor = context->input(0);
    auto date = date_tensor.flat<tstring>();
    // Grab the interval tensor
    const Tensor& interval_tensor = context->input(1);
    auto interval_int = interval_tensor.flat<int64_t>();
    OP_REQUIRES(context, date.size() == interval_int.size(),
                InvalidArgument(absl::Substitute(
                    "Error in $0: date and interval must have the same shape, "
                    "but are $1, $2",
                    name(), date.size(), interval_int.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::DAY, functions::WEEK, functions::MONTH,
             functions::QUARTER, functions::YEAR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_in;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_in));

      // Sub interval.
      int32_t date_out;
      OP_REQUIRES_OK(
          context,
          ToStatus(name(), functions::AddDate(date_in, part_enum,
                                              -interval_int(i), &date_out)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date_out, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class DateDiff : public OpKernel {
 public:
  explicit DateDiff(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date_a tensor
    const Tensor& date_a_tensor = context->input(0);
    auto date_a = date_a_tensor.flat<tstring>();
    // Grab the date_b tensor
    const Tensor& date_b_tensor = context->input(1);
    auto date_b = date_b_tensor.flat<tstring>();
    OP_REQUIRES(context, date_a.size() == date_b.size(),
                InvalidArgument(absl::Substitute(
                    "Error in $0: date_a and date_b must have the same shape, "
                    "but are $1, $2",
                    name(), date_a.size(), date_b.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::DAY, functions::WEEK, functions::WEEK_MONDAY,
             functions::WEEK_TUESDAY, functions::WEEK_WEDNESDAY,
             functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
             functions::WEEK_SATURDAY, functions::ISOWEEK, functions::MONTH,
             functions::QUARTER, functions::YEAR, functions::ISOYEAR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_a_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = date_a.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_a_int;
      OP_REQUIRES_OK(context, ParseInputDate(date_a(i), name(), &date_a_int));
      int32_t date_b_int;
      OP_REQUIRES_OK(context, ParseInputDate(date_b(i), name(), &date_b_int));

      // Compute diff.
      int32_t out;
      OP_REQUIRES_OK(
          context, ToStatus(name(), functions::DiffDates(date_a_int, date_b_int,
                                                         part_enum, &out)));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

class DateTrunc : public OpKernel {
 public:
  explicit DateTrunc(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor
    const Tensor& date_tensor = context->input(0);
    auto date = date_tensor.flat<tstring>();
    // Grab the part tensor
    const Tensor& part_tensor = context->input(1);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::DAY, functions::WEEK, functions::WEEK_MONDAY,
             functions::WEEK_TUESDAY, functions::WEEK_WEDNESDAY,
             functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
             functions::WEEK_SATURDAY, functions::ISOWEEK, functions::MONTH,
             functions::QUARTER, functions::YEAR, functions::ISOYEAR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_in;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_in));

      // Truncate date.
      int32_t date_out;
      OP_REQUIRES_OK(
          context, ToStatus(name(), functions::TruncateDate(date_in, part_enum,
                                                            &date_out)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date_out, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class FormatDate : public OpKernel {
 public:
  explicit FormatDate(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format string tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the date tensor
    const Tensor& date_tensor = context->input(1);
    auto date = date_tensor.flat<tstring>();

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_int;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_int));

      // Format date based on format.
      std::string out;
      OP_REQUIRES_OK(
          context,
          ToStatus(name(), functions::FormatDateToString(
                               format, date_int,
                               {.expand_Q = true, .expand_J = true}, &out)));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class LastDayFromDate : public OpKernel {
 public:
  explicit LastDayFromDate(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor.
    const Tensor& date_tensor = context->input(0);
    auto date = date_tensor.flat<tstring>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(1);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>({
            functions::WEEK,
            functions::WEEK_MONDAY,
            functions::WEEK_TUESDAY,
            functions::WEEK_WEDNESDAY,
            functions::WEEK_THURSDAY,
            functions::WEEK_FRIDAY,
            functions::WEEK_SATURDAY,
            functions::ISOWEEK,
            functions::MONTH,
            functions::QUARTER,
            functions::YEAR,
            functions::ISOYEAR,
        });
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the date tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_value;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_value));

      // Extract LAST_DAY from the datetime value.
      int32_t date_int;
      OP_REQUIRES_OK(context,
                     ToStatus(name(), functions::LastDayOfDate(
                                          date_value, part_enum, &date_int)));

      // Set the output value.
      std::string output_str;
      OP_REQUIRES_OK(context, FormatOutputDate(date_int, name(), &output_str));

      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class ParseDate : public OpKernel {
 public:
  explicit ParseDate(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format string tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the date tensor
    const Tensor& date_tensor = context->input(1);
    auto date = date_tensor.flat<tstring>();

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_in;
      OP_REQUIRES_OK(context,
                     ToStatus(name(), functions::ParseStringToDate(
                                          format, date(i),
                                          /*parse_version2=*/true, &date_in)));

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date_in, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class SafeParseDate : public OpKernel {
 public:
  explicit SafeParseDate(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format string tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the date tensor
    const Tensor& date_tensor = context->input(1);
    auto date = date_tensor.flat<tstring>();

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_in;
      if (!functions::ParseStringToDate(format, date(i),
                                        /*parse_version2=*/true, &date_in)
               .ok()) {
        // Set the NULL-equivalent output value for unsuccessful parsing
        OP_REQUIRES_OK(
            context, ToStatus(name(), functions::ParseStringToDate(
                                          kDateFormatString, kNullDate,
                                          /*parse_version2=*/true, &date_in)));
      }

      // Format date to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputDate(date_in, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class UnixDate : public OpKernel {
 public:
  explicit UnixDate(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor
    const Tensor& date_string_tensor = context->input(0);
    auto date_string = date_string_tensor.flat<tstring>();

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, date_string_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = date_string.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date;
      OP_REQUIRES_OK(context, ParseInputDate(date_string(i), name(), &date));

      // Convert date to days.
      int64_t out;
      out = static_cast<int64_t>(date);

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

// Register the kernels
REGISTER_KERNEL_BUILDER(Name("ExtractFromDate").Device(DEVICE_CPU),
                        ExtractFromDate);
REGISTER_KERNEL_BUILDER(Name("DateFromComponents").Device(DEVICE_CPU),
                        DateFromComponents);
REGISTER_KERNEL_BUILDER(Name("DateFromTimestamp").Device(DEVICE_CPU),
                        DateFromTimestamp);
REGISTER_KERNEL_BUILDER(Name("DateFromDatetime").Device(DEVICE_CPU),
                        DateFromDatetime);
REGISTER_KERNEL_BUILDER(Name("CastToDateFromString").Device(DEVICE_CPU),
                        CastToDateFromString);
REGISTER_KERNEL_BUILDER(Name("DateFromUnixDate").Device(DEVICE_CPU),
                        DateFromUnixDate);
REGISTER_KERNEL_BUILDER(Name("DateAdd").Device(DEVICE_CPU), DateAdd);
REGISTER_KERNEL_BUILDER(Name("DateSub").Device(DEVICE_CPU), DateSub);
REGISTER_KERNEL_BUILDER(Name("DateDiff").Device(DEVICE_CPU), DateDiff);
REGISTER_KERNEL_BUILDER(Name("DateTrunc").Device(DEVICE_CPU), DateTrunc);
REGISTER_KERNEL_BUILDER(Name("FormatDate").Device(DEVICE_CPU), FormatDate);
REGISTER_KERNEL_BUILDER(Name("LastDayFromDate").Device(DEVICE_CPU),
                        LastDayFromDate);
REGISTER_KERNEL_BUILDER(Name("ParseDate").Device(DEVICE_CPU), ParseDate);
REGISTER_KERNEL_BUILDER(Name("SafeParseDate").Device(DEVICE_CPU),
                        SafeParseDate);
REGISTER_KERNEL_BUILDER(Name("UnixDate").Device(DEVICE_CPU), UnixDate);

}  // namespace bigquery_ml_utils
