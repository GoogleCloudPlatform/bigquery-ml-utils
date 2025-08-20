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

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "sql_utils/public/functions/arithmetics.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "sql_utils/public/functions/parse_date_time.h"
#include "sql_utils/public/types/timestamp_util.h"
#include "tensorflow_ops/constants.h"
#include "tensorflow_ops/utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::tstring;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

namespace bigquery_ml_utils {

class ExtractFromTimestamp : public OpKernel {
 public:
  explicit ExtractFromTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the part tensor
    const Tensor& part_tensor = context->input(0);
    std::string part = absl::AsciiStrToLower(part_tensor.flat<tstring>()(0));
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND,   functions::MILLISECOND,
             functions::SECOND,        functions::MINUTE,
             functions::HOUR,          functions::DAYOFWEEK,
             functions::DAY,           functions::DAYOFYEAR,
             functions::WEEK,          functions::WEEK_MONDAY,
             functions::WEEK_TUESDAY,  functions::WEEK_WEDNESDAY,
             functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
             functions::WEEK_SATURDAY, functions::ISOWEEK,
             functions::MONTH,         functions::QUARTER,
             functions::YEAR,          functions::ISOYEAR});
    functions::DateTimestampPart part_enum;
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(1);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &ts));

      // Extract part from the timestamp.
      int32_t out;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(),
                      functions::ExtractFromTimestamp(
                          part_enum, ts, functions::kMicroseconds, tz, &out)));

      // Set the output value.
      // Currently, BQML util inference only supports int64.
      output_flat(i) = static_cast<int64_t>(out);
    }
  }
};

class StringFromTimestamp : public OpKernel {
 public:
  explicit StringFromTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(1);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    // Parse and validate the timezone.
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &ts));

      // Convert timestamp to string.
      std::string out;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(),
                      functions::ConvertTimestampMicrosToStringWithTruncation(
                          ts, tz, &out)));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampFromString : public OpKernel {
 public:
  explicit TimestampFromString(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(1);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    // Grab the allow_tz_in_str tensor
    const Tensor& allow_tz_in_str_tensor = context->input(2);
    bool allow_tz_in_str = allow_tz_in_str_tensor.flat<bool>()(0);

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    // Parse and validate the timezone.
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(), functions::ConvertStringToTimestamp(
                                  timestamp(i), tz, functions::kMicroseconds,
                                  allow_tz_in_str, &ts)));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampFromDate : public OpKernel {
 public:
  explicit TimestampFromDate(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor
    const Tensor& date_tensor = context->input(0);
    auto date = date_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(1);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);

    // Create an output tensor with the shape of the date tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    // Parse and validate the timezone.
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_int;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_int));

      int64_t ts;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::ConvertDateToTimestamp(
                                             date_int, functions::kMicroseconds,
                                             tz, &ts)));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampFromDatetime : public OpKernel {
 public:
  explicit TimestampFromDatetime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor
    const Tensor& date_tensor = context->input(0);
    auto datetime = date_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(1);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);

    // Create an output tensor with the shape of the datetime tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    // Parse and validate the timezone.
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue dt;
      OP_REQUIRES_OK(context, ParseInputDatetime(datetime(i), name(), &dt));

      absl::Time base_time;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::ConvertDatetimeToTimestamp(
                                             DatetimeValue::FromPacked64Micros(
                                                 dt.Packed64DatetimeMicros()),
                                             tz, &base_time)));
      int64_t ts = absl::ToUnixMicros(base_time);

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampAdd : public OpKernel {
 public:
  explicit TimestampAdd(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the interval tensor
    const Tensor& diff_tensor = context->input(1);
    auto interval_int = diff_tensor.flat<int64_t>();
    OP_REQUIRES(
        context, interval_int.size() == timestamp.size(),
        InvalidArgument(absl::Substitute(
            "Error in $0: timestamp and interval must have the same shape, "
            "but are $1, $2",
            name(), timestamp.size(), interval_int.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR, functions::DAY});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Default time zone.
      absl::TimeZone tz = absl::UTCTimeZone();

      // Parse the timestamp.
      int64_t input_ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &input_ts));

      absl::StatusOr<IntervalValue> interval =
          GetIntervalValue(interval_int(i), part_enum);
      OP_REQUIRES(
          context, interval.ok(),
          Internal("Error in getting interval of TimestampAdd with status: ",
                   interval.status()));
      absl::Time base_time;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::AddTimestamp(
                                             absl::FromUnixMicros(input_ts), tz,
                                             *interval, &base_time)));
      int64_t ts = absl::ToUnixMicros(base_time);

      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampSub : public OpKernel {
 public:
  explicit TimestampSub(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the interval tensor
    const Tensor& diff_tensor = context->input(1);
    auto interval_int = diff_tensor.flat<int64_t>();
    OP_REQUIRES(
        context, interval_int.size() == timestamp.size(),
        InvalidArgument(absl::Substitute(
            "Error in $0: timestamp and interval must have the same shape, "
            "but are $1, $2",
            name(), timestamp.size(), interval_int.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR, functions::DAY});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Default time zone.
      absl::TimeZone tz = absl::UTCTimeZone();

      // Parse the timestamp.
      int64_t input_ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &input_ts));

      absl::StatusOr<IntervalValue> interval =
          GetIntervalValue(-interval_int(i), part_enum);
      OP_REQUIRES(
          context, interval.ok(),
          Internal("Error in getting interval of TimestampSub with status: ",
                   interval.status()));
      absl::Time base_time;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::AddTimestamp(
                                             absl::FromUnixMicros(input_ts), tz,
                                             *interval, &base_time)));
      int64_t ts = absl::ToUnixMicros(base_time);

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampDiff : public OpKernel {
 public:
  explicit TimestampDiff(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp_a tensor
    const Tensor& timestamp_a_tensor = context->input(0);
    auto timestamp_a = timestamp_a_tensor.flat<tstring>();
    // Grab the timestamp_b tensor
    const Tensor& timestamp_b_tensor = context->input(1);
    auto timestamp_b = timestamp_b_tensor.flat<tstring>();
    OP_REQUIRES(context, timestamp_a.size() == timestamp_b.size(),
                InvalidArgument(
                    "Timestamps in TimestampDiff must have the same length."));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR, functions::DAY});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_a_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = timestamp_a.size();
    for (int i = 0; i < N; i++) {
      // Default time zone.
      absl::TimeZone tz = absl::UTCTimeZone();

      // Parse the timestamp.
      int64_t ts_a;
      int64_t ts_b;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp_a(i), tz, name(), &ts_a));
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp_b(i), tz, name(), &ts_b));

      int64_t out;
      OP_REQUIRES_OK(
          context, ToTslStatus(name(), functions::TimestampDiff(
                                           ts_a, ts_b, functions::kMicroseconds,
                                           part_enum, &out)));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

class TimestampTrunc : public OpKernel {
 public:
  explicit TimestampTrunc(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the part tensor
    const Tensor& part_tensor = context->input(1);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR, functions::DAY,
             functions::WEEK, functions::WEEK_MONDAY, functions::WEEK_TUESDAY,
             functions::WEEK_WEDNESDAY, functions::WEEK_THURSDAY,
             functions::WEEK_FRIDAY, functions::WEEK_SATURDAY,
             functions::ISOWEEK, functions::MONTH, functions::QUARTER,
             functions::YEAR, functions::ISOYEAR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t input_ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &input_ts));

      int64_t out_ts;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::TruncateTimestamp(
                                             input_ts, functions::kMicroseconds,
                                             tz, part_enum, &out_ts)));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(out_ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class FormatTimestamp : public OpKernel {
 public:
  explicit FormatTimestamp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format string tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(1);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &ts));

      // Format the timestamp string.
      functions::FormatDateTimestampOptions format_options = {
          .expand_Q = true,
          .expand_J = true,
      };
      std::string out;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(), functions::FormatTimestampToString(
                                  format, ts, tz, format_options, &out)));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class ParseTimestamp : public OpKernel {
 public:
  explicit ParseTimestamp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format string tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(1);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    absl::TimeZone tz;
    OP_REQUIRES_OK(
        context, ToTslStatus(name(), functions::MakeTimeZone(time_zone, &tz)));

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::ParseStringToTimestamp(
                                             format, timestamp(i), time_zone,
                                             /*parse_version2=*/true, &ts)));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class SafeParseTimestamp : public OpKernel {
 public:
  explicit SafeParseTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format string tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(1);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    absl::TimeZone tz;

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Safe parse the timestamp.
      int64_t ts;
      if (!functions::MakeTimeZone(time_zone, &tz).ok() ||
          !functions::ParseStringToTimestamp(format, timestamp(i), time_zone,
                                             /*parse_version2=*/true, &ts)
               .ok()) {
        // Set the NULL-equivalent output value for unsuccessful parsing.
        OP_REQUIRES_OK(
            context,
            ToTslStatus(name(), functions::ParseStringToTimestamp(
                                    kTimestampFormatString, kNullTimestamp,
                                    absl::UTCTimeZone(),
                                    /*parse_version2=*/true, &ts)));
      }

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

::tsl::Status TimestampFromIntOperator(int64_t in, int64_t scale,
                                       absl::string_view function_name,
                                       int64_t* out) {
  // SECONDS: 1000000, MILLIS: 1000, MICROS: 1
  if (scale != 1000000 && scale != 1000 && scale != 1) {
    return Internal(absl::Substitute("Invalid scale $0 called by $1", scale,
                                     function_name));
  }

  *out = in;
  if (scale != 1 && !functions::Multiply(in, scale, out, /*error=*/nullptr)) {
    // Only the SECONDS and MILLIS versions can overflow due to multiplication,
    // since the MICROS version has a scale of 1. It's possible that the
    // multiplication can succeed but the result is still out of range, in which
    // case we return the error message about the timestamp range.
    return InvalidArgument(absl::Substitute(
        "Timestamp value in $0 overflows: $1", function_name, in));
  }
  if (*out > types::kTimestampMax || *out < types::kTimestampMin) {
    std::string ts_min;
    functions::ConvertTimestampToStringWithoutTruncation(
        types::kTimestampMin, functions::kMicroseconds, absl::UTCTimeZone(),
        &ts_min)
        .IgnoreError();
    std::string ts_max;
    functions::ConvertTimestampToStringWithoutTruncation(
        types::kTimestampMax, functions::kMicroseconds, absl::UTCTimeZone(),
        &ts_max)
        .IgnoreError();
    return InvalidArgument(absl::Substitute(
        "Timestamp value in $0 is out of allowed range: from $1 to $2.",
        function_name, ts_min, ts_max));
  }
  return absl::OkStatus();
}

class TimestampMicros : public OpKernel {
 public:
  explicit TimestampMicros(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_int_tensor = context->input(0);
    auto timestamp_int = timestamp_int_tensor.flat<int64_t>();

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, timestamp_int_tensor.shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp_int.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     TimestampFromIntOperator(timestamp_int(i),
                                              /* scale= */ 1, name(), &ts));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampMillis : public OpKernel {
 public:
  explicit TimestampMillis(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_int_tensor = context->input(0);
    auto timestamp_int = timestamp_int_tensor.flat<int64_t>();

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, timestamp_int_tensor.shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp_int.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     TimestampFromIntOperator(timestamp_int(i),
                                              /* scale= */ 1000, name(), &ts));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimestampSeconds : public OpKernel {
 public:
  explicit TimestampSeconds(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_int_tensor = context->input(0);
    auto timestamp_int = timestamp_int_tensor.flat<int64_t>();

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, timestamp_int_tensor.shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamp_int.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(
          context, TimestampFromIntOperator(timestamp_int(i),
                                            /* scale= */ 1000000, name(), &ts));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

::tsl::Status IntFromTimestampOperator(int64_t in, int64_t scale,
                                       absl::string_view function_name,
                                       int64_t* out) {
  // SECONDS: 1000000, MILLIS: 1000, MICROS: 1
  if (scale != 1000000 && scale != 1000 && scale != 1) {
    return Internal(absl::Substitute("Invalid scale $0 called by $1", scale,
                                     function_name));
  }

  // No overflows possible with division, result truncated downwards;
  *out = static_cast<int64_t>(in / scale);
  if (in < 0 && in % scale != 0) {
    (*out)--;
  }
  return absl::OkStatus();
}

class UnixMicros : public OpKernel {
 public:
  explicit UnixMicros(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(
          context,
          ParseInputTimestamp(timestamp(i), absl::UTCTimeZone(), name(), &ts));

      // Convert timestamp to micros.
      int64_t out;
      OP_REQUIRES_OK(context,
                     IntFromTimestampOperator(ts,
                                              /* scale= */ 1, name(), &out));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

class UnixMillis : public OpKernel {
 public:
  explicit UnixMillis(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(
          context,
          ParseInputTimestamp(timestamp(i), absl::UTCTimeZone(), name(), &ts));

      // Convert timestamp to millis.
      int64_t out;
      OP_REQUIRES_OK(context,
                     IntFromTimestampOperator(ts,
                                              /* scale= */ 1000, name(), &out));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

class UnixSeconds : public OpKernel {
 public:
  explicit UnixSeconds(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(
          context,
          ParseInputTimestamp(timestamp(i), absl::UTCTimeZone(), name(), &ts));

      // Convert timestamp to seconds.
      int64_t out;
      OP_REQUIRES_OK(context, IntFromTimestampOperator(ts,
                                                       /* scale= */ 1000000,
                                                       name(), &out));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

// Register the kernels.
REGISTER_KERNEL_BUILDER(Name("ExtractFromTimestamp").Device(DEVICE_CPU),
                        ExtractFromTimestamp);
REGISTER_KERNEL_BUILDER(Name("StringFromTimestamp").Device(DEVICE_CPU),
                        StringFromTimestamp);
REGISTER_KERNEL_BUILDER(Name("TimestampFromString").Device(DEVICE_CPU),
                        TimestampFromString);
REGISTER_KERNEL_BUILDER(Name("TimestampFromDate").Device(DEVICE_CPU),
                        TimestampFromDate);
REGISTER_KERNEL_BUILDER(Name("TimestampFromDatetime").Device(DEVICE_CPU),
                        TimestampFromDatetime);
REGISTER_KERNEL_BUILDER(Name("TimestampAdd").Device(DEVICE_CPU), TimestampAdd);
REGISTER_KERNEL_BUILDER(Name("TimestampSub").Device(DEVICE_CPU), TimestampSub);
REGISTER_KERNEL_BUILDER(Name("TimestampDiff").Device(DEVICE_CPU),
                        TimestampDiff);
REGISTER_KERNEL_BUILDER(Name("TimestampTrunc").Device(DEVICE_CPU),
                        TimestampTrunc);
REGISTER_KERNEL_BUILDER(Name("FormatTimestamp").Device(DEVICE_CPU),
                        FormatTimestamp);
REGISTER_KERNEL_BUILDER(Name("ParseTimestamp").Device(DEVICE_CPU),
                        ParseTimestamp);
REGISTER_KERNEL_BUILDER(Name("SafeParseTimestamp").Device(DEVICE_CPU),
                        SafeParseTimestamp);
REGISTER_KERNEL_BUILDER(Name("TimestampMicros").Device(DEVICE_CPU),
                        TimestampMicros);
REGISTER_KERNEL_BUILDER(Name("TimestampMillis").Device(DEVICE_CPU),
                        TimestampMillis);
REGISTER_KERNEL_BUILDER(Name("TimestampSeconds").Device(DEVICE_CPU),
                        TimestampSeconds);
REGISTER_KERNEL_BUILDER(Name("UnixMicros").Device(DEVICE_CPU), UnixMicros);
REGISTER_KERNEL_BUILDER(Name("UnixMillis").Device(DEVICE_CPU), UnixMillis);
REGISTER_KERNEL_BUILDER(Name("UnixSeconds").Device(DEVICE_CPU), UnixSeconds);

}  // namespace bigquery_ml_utils
