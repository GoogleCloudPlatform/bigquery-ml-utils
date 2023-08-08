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
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "sql_utils/public/functions/cast_date_time.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "sql_utils/public/functions/parse_date_time.h"
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
using ::tensorflow::errors::InvalidArgument;

namespace bigquery_ml_utils {

class TimeFromComponents : public OpKernel {
 public:
  explicit TimeFromComponents(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the hour tensor
    const Tensor& hour_tensor = context->input(0);
    auto hour = hour_tensor.flat<int64_t>();
    // Grab the minute tensor
    const Tensor& minute_tensor = context->input(1);
    auto minute = minute_tensor.flat<int64_t>();
    // Grab the second tensor
    const Tensor& second_tensor = context->input(2);
    auto second = second_tensor.flat<int64_t>();

    OP_REQUIRES(
        context, hour.size() == minute.size() && hour.size() == second.size(),
        InvalidArgument(absl::Substitute("Errors in $0: Inputs must have the "
                                         "same shape, but are: $1, $2, $3",
                                         name(), hour.size(), minute.size(),
                                         second.size())));

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, hour_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = hour.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time;
      OP_REQUIRES_OK(context, ToTslStatus(name(), functions::ConstructTime(
                                                      hour(i), minute(i),
                                                      second(i), &time)));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimeFromTimestamp : public OpKernel {
 public:
  explicit TimeFromTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time zone tensor
    const Tensor& time_zone_tensor = context->input(1);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);

    // Create an output tensor with the shape of the time tensor
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

      // Extract time from timestamp.
      TimeValue time;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(), functions::ConvertTimestampToTime(
                                  absl::FromUnixMicros(ts), time_zone, &time)));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimeFromDatetime : public OpKernel {
 public:
  explicit TimeFromDatetime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor
    const Tensor& datetime_tensor = context->input(0);
    auto datetime = datetime_tensor.flat<tstring>();

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue dt;
      OP_REQUIRES_OK(context, ParseInputDatetime(datetime(i), name(), &dt));

      // Extract time from datetime.
      TimeValue time;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(), functions::ExtractTimeFromDatetime(dt, &time)));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class CastToTimeFromString : public OpKernel {
 public:
  explicit CastToTimeFromString(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the time_string tensor
    const Tensor& time_string_tensor = context->input(0);
    auto time_string = time_string_tensor.flat<tstring>();
    // Grab the format tensor
    const Tensor& format_tensor = context->input(1);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the with_format tensor
    const Tensor& with_format_tensor = context->input(2);
    bool with_format = with_format_tensor.flat<bool>()(0);

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, time_string_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time_string.size();
    for (int i = 0; i < N; i++) {
      // Convert string to time.
      TimeValue time;
      if (with_format) {
        // Convert string with format
        OP_REQUIRES_OK(
            context, ToTslStatus(name(), functions::CastStringToTime(
                                             format, time_string(i),
                                             functions::kMicroseconds, &time)));
      } else {
        // Convert string without format
        OP_REQUIRES_OK(
            context, ToTslStatus(name(), functions::ConvertStringToTime(
                                             time_string(i),
                                             functions::kMicroseconds, &time)));
      }
      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

::tsl::Status TimeAddOperator(TimeValue& time, int64_t interval,
                              functions::DateTimestampPart& time_part,
                              absl::string_view function_name, TimeValue* out) {
  return ToTslStatus(function_name,
                     functions::AddTime(time, time_part, interval, out));
}

class TimeAdd : public OpKernel {
 public:
  explicit TimeAdd(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the time tensor
    const Tensor& time_tensor = context->input(0);
    auto time = time_tensor.flat<tstring>();
    // Grab the interval tensor
    const Tensor& diff_tensor = context->input(1);
    auto interval_int = diff_tensor.flat<int64_t>();
    OP_REQUIRES(context, time.size() == interval_int.size(),
                InvalidArgument(absl::Substitute(
                    "Error in $0: time and interval must have the same shape, "
                    "but are $1, $2",
                    name(), time.size(), interval_int.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, time_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time_value;
      OP_REQUIRES_OK(context, ParseInputTime(time(i), name(), &time_value));

      // Extract time from datetime.
      TimeValue out_time;
      OP_REQUIRES_OK(context, TimeAddOperator(time_value, interval_int(i),
                                              part_enum, name(), &out_time));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(out_time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimeSub : public OpKernel {
 public:
  explicit TimeSub(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the time tensor
    const Tensor& time_tensor = context->input(0);
    auto time = time_tensor.flat<tstring>();
    // Grab the interval tensor
    const Tensor& diff_tensor = context->input(1);
    auto interval_int = diff_tensor.flat<int64_t>();
    OP_REQUIRES(context, interval_int.size() == time.size(),
                InvalidArgument(absl::Substitute(
                    "Error in $0: time and interval must have the same shape, "
                    "but are $1, $2",
                    name(), time.size(), interval_int.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, time_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time_value;
      OP_REQUIRES_OK(context, ParseInputTime(time(i), name(), &time_value));

      // Extract time from datetime.
      TimeValue out_time;
      OP_REQUIRES_OK(context, TimeAddOperator(time_value, -interval_int(i),
                                              part_enum, name(), &out_time));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(out_time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class TimeDiff : public OpKernel {
 public:
  explicit TimeDiff(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the time_a tensor
    const Tensor& time_a_tensor = context->input(0);
    auto time_a = time_a_tensor.flat<tstring>();
    // Grab the time_b tensor
    const Tensor& time_b_tensor = context->input(1);
    auto time_b = time_b_tensor.flat<tstring>();
    OP_REQUIRES(context, time_a.size() == time_b.size(),
                InvalidArgument(absl::Substitute(
                    "Error in $0: time_a and time_b must have the same shape, "
                    "but are $1, $2",
                    name(), time_a.size(), time_b.size())));
    // Grab the part tensor
    const Tensor& part_tensor = context->input(2);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, time_a_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = time_a.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time_a_value;
      OP_REQUIRES_OK(context, ParseInputTime(time_a(i), name(), &time_a_value));
      TimeValue time_b_value;
      OP_REQUIRES_OK(context, ParseInputTime(time_b(i), name(), &time_b_value));

      // Compute diff.
      int64_t out;
      OP_REQUIRES_OK(
          context,
          ToTslStatus(name(), functions::DiffTimes(time_a_value, time_b_value,
                                                   part_enum, &out)));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

class TimeTrunc : public OpKernel {
 public:
  explicit TimeTrunc(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the time tensor
    const Tensor& time_tensor = context->input(0);
    auto time = time_tensor.flat<tstring>();
    // Grab the part tensor
    const Tensor& part_tensor = context->input(1);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, time_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time_value;
      OP_REQUIRES_OK(context, ParseInputTime(time(i), name(), &time_value));

      // Extract time from datetime.
      TimeValue out_time;
      OP_REQUIRES_OK(
          context, ToTslStatus(name(), functions::TruncateTime(
                                           time_value, part_enum, &out_time)));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(out_time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class ExtractFromTime : public OpKernel {
 public:
  explicit ExtractFromTime(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the time tensor
    const Tensor& time_tensor = context->input(0);
    auto time = time_tensor.flat<tstring>();
    // Grab the part tensor
    const Tensor& part_tensor = context->input(1);
    std::string part = part_tensor.flat<tstring>()(0);
    functions::DateTimestampPart part_enum;
    static auto* supported_parts =
        new absl::flat_hash_set<functions::DateTimestampPart>(
            {functions::MICROSECOND, functions::MILLISECOND, functions::SECOND,
             functions::MINUTE, functions::HOUR});
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, *supported_parts));

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, time_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = time.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time_value;
      OP_REQUIRES_OK(context, ParseInputTime(time(i), name(), &time_value));

      // Extract time from datetime.
      int32_t out;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::ExtractFromTime(
                                             part_enum, time_value, &out)));

      // Set the output value.
      // Currently, BQML util inference only supports int64.
      output_flat(i) = static_cast<int64_t>(out);
    }
  }
};

class ParseTime : public OpKernel {
 public:
  explicit ParseTime(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the time tensor
    const Tensor& time_string_tensor = context->input(1);
    auto time_string = time_string_tensor.flat<tstring>();

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, time_string_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time_string.size();
    for (int i = 0; i < N; i++) {
      // Parse time.
      TimeValue out_time;
      OP_REQUIRES_OK(context, ToTslStatus(name(), functions::ParseStringToTime(
                                                      format, time_string(i),
                                                      functions::kMicroseconds,
                                                      &out_time)));

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(out_time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class SafeParseTime : public OpKernel {
 public:
  explicit SafeParseTime(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the time tensor
    const Tensor& time_string_tensor = context->input(1);
    auto time_string = time_string_tensor.flat<tstring>();

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, time_string_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time_string.size();
    for (int i = 0; i < N; i++) {
      // Parse time.
      TimeValue out_time;
      if (!functions::ParseStringToTime(format, time_string(i),
                                        functions::kMicroseconds, &out_time)
               .ok()) {
        // Set the NULL-equivalent output value for unsuccessful parsing.
        OP_REQUIRES_OK(
            context,
            ToTslStatus(name(), functions::ParseStringToTime(
                                    kTimeFormatString, kNullTime,
                                    functions::kMicroseconds, &out_time)));
      }

      // Format time to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTime(out_time, name(), &out));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

class FormatTime : public OpKernel {
 public:
  explicit FormatTime(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format tensor
    const Tensor& format_tensor = context->input(0);
    std::string format = format_tensor.flat<tstring>()(0);
    // Grab the time tensor
    const Tensor& time_string_tensor = context->input(1);
    auto time = time_string_tensor.flat<tstring>();

    // Create an output tensor with the shape of the time tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, time_string_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = time.size();
    for (int i = 0; i < N; i++) {
      // Parse the time.
      TimeValue time_value;
      OP_REQUIRES_OK(context, ParseInputTime(time(i), name(), &time_value));

      // Format time.
      std::string out;
      OP_REQUIRES_OK(context,
                     ToTslStatus(name(), functions::FormatTimeToString(
                                             format, time_value, &out)));

      // Set the output value.
      output_flat(i).reserve(out.size());
      output_flat(i) = std::move(out);
    }
  }
};

// Register the kernels.
REGISTER_KERNEL_BUILDER(Name("TimeFromComponents").Device(DEVICE_CPU),
                        TimeFromComponents);
REGISTER_KERNEL_BUILDER(Name("TimeFromTimestamp").Device(DEVICE_CPU),
                        TimeFromTimestamp);
REGISTER_KERNEL_BUILDER(Name("TimeFromDatetime").Device(DEVICE_CPU),
                        TimeFromDatetime);
REGISTER_KERNEL_BUILDER(Name("CastToTimeFromString").Device(DEVICE_CPU),
                        CastToTimeFromString);
REGISTER_KERNEL_BUILDER(Name("TimeAdd").Device(DEVICE_CPU), TimeAdd);
REGISTER_KERNEL_BUILDER(Name("TimeSub").Device(DEVICE_CPU), TimeSub);
REGISTER_KERNEL_BUILDER(Name("TimeDiff").Device(DEVICE_CPU), TimeDiff);
REGISTER_KERNEL_BUILDER(Name("TimeTrunc").Device(DEVICE_CPU), TimeTrunc);
REGISTER_KERNEL_BUILDER(Name("ExtractFromTime").Device(DEVICE_CPU),
                        ExtractFromTime);
REGISTER_KERNEL_BUILDER(Name("ParseTime").Device(DEVICE_CPU), ParseTime);
REGISTER_KERNEL_BUILDER(Name("SafeParseTime").Device(DEVICE_CPU),
                        SafeParseTime);
REGISTER_KERNEL_BUILDER(Name("FormatTime").Device(DEVICE_CPU), FormatTime);

}  // namespace bigquery_ml_utils
