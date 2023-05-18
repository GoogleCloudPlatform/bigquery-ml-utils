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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "tensorflow_ops/utils.h"
#include "tensorflow/core/framework/op_kernel.h"

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
    absl::flat_hash_set<functions::DateTimestampPart> supported_parts = {
        functions::MICROSECOND,   functions::MILLISECOND,
        functions::SECOND,        functions::MINUTE,
        functions::HOUR,          functions::DAYOFWEEK,
        functions::DAY,           functions::DAYOFYEAR,
        functions::WEEK,          functions::WEEK_MONDAY,
        functions::WEEK_TUESDAY,  functions::WEEK_WEDNESDAY,
        functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
        functions::WEEK_SATURDAY, functions::ISOWEEK,
        functions::MONTH,         functions::QUARTER,
        functions::YEAR,          functions::ISOYEAR};
    functions::DateTimestampPart part_enum;
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, supported_parts));
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(1);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);
    absl::TimeZone tz;
    OP_REQUIRES_OK(context, ParseInputTimeZone(time_zone, name(), &tz));

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
      absl::Status status = functions::ExtractFromTimestamp(
          part_enum, ts, functions::kMicroseconds, tz, &out);
      OP_REQUIRES(context, status.ok(),
                  Internal(absl::Substitute(
                      "Error in ExtractFromTimestamp of $0 with status: $1",
                      name(), status.ToString())));

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
    OP_REQUIRES_OK(context, ParseInputTimeZone(time_zone, name(), &tz));

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      OP_REQUIRES_OK(context,
                     ParseInputTimestamp(timestamp(i), tz, name(), &ts));

      // Convert timestamp to string.
      std::string out;
      absl::Status status =
          functions::ConvertTimestampMicrosToStringWithTruncation(ts, tz, &out);
      OP_REQUIRES(context, status.ok(),
                  Internal(absl::Substitute("Error in $0 with status: $1",
                                            name(), status.ToString())));

      // Set the output value.
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
    OP_REQUIRES_OK(context, ParseInputTimeZone(time_zone, name(), &tz));

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      absl::Status status = functions::ConvertStringToTimestamp(
          timestamp(i), tz, functions::kMicroseconds, allow_tz_in_str, &ts);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument(absl::Substitute("Invalid timestamp in $0: $1",
                                           name(), timestamp(i).c_str())));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
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
    OP_REQUIRES_OK(context, ParseInputTimeZone(time_zone, name(), &tz));

    const int N = date.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_int;
      OP_REQUIRES_OK(context, ParseInputDate(date(i), name(), &date_int));

      int64_t ts;
      absl::Status status = functions::ConvertDateToTimestamp(
          date_int, functions::kMicroseconds, tz, &ts);
      OP_REQUIRES(context, status.ok(),
                  Internal(absl::Substitute(
                      "Error in ConvertDateToTimestamp of $0 with status: $1",
                      name(), status.ToString())));

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
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
    OP_REQUIRES_OK(context, ParseInputTimeZone(time_zone, name(), &tz));

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue dt;
      OP_REQUIRES_OK(context, ParseInputDatetime(datetime(i), name(), &dt));

      absl::Time base_time;
      absl::Status status = functions::ConvertDatetimeToTimestamp(
          DatetimeValue::FromPacked64Micros(dt.Packed64DatetimeMicros()), tz,
          &base_time);
      OP_REQUIRES(
          context, status.ok(),
          Internal(absl::Substitute(
              "Error in ConvertDatetimeToTimestamp of $0 with status: $1",
              name(), status.ToString())));
      int64_t ts = absl::ToUnixMicros(base_time);

      // Format timestamp to string.
      std::string out;
      OP_REQUIRES_OK(context, FormatOutputTimestamp(ts, name(), &out));

      // Set the output value.
      output_flat(i) = std::move(out);
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

}  // namespace bigquery_ml_utils
