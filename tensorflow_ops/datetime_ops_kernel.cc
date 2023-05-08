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
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "sql_utils/public/civil_time.h"
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "sql_utils/public/functions/parse_date_time.h"
#include "tensorflow_ops/constants.h"
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
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

namespace bigquery_ml_utils {

class ExtractFromDatetime : public OpKernel {
 public:
  explicit ExtractFromDatetime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor.
    const Tensor& datetime_tensor = context->input(0);
    auto datetime = datetime_tensor.flat<tstring>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(1);
    std::string part = part_tensor.flat<tstring>()(0);
    int part_int = functions::DateTimestampPart_FromName(part);
    bool valid_part = part_int != -1;
    functions::DateTimestampPart part_enum;
    if (valid_part) {
      part_enum = static_cast<functions::DateTimestampPart>(part_int);
      static auto* kSupportedPart =
          new absl::flat_hash_set<functions::DateTimestampPart>(
              {functions::MICROSECOND,   functions::MILLISECOND,
               functions::SECOND,        functions::MINUTE,
               functions::HOUR,          functions::DAY,
               functions::DAYOFWEEK,     functions::DAYOFYEAR,
               functions::WEEK,          functions::WEEK_MONDAY,
               functions::WEEK_TUESDAY,  functions::WEEK_WEDNESDAY,
               functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
               functions::WEEK_SATURDAY, functions::ISOWEEK,
               functions::MONTH,         functions::QUARTER,
               functions::YEAR,          functions::ISOYEAR});
      valid_part = kSupportedPart->contains(part_enum);
    }
    OP_REQUIRES(context, valid_part,
                InvalidArgument("Invalid part in ExtractFromDatetime: ", part));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, datetime(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument("Invalid datetime in ExtractFromDatetime: ",
                                  datetime(i)));

      // Extract part from the datetime.
      int32_t out;
      status = functions::ExtractFromDatetime(part_enum, datetime_value, &out);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument(
              "InvalidArgument in ExtractFromDatetime with status: ", status));

      // Set the output value.
      output_flat(i) = static_cast<int64_t>(out);
    }
  }
};

class ExtractDateFromDatetime : public OpKernel {
 public:
  explicit ExtractDateFromDatetime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor.
    const Tensor& datetime_tensor = context->input(0);
    auto datetime = datetime_tensor.flat<tstring>();

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, datetime(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument("Invalid datetime in ExtractDateFromDatetime: ",
                          datetime(i)));

      // Extract DATE from the datetime.
      int32_t out;
      status =
          functions::ExtractFromDatetime(functions::DATE, datetime_value, &out);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument(
              "InvalidArgument in ExtractDateFromDatetime with status: ",
              status));

      std::string output_str;
      status = functions::ConvertDateToString(out, &output_str);
      OP_REQUIRES(context, status.ok(),
                  Internal("Internal error in ConvertDateToString with value ",
                           out, " and status: ", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = output_str;
    }
  }
};

class ExtractTimeFromDatetime : public OpKernel {
 public:
  explicit ExtractTimeFromDatetime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor.
    const Tensor& datetime_tensor = context->input(0);
    auto datetime = datetime_tensor.flat<tstring>();

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, datetime(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument("Invalid datetime in ExtractTimeFromDatetime: ",
                          datetime(i)));

      // Extract TIME from the datetime value.
      TimeValue time_value;
      status = functions::ExtractTimeFromDatetime(datetime_value, &time_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument(
                      "InvalidArgument in ExtractTimeFromDatetime with status ",
                      status));

      std::string output_str;
      status = functions::ConvertTimeToString(
          time_value, functions::kMicroseconds, &output_str);
      OP_REQUIRES(context, status.ok(),
                  Internal("Internal error in ConvertTimeToString with status ",
                           status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = output_str;
    }
  }
};

// Register the kernels
REGISTER_KERNEL_BUILDER(Name("ExtractFromDatetime").Device(DEVICE_CPU),
                        ExtractFromDatetime);
REGISTER_KERNEL_BUILDER(Name("ExtractDateFromDatetime").Device(DEVICE_CPU),
                        ExtractDateFromDatetime);
REGISTER_KERNEL_BUILDER(Name("ExtractTimeFromDatetime").Device(DEVICE_CPU),
                        ExtractTimeFromDatetime);

}  // namespace bigquery_ml_utils
