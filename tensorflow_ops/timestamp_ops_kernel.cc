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

#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/parse_date_time.h"
#include "tensorflow_ops/constants.h"
#include "tensorflow/core/framework/op_kernel.h"

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::tstring;

namespace bigquery_ml_utils {

class ExtractFromTimestamp : public OpKernel {
 public:
  explicit ExtractFromTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the part tensor
    const Tensor& part_tensor = context->input(0);
    std::string part = absl::AsciiStrToLower(part_tensor.flat<tstring>()(0));
    // Grab the timestamp tensor
    const Tensor& timestamp_tensor = context->input(1);
    auto timestamp = timestamp_tensor.flat<tstring>();
    // Grab the time_zone tensor
    const Tensor& time_zone_tensor = context->input(2);
    std::string time_zone = time_zone_tensor.flat<tstring>()(0);

    // Create an output tensor with the shape of the timestamp tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int32_t>();

    // Parse and validate the timezone.
    absl::TimeZone tz;
    absl::Status status =
        bigquery_ml_utils::functions::MakeTimeZone(time_zone, &tz);
    OP_REQUIRES(context, status.ok(),
                ::tensorflow::errors::InvalidArgument(
                    "Invalid timezone in ExtractFromTimestamp: ", time_zone));

    const int N = timestamp.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t ts;
      status = bigquery_ml_utils::functions::ParseStringToTimestamp(
          kTimestampFormatString, timestamp(i), tz,
          /*parse_version2=*/true, &ts);
      OP_REQUIRES(
          context, status.ok(),
          ::tensorflow::errors::InvalidArgument(
              "Invalid timestamp in ExtractFromTimestamp: ", timestamp(i)));

      // Extract part from the timestamp.
      int32_t out;
      status = bigquery_ml_utils::functions::ExtractFromTimestamp(
          static_cast<bigquery_ml_utils::functions::DateTimestampPart>(
              bigquery_ml_utils::functions::DateTimestampPart_FromName(part)),
          ts, bigquery_ml_utils::functions::kMicroseconds, tz, &out);
      OP_REQUIRES(context, status.ok(),
                  ::tensorflow::errors::InvalidArgument(
                      "Invalid part in ExtractFromTimestamp: ", part));

      // Set the output value.
      output_flat(i) = out;
    }
  }
};

// Register the kernel.
REGISTER_KERNEL_BUILDER(Name("ExtractFromTimestamp").Device(DEVICE_CPU),
                        ExtractFromTimestamp);

}  // namespace bigquery_ml_utils
