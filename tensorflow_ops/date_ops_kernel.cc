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
#include "sql_utils/public/functions/date_time_util.h"
#include "sql_utils/public/functions/datetime.pb.h"
#include "tensorflow_ops/utils.h"
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
    absl::flat_hash_set<functions::DateTimestampPart> supported_parts = {
        functions::DAY,
        functions::DAYOFWEEK,
        functions::DAYOFYEAR,
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
        functions::ISOYEAR};
    functions::DateTimestampPart part_enum;
    OP_REQUIRES_OK(context, ParseInputDateTimestampPart(
                                part, name(), &part_enum, supported_parts));

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
      absl::Status status =
          functions::ExtractFromDate(part_enum, date_value, &out);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Internal error in ExtractFromDate with status: ", status));

      // Set the output value.
      output_flat(i) = static_cast<int64_t>(out);
    }
  }
};

// Register the kernels
REGISTER_KERNEL_BUILDER(Name("ExtractFromDate").Device(DEVICE_CPU),
                        ExtractFromDate);

}  // namespace bigquery_ml_utils
