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
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
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

class DatetimeFromComponents : public OpKernel {
 public:
  explicit DatetimeFromComponents(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    bool valid_input = true;
    // Grab the year tensor.
    const Tensor& year_tensor = context->input(0);
    auto years = year_tensor.flat<int64_t>();

    // Grab the month tensor.
    const Tensor& month_tensor = context->input(1);
    auto months = month_tensor.flat<int64_t>();
    valid_input = valid_input && (years.size() == months.size());

    // Grab the day tensor.
    const Tensor& day_tensor = context->input(2);
    auto days = day_tensor.flat<int64_t>();
    valid_input = valid_input && (years.size() == days.size());

    // Grab the hour tensor.
    const Tensor& hour_tensor = context->input(3);
    auto hours = hour_tensor.flat<int64_t>();
    valid_input = valid_input && (years.size() == hours.size());

    // Grab the minute tensor.
    const Tensor& minute_tensor = context->input(4);
    auto minutes = minute_tensor.flat<int64_t>();
    valid_input = valid_input && (years.size() == minutes.size());

    // Grab the second tensor.
    const Tensor& second_tensor = context->input(5);
    auto seconds = second_tensor.flat<int64_t>();
    valid_input = valid_input && (years.size() == seconds.size());

    OP_REQUIRES(context, valid_input,
                InvalidArgument("Invalid input in DatetimeFromComponents: all "
                                "the inputs must have the same shape."));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, year_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = years.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status =
          functions::ConstructDatetime(years(i), months(i), days(i), hours(i),
                                       minutes(i), seconds(i), &datetime_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument(absl::Substitute(
                      "Errors in DatetimeFromComponents with input '$0', '$1', "
                      "'$2', '$3', '$4', '$5': $6",
                      years(i), months(i), days(i), hours(i), minutes(i),
                      seconds(i), status.ToString())));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 datetime_value, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: ", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class DatetimeFromDate : public OpKernel {
 public:
  explicit DatetimeFromDate(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor.
    const Tensor& date_tensor = context->input(0);
    auto dates = date_tensor.flat<tstring>();

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = dates.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_int;
      absl::Status status = functions::ParseStringToDate(
          kDateFormatString, dates(i), /*parse_version2=*/true, &date_int);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument(absl::Substitute(
                      "Invalid date input '$0' in DatetimeFromDate.",
                      dates(i).c_str())));

      // Parse the datetime.
      DatetimeValue datetime_value;
      status =
          functions::ConstructDatetime(date_int, TimeValue(), &datetime_value);
      OP_REQUIRES(context, status.ok(),
                  Internal(absl::Substitute(
                      "Internal error in DatetimeFromDate with input '$0': $1",
                      dates(i).c_str(), status.ToString())));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 datetime_value, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: ", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class DatetimeFromDateAndTime : public OpKernel {
 public:
  explicit DatetimeFromDateAndTime(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the date tensor.
    const Tensor& date_tensor = context->input(0);
    auto dates = date_tensor.flat<tstring>();

    // Grab the time tensor.
    const Tensor& time_tensor = context->input(1);
    auto times = time_tensor.flat<tstring>();

    OP_REQUIRES(
        context, dates.size() == times.size(),
        InvalidArgument(
            "Inputs in DatetimeFromDateAndTime must have the same length."));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, date_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = dates.size();
    for (int i = 0; i < N; i++) {
      // Parse the date.
      int32_t date_int;
      absl::Status status = functions::ParseStringToDate(
          kDateFormatString, dates(i), /*parse_version2=*/true, &date_int);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument(absl::Substitute(
                      "Invalid date input '$0' in DatetimeFromDateAndTime.",
                      dates(i).c_str())));

      // Parse the time.
      TimeValue time_value;
      status = functions::ParseStringToTime(
          kTimeFormatString, times(i), functions::kMicroseconds, &time_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument(absl::Substitute(
                      "Invalid time input '$0' in DatetimeFromDateAndTime.",
                      times(i).c_str())));

      // Construct the datetime.
      DatetimeValue datetime_value;
      status =
          functions::ConstructDatetime(date_int, time_value, &datetime_value);
      OP_REQUIRES(
          context, status.ok(),
          Internal(absl::Substitute(
              "Internal error in DatetimeFromDateAndTime with input '$0': $1",
              dates(i).c_str(), status.ToString())));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 datetime_value, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: ", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class DatetimeFromTimestamp : public OpKernel {
 public:
  explicit DatetimeFromTimestamp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the Timestamp tensor.
    const Tensor& timestamp_tensor = context->input(0);
    auto timestamps = timestamp_tensor.flat<tstring>();

    // Grab the time_zone tensor.
    const Tensor& timezone_tensor = context->input(1);
    absl::string_view timezone_str = timezone_tensor.flat<tstring>()(0);
    // Parse and validate the timezone.
    absl::TimeZone timezone;
    absl::Status status = functions::MakeTimeZone(timezone_str, &timezone);
    OP_REQUIRES(context, status.ok(),
                InvalidArgument("Invalid timezone in DatetimeFromTimestamp: ",
                                timezone_str));

    // Create an output tensor with the shape of the Timestamp tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, timestamp_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = timestamps.size();
    for (int i = 0; i < N; i++) {
      // Parse the timestamp.
      int64_t timestamp_int;
      absl::Status status = functions::ParseStringToTimestamp(
          kTimestampFormatString, timestamps(i), timezone,
          /*parse_version2=*/true, &timestamp_int);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument(absl::Substitute(
                      "Invalid date input '$0' in DatetimeFromTimestamp.",
                      timestamps(i).c_str())));

      // Construct the datetime.
      DatetimeValue datetime_value;
      status = functions::ConvertTimestampToDatetime(
          absl::FromUnixMicros(timestamp_int), timezone, &datetime_value);
      OP_REQUIRES(
          context, status.ok(),
          Internal(absl::Substitute(
              "Internal error in DatetimeFromTimestamp with input '$0': $1",
              timestamps(i).c_str(), status.ToString())));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 datetime_value, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: ", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class DatetimeAdd : public OpKernel {
 public:
  explicit DatetimeAdd(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor.
    const Tensor& datetime_tensor = context->input(0);
    auto input_datetime = datetime_tensor.flat<tstring>();

    // Grab the interval tensor.
    const Tensor& interval_tensor = context->input(1);
    auto input_interval = interval_tensor.flat<int64_t>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(2);
    absl::string_view part = part_tensor.flat<tstring>()(0);
    int part_int = functions::DateTimestampPart_FromName(part);
    bool valid_part = part_int != -1;
    functions::DateTimestampPart part_enum;
    if (valid_part) {
      part_enum = static_cast<functions::DateTimestampPart>(part_int);
      static auto* kSupportedPart =
          new absl::flat_hash_set<functions::DateTimestampPart>(
              {functions::MICROSECOND, functions::MILLISECOND,
               functions::SECOND, functions::MINUTE, functions::HOUR,
               functions::DAY, functions::WEEK, functions::MONTH,
               functions::QUARTER, functions::YEAR});
      valid_part = kSupportedPart->contains(part_enum);
    }
    OP_REQUIRES(context, valid_part,
                InvalidArgument("Invalid part in DatetimeAdd: ", part));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    OP_REQUIRES(context, input_datetime.size() == input_interval.size(),
                InvalidArgument("DatetimeAdd expects the same length of "
                                "datetime and internval inputs."));
    const int N = input_datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, input_datetime(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument("Invalid datetime in DatetimeAdd: ",
                                  input_datetime(i)));

      // Add the part of the internal to the datetime.
      DatetimeValue output_datetime;
      status = functions::AddDatetime(datetime_value, part_enum,
                                      input_interval(i), &output_datetime);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in AddDatetime of DatetimeAdd with status: $0",
                   status));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 output_datetime, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: $0", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class DatetimeDiff : public OpKernel {
 public:
  explicit DatetimeDiff(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime_a tensor.
    const Tensor& datetime_a_tensor = context->input(0);
    auto datetime_a = datetime_a_tensor.flat<tstring>();

    // Grab the datetime_b tensor.
    const Tensor& datetime_b_tensor = context->input(1);
    auto datetime_b = datetime_b_tensor.flat<tstring>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(2);
    absl::string_view part = part_tensor.flat<tstring>()(0);
    int part_int = functions::DateTimestampPart_FromName(part);
    bool valid_part = part_int != -1;
    functions::DateTimestampPart part_enum;
    if (valid_part) {
      part_enum = static_cast<functions::DateTimestampPart>(part_int);
      static auto* kSupportedPart =
          new absl::flat_hash_set<functions::DateTimestampPart>(
              {functions::MICROSECOND, functions::MILLISECOND,
               functions::SECOND, functions::MINUTE, functions::HOUR,
               functions::DAY, functions::WEEK, functions::WEEK_MONDAY,
               functions::WEEK_TUESDAY, functions::WEEK_WEDNESDAY,
               functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
               functions::WEEK_SATURDAY, functions::ISOWEEK, functions::MONTH,
               functions::QUARTER, functions::YEAR, functions::ISOYEAR});
      valid_part = kSupportedPart->contains(part_enum);
    }
    OP_REQUIRES(context, valid_part,
                InvalidArgument("Invalid part in DatetimeDiff: ", part));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, datetime_a_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    OP_REQUIRES(context, datetime_a.size() == datetime_b.size(),
                InvalidArgument("DatetimeDiff expects the same length of "
                                "datetime_a and datetime_b."));
    const int N = datetime_a.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime_a.
      DatetimeValue datetime_a_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, datetime_a(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_a_value);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument("Invalid datetime in DatetimeDiff: ", datetime_a(i)));

      // Parse the datetime_b.
      DatetimeValue datetime_b_value;
      status = functions::ParseStringToDatetime(
          kDatetimeFormatString, datetime_b(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_b_value);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument("Invalid datetime in DatetimeDiff: ", datetime_b(i)));

      // Get the diff of datetime_a and datetime_b in part.
      int64_t output;
      status = functions::DiffDatetimes(datetime_a_value, datetime_b_value,
                                        part_enum, &output);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in DiffDatetimes of DatetimeDiff with status: $0",
                   status));

      // Set the output value.
      output_flat(i) = output;
    }
  }
};

class DatetimeSub : public OpKernel {
 public:
  explicit DatetimeSub(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor.
    const Tensor& datetime_tensor = context->input(0);
    auto input_datetime = datetime_tensor.flat<tstring>();

    // Grab the interval tensor.
    const Tensor& interval_tensor = context->input(1);
    auto input_interval = interval_tensor.flat<int64_t>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(2);
    absl::string_view part = part_tensor.flat<tstring>()(0);
    int part_int = functions::DateTimestampPart_FromName(part);
    bool valid_part = part_int != -1;
    functions::DateTimestampPart part_enum;
    if (valid_part) {
      part_enum = static_cast<functions::DateTimestampPart>(part_int);
      static auto* kSupportedPart =
          new absl::flat_hash_set<functions::DateTimestampPart>(
              {functions::MICROSECOND, functions::MILLISECOND,
               functions::SECOND, functions::MINUTE, functions::HOUR,
               functions::DAY, functions::WEEK, functions::MONTH,
               functions::QUARTER, functions::YEAR});
      valid_part = kSupportedPart->contains(part_enum);
    }
    OP_REQUIRES(context, valid_part,
                InvalidArgument("Invalid part in DatetimeSub: ", part));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    OP_REQUIRES(context, input_datetime.size() == input_interval.size(),
                InvalidArgument("DatetimeSub expects the same length of "
                                "datetime and internval inputs."));
    const int N = input_datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, input_datetime(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument("Invalid datetime in DatetimeSub: ",
                                  input_datetime(i)));

      // Add the part of the internal to the datetime.
      DatetimeValue output_datetime;
      status = functions::SubDatetime(datetime_value, part_enum,
                                      input_interval(i), &output_datetime);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in SubDatetime of DatetimeSub with status: $0",
                   status));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 output_datetime, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: $0", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

class DatetimeTrunc : public OpKernel {
 public:
  explicit DatetimeTrunc(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the datetime tensor.
    const Tensor& datetime_tensor = context->input(0);
    auto input_datetime = datetime_tensor.flat<tstring>();

    // Grab the part tensor.
    const Tensor& part_tensor = context->input(1);
    absl::string_view part = part_tensor.flat<tstring>()(0);
    int part_int = functions::DateTimestampPart_FromName(part);
    bool valid_part = part_int != -1;
    functions::DateTimestampPart part_enum;
    if (valid_part) {
      part_enum = static_cast<functions::DateTimestampPart>(part_int);
      static auto* kSupportedPart =
          new absl::flat_hash_set<functions::DateTimestampPart>(
              {functions::MICROSECOND, functions::MILLISECOND,
               functions::SECOND, functions::MINUTE, functions::HOUR,
               functions::DAY, functions::WEEK, functions::WEEK_MONDAY,
               functions::WEEK_TUESDAY, functions::WEEK_WEDNESDAY,
               functions::WEEK_THURSDAY, functions::WEEK_FRIDAY,
               functions::WEEK_SATURDAY, functions::ISOWEEK, functions::MONTH,
               functions::QUARTER, functions::YEAR, functions::ISOYEAR});
      valid_part = kSupportedPart->contains(part_enum);
    }
    OP_REQUIRES(context, valid_part,
                InvalidArgument("Invalid part in DatetimeTrunc: ", part));

    // Create an output tensor with the shape of the datetime tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, datetime_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = input_datetime.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          kDatetimeFormatString, input_datetime(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(context, status.ok(),
                  InvalidArgument("Invalid datetime in DatetimeTrunc: ",
                                  input_datetime(i)));

      DatetimeValue output_datetime;
      status = functions::TruncateDatetime(datetime_value, part_enum,
                                           &output_datetime);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in SubDatetime of DatetimeTrunc with status: $0",
                   status));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 output_datetime, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: $0", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

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

class ParseDatetime : public OpKernel {
 public:
  explicit ParseDatetime(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the format_string tensor.
    const Tensor& format_string_tensor = context->input(0);
    absl::string_view format_string = format_string_tensor.flat<tstring>()(0);

    // Grab the datetime_string tensor.
    const Tensor& datetime_string_tensor = context->input(1);
    auto datetime_strings = datetime_string_tensor.flat<tstring>();

    // Create an output tensor with the shape of the datetime_string tensor.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, datetime_string_tensor.shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    const int N = datetime_strings.size();
    for (int i = 0; i < N; i++) {
      // Parse the datetime.
      DatetimeValue datetime_value;
      absl::Status status = functions::ParseStringToDatetime(
          format_string, datetime_strings(i), functions::kMicroseconds,
          /*parse_version2=*/true, &datetime_value);
      OP_REQUIRES(
          context, status.ok(),
          InvalidArgument(absl::Substitute(
              "Error in ParseDatetime with format_string '$0' and "
              "datetime_string '$1': $2",
              format_string, datetime_strings(i).c_str(), status.ToString())));

      // Convert output_datetime to string.
      std::string output_str;
      status = functions::FormatDatetimeToString(kDatetimeFormatString,
                                                 datetime_value, &output_str);
      OP_REQUIRES(
          context, status.ok(),
          Internal("Error in FormatDatetimeToString with status: ", status));

      // Set the output value.
      output_flat(i).reserve(output_str.size());
      output_flat(i) = std::move(output_str);
    }
  }
};

// Register the kernels.
REGISTER_KERNEL_BUILDER(Name("DatetimeFromComponents").Device(DEVICE_CPU),
                        DatetimeFromComponents);
REGISTER_KERNEL_BUILDER(Name("DatetimeFromDate").Device(DEVICE_CPU),
                        DatetimeFromDate);
REGISTER_KERNEL_BUILDER(Name("DatetimeFromDateAndTime").Device(DEVICE_CPU),
                        DatetimeFromDateAndTime);
REGISTER_KERNEL_BUILDER(Name("DatetimeFromTimestamp").Device(DEVICE_CPU),
                        DatetimeFromTimestamp);
REGISTER_KERNEL_BUILDER(Name("DatetimeAdd").Device(DEVICE_CPU), DatetimeAdd);
REGISTER_KERNEL_BUILDER(Name("DatetimeDiff").Device(DEVICE_CPU), DatetimeDiff);
REGISTER_KERNEL_BUILDER(Name("DatetimeSub").Device(DEVICE_CPU), DatetimeSub);
REGISTER_KERNEL_BUILDER(Name("DatetimeTrunc").Device(DEVICE_CPU),
                        DatetimeTrunc);
REGISTER_KERNEL_BUILDER(Name("ExtractFromDatetime").Device(DEVICE_CPU),
                        ExtractFromDatetime);
REGISTER_KERNEL_BUILDER(Name("ExtractDateFromDatetime").Device(DEVICE_CPU),
                        ExtractDateFromDatetime);
REGISTER_KERNEL_BUILDER(Name("ExtractTimeFromDatetime").Device(DEVICE_CPU),
                        ExtractTimeFromDatetime);
REGISTER_KERNEL_BUILDER(Name("ParseDatetime").Device(DEVICE_CPU),
                        ParseDatetime);

}  // namespace bigquery_ml_utils
