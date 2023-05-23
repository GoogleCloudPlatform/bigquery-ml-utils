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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace bigquery_ml_utils {

// NOTE: changing signature will break the existing SavedModel.

// Register ExtractFromTimestamp op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("ExtractFromTimestamp")
    .Input("part: string")
    .Input("timestamp: string")
    .Input("time_zone: string")
    .Output("part_out: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

// Register StringFromTimestamp op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("StringFromTimestamp")
    .Input("timestamp: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampFromString op with signature.
// Output has the same shape of the input string.
REGISTER_OP("TimestampFromString")
    .Input("timestamp_string: string")
    .Input("time_zone: string")
    .Input("allow_tz_in_str: bool")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampFromDate op with signature.
// Output has the same shape of the input date.
REGISTER_OP("TimestampFromDate")
    .Input("date: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampFromDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("TimestampFromDatetime")
    .Input("datetime: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampAdd op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampAdd")
    .Input("timestamp: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampSub op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampSub")
    .Input("timestamp: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampDiff op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampDiff")
    .Input("timestamp_a: string")
    .Input("timestamp_b: string")
    .Input("part: string")
    .Output("output: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampTrunc op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampTrunc")
    .Input("timestamp: string")
    .Input("part: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register FormatTimestamp op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("FormatTimestamp")
    .Input("format_string: string")
    .Input("timestamp: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register ParseTimestamp op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("ParseTimestamp")
    .Input("format_string: string")
    .Input("timestamp_string: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

// Register TimestampMicros op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampMicros")
    .Input("timestamp_micro: int64")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampMillis op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampMillis")
    .Input("timestamp_milli: int64")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimestampSeconds op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("TimestampSeconds")
    .Input("timestamp_sec: int64")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register UnixMicros op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("UnixMicros")
    .Input("timestamp: string")
    .Output("output: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register UnixMillis op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("UnixMillis")
    .Input("timestamp: string")
    .Output("output: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register UnixSeconds op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("UnixSeconds")
    .Input("timestamp: string")
    .Output("output: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

}  // namespace bigquery_ml_utils
