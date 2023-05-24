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

// Register TimeFromComponents op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("TimeFromComponents")
    .Input("hour: int64")
    .Input("minute: int64")
    .Input("second: int64")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimeFromTimestamp op with signature.
// Output has the same shape of the timestamp.
REGISTER_OP("TimeFromTimestamp")
    .Input("timestamp: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimeFromDatetime op with signature.
// Output has the same shape of the datetime.
REGISTER_OP("TimeFromDatetime")
    .Input("datetime: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimeAdd op with signature.
// Output has the same shape of the time.
REGISTER_OP("TimeAdd")
    .Input("time: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimeSub op with signature.
// Output has the same shape of the time.
REGISTER_OP("TimeSub")
    .Input("time: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimeDiff op with signature.
// Output has the same shape of the times.
REGISTER_OP("TimeDiff")
    .Input("time_a: string")
    .Input("time_b: string")
    .Input("part: string")
    .Output("output: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register TimeTrunc op with signature.
// Output has the same shape of the time.
REGISTER_OP("TimeTrunc")
    .Input("time: string")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register ExtractFromTime op with signature.
// Output has the same shape of the time.
REGISTER_OP("ExtractFromTime")
    .Input("time: string")
    .Input("part: string")
    .Output("output: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register ParseTime op with signature.
// Output has the same shape of the time_string.
REGISTER_OP("ParseTime")
    .Input("format_string: string")
    .Input("time_string: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

// Register FormatTime op with signature.
// Output has the same shape of the time.
REGISTER_OP("FormatTime")
    .Input("format_string: string")
    .Input("time: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

}  // namespace bigquery_ml_utils
