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
#include "tensorflow/core/lib/core/status.h"

namespace bigquery_ml_utils {

// NOTE: changing signature will break the existing SavedModel.

// Register ExtractFromDate op with signature.
// Output has the same shape of the input date.
REGISTER_OP("ExtractFromDate")
    .Input("date: string")
    .Input("part: string")
    .Output("part_out: int64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateFromComponents op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DateFromComponents")
    .Input("year: int64")
    .Input("month: int64")
    .Input("day: int64")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateFromTimestamp op with signature.
// Output has the same shape of the input timestamp.
REGISTER_OP("DateFromTimestamp")
    .Input("timestamp: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateFromDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("DateFromDatetime")
    .Input("datetime: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateAdd op with signature.
// Output has the same shape of the input date.
REGISTER_OP("DateAdd")
    .Input("date: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateSub op with signature.
// Output has the same shape of the input date.
REGISTER_OP("DateSub")
    .Input("date: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateDiff op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DateDiff")
    .Input("date_a: string")
    .Input("date_b: string")
    .Input("part: string")
    .Output("output: int64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DateTrunc op with signature.
// Output has the same shape of the input date.
REGISTER_OP("DateTrunc")
    .Input("date: string")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register FormatDate op with signature.
// Output has the same shape of the input date.
REGISTER_OP("FormatDate")
    .Input("format_string: string")
    .Input("date: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

// Register ParseDate op with signature.
// Output has the same shape of the input date_string.
REGISTER_OP("ParseDate")
    .Input("format_string: string")
    .Input("date_string: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

// Register SafeParseDate op with signature.
// Output has the same shape of the input date_string.
REGISTER_OP("SafeParseDate")
    .Input("format_string: string")
    .Input("date_string: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

}  // namespace bigquery_ml_utils
