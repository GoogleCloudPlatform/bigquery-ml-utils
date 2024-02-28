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

// Register DatetimeFromComponents op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeFromComponents")
    .Input("year: int64")
    .Input("month: int64")
    .Input("day: int64")
    .Input("hour: int64")
    .Input("minute: int64")
    .Input("second: int64")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DatetimeFromDate op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeFromDate")
    .Input("date: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DatetimeFromDateAndTime op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeFromDateAndTime")
    .Input("date: string")
    .Input("time: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DatetimeFromTimestamp op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeFromTimestamp")
    .Input("timestamp: string")
    .Input("time_zone: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register CastToDatetimeFromString op with signature.
// Output has the same shape of the datetime_string.
REGISTER_OP("CastToDatetimeFromString")
    .Input("datetime_string: string")
    .Input("format_string: string")
    .Input("with_format: bool")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

// Register DatetimeAdd op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeAdd")
    .Input("datetime: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DatetimeDiff op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeDiff")
    .Input("datetime_a: string")
    .Input("datetime_b: string")
    .Input("part: string")
    .Output("output: int64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DatetimeSub op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeSub")
    .Input("datetime: string")
    .Input("interval: int64")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register DatetimeTrunc op with signature.
// Output has the same shape of the inputs.
REGISTER_OP("DatetimeTrunc")
    .Input("datetime: string")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register ExtractFromDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("ExtractFromDatetime")
    .Input("datetime: string")
    .Input("part: string")
    .Output("part_out: int64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register ExtractDateFromDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("ExtractDateFromDatetime")
    .Input("datetime: string")
    .Output("part_out: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register ExtractTimeFromDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("ExtractTimeFromDatetime")
    .Input("datetime: string")
    .Output("part_out: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register LastDay op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("LastDayFromDatetime")
    .Input("datetime: string")
    .Input("part: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::OkStatus();
    });

// Register FormatDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("FormatDatetime")
    .Input("format_string: string")
    .Input("datetime: string")
    .Output("output: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return ::tensorflow::OkStatus();
    });

// Register ParseDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("ParseDatetime")
    .Input("format_string: string")
    .Input("datetime_string: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return tensorflow::OkStatus();
    });

// Register SafeParseDatetime op with signature.
// Output has the same shape of the input datetime.
REGISTER_OP("SafeParseDatetime")
    .Input("format_string: string")
    .Input("datetime_string: string")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return tensorflow::OkStatus();
    });

}  // namespace bigquery_ml_utils
