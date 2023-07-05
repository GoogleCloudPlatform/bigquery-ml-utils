# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file is to serve BQML model trained with TRANSFORM."""

import collections
import os
from typing import Any, Optional

from bigquery_ml_utils.inference.xgboost_predictor import predictor as bqml_xgboost_predictor
import tensorflow as tf
from bigquery_ml_utils.tensorflow_ops.load_module import load_module

gen_date_ops = load_module("_date_ops.so")
gen_datetime_ops = load_module("_datetime_ops.so")
gen_time_ops = load_module("_time_ops.so")
gen_timestamp_ops = load_module("_timestamp_ops.so")


class Predictor:
  """Class to feed input data into BQML model trained with TRANSFORM.

  It performs both preprocessing and postprocessing on the input and output.
  """

  def __init__(
      self, transform_savedmodel, model_tensorflow=None, model_xgboost=None
  ):
    """Initializes a Predictor to serve BQML models trained with TRANSFORM.

    Args:
      transform_savedmodel: SavedModel pb of the TRANSFORM.
      model_tensorflow: BQML model in tensorflow savedmodel format.
      model_xgboost: BQML model in booster format.

    Returns:
      A 'Predictor' instance.
    """
    self._transform_savedmodel = transform_savedmodel
    self._model_tensorflow = model_tensorflow
    self._model_xgboost = model_xgboost
    # Number of input in the Predict call.
    self._num_input = 0

  def _get_transform_result(self, raw_input):
    """Gets the TRANSFORM result from the raw input data.

    Args:
      raw_input: Raw input.

    Returns:
      TRANSFORM result.
    """
    input_dict = collections.defaultdict(list)
    for row in raw_input:
      for key, value in row.items():
        input_dict[key].append(value)

    infer = self._transform_savedmodel.signatures['serving_default']
    input_signature = infer.structured_input_signature[1]
    transform_input = dict()
    for key, value in input_dict.items():
      if key not in input_signature:
        raise ValueError(f'"{key}" in not an input of the TRANSFORM.')
      transform_input[key] = tf.constant(value, input_signature[key].dtype)
    return infer(**transform_input)

  def _convert_transform_result(self, transform_result):
    """Converts the TRANSFORM result to a list.

    Args:
      transform_result: TRANSFORM result from the raw input.

    Returns:
      TRANSFORM results in a list. Each element in the list is the
      TRANSFORM result of each raw input instance.
    """
    if len(transform_result) == 1:
      for value in transform_result.values():
        return value.numpy().tolist()

    # Initialize the output as a list of dict.
    output = [{} for _ in range(self._num_input)]

    # Convert the transform_result in batch representation to a list, in which
    # each element is the transform result of each raw input instance.
    for key, value in transform_result.items():
      if value.dtype == tf.string:
        batch_result = value.numpy().astype(str).tolist()
      else:
        batch_result = value.numpy().tolist()

      for i, value in enumerate(batch_result):
        if isinstance(value, list):
          output[i][key] = value.copy()
        else:
          output[i][key] = value
    return output

  def _get_tf_model_result(self, transform_result):
    """Gets the model result from the TRANSFORM result for tensorflow models.

    Args:
      transform_result: TRANSFORM result from the raw input.

    Returns:
      Model prediction results in a list. Each element in the list is the
      prediction result of each raw input instance.
    """
    model_input = dict()
    for key, value in transform_result.items():
      model_input[key] = value
      # BQML trained model uses float64 as the input dtype for all numerical
      # features.
      if value.dtype == tf.int64:
        model_input[key] = tf.cast(value, tf.float64)

    infer = None
    if 'predict' in self._model_tensorflow.signatures:
      # By default, BQML DNN model is exported with 'predict' signature.
      infer = self._model_tensorflow.signatures['predict']
    else:
      infer = self._model_tensorflow.signatures['serving_default']
    inference_result = infer(**model_input)

    # If the inference_result of the model contains only one named tensor, we
    # omit the name. This aligns with the TF serving response.
    if len(inference_result) == 1:
      for value in inference_result.values():
        # TODO(b/253233131): Support array<struct> as the output.
        return value.numpy().tolist()

    # Initialize the output as a list of dict.
    output = [{} for i in range(self._num_input)]

    # Convert the inference_result in batch representation to a list, in which
    # each element is the prediction result of each raw input instance.
    for key, value in inference_result.items():
      # TODO(b/253233131): Support array<struct> as the output.
      batch_result = list()
      # The numpy() of string tensor has type BYTE, which needs to be converted
      # to string.
      if value.dtype == tf.string:
        batch_result = value.numpy().astype(str).tolist()
      else:
        batch_result = value.numpy().tolist()

      for i, value in enumerate(batch_result):
        if isinstance(value, list):
          output[i][key] = value.copy()
        else:
          output[i][key] = value
    return output

  def _get_xgboost_model_result(self, transform_result):
    """Gets the model result from the TRANSFORM result for xgboost models.

    Args:
      transform_result: TRANSFORM result from the raw input.

    Returns:
      Model prediction results in a list. Each element in the list is the
      prediction result of each raw input instance.
    """
    xgb_model_input = [{} for i in range(self._num_input)]

    # Convert the transform_result in batch representation to a list, in which
    # each element is the prediction result of each raw input instance.
    for key, value in transform_result.items():
      # TODO(b/253233131): Support array<struct> as the output.
      batch_result = list()
      # The numpy() of string tensor has type BYTE, which needs to be converted
      # to string.
      if value.dtype == tf.string:
        batch_result = value.numpy().astype(str).tolist()
      else:
        batch_result = value.numpy().tolist()

      for i, value in enumerate(batch_result):
        if isinstance(value, list):
          xgb_model_input[i][key] = value.copy()
        else:
          xgb_model_input[i][key] = value
    return self._model_xgboost.predict(xgb_model_input)

  def _get_model_result(self, transform_result):
    """Gets the model result from the TRANSFORM result.

    Args:
      transform_result: TRANSFORM result from the raw input.

    Returns:
      Model prediction results in a list. Each element in the list is the
      prediction result of each raw input instance.
    """
    if self._model_tensorflow is not None:
      return self._get_tf_model_result(transform_result)
    if self._model_xgboost is not None:
      return self._get_xgboost_model_result(transform_result)
    return self._convert_transform_result(transform_result)

  def predict(self, raw_input, **kwargs):
    """Performs prediction.

    Args:
      raw_input: A list of prediction input instances.
      **kwargs: A dictionary of keyword args provided as additional fields on
        the predict request body.

    Returns:
      A dict containing the prediction results.
    """
    del kwargs
    self._num_input = len(raw_input)
    return self._get_model_result(self._get_transform_result(raw_input))

  @classmethod
  def from_path(cls, model_dir):
    """Creates an instance of Predictor using the given path.

    Args:
      model_dir: The local directory that contains the BQML model trained with
        TRANSFORM.

    Returns:
      An instance of 'Predictor'.
    """
    if not tf.io.gfile.exists(os.path.join(model_dir, 'transform')):
      raise ValueError('TRANSFORM subdirectory is not found in the given path.')
    transform_savedmodel = tf.saved_model.load(
        os.path.join(model_dir, 'transform')
    )

    if tf.io.gfile.exists(os.path.join(model_dir, 'saved_model.pb')):
      model_tensorflow = tf.saved_model.load(model_dir)
      return cls(transform_savedmodel, model_tensorflow=model_tensorflow)

    if tf.io.gfile.exists(os.path.join(model_dir, 'model.bst')):
      model_xgboost = bqml_xgboost_predictor.Predictor.from_path(model_dir)
      return cls(transform_savedmodel, model_xgboost=model_xgboost)

    return cls(transform_savedmodel)
