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

"""This file is to serve XGBoost model trained in BQML."""

import glob
import json
import os
import re
import numpy as np
import xgboost as xgb


class Predictor(object):
  """Class to feed input data into XGBoost model.

  It performs both preprocessing and postprocessing on the input and output.
  """

  def __init__(
      self,
      model,
      model_metadata,
      categorical_one_hot_vocab,
      categorical_target_vocab,
      categorical_label_vocab,
      array_one_hot_vocab,
      array_target_vocab,
      array_struct_dimension_dict,
      array_numerical_length_dict,
  ):
    """Initializes a Predictor for XGBoost model serving.

    Args:
      model: XGBoost model.
      model_metadata: The metadata of the model.
      categorical_one_hot_vocab: dict for one-hot encode categorical features.
      categorical_target_vocab: dict for target encode categorical features
      categorical_label_vocab: dict for label encode categorical features.
      array_one_hot_vocab: dict for one-hot encode array features.
      array_target_vocab: dict for target encode array features.
      array_struct_dimension_dict: dict for array struct dimensions.
      array_numerical_length_dict: dict for array numerical lengths.

    Returns:
      A 'Predictor' instance.
    """
    self._model = model
    self._model_metadata = model_metadata
    self._categorical_one_hot_vocab = categorical_one_hot_vocab
    self._categorical_target_vocab = categorical_target_vocab
    self._categorical_label_vocab = categorical_label_vocab
    self._array_one_hot_vocab = array_one_hot_vocab
    self._array_target_vocab = array_target_vocab
    self._array_struct_dimension_dict = array_struct_dimension_dict
    self._array_numerical_length_dict = array_numerical_length_dict
    self._model_type = None
    self._label_col = None
    self._feature_name_to_index_map = {}
    # This is to keep the order of features used in training.
    self._feature_names = []
    self._class_names = []

  def _extract_model_metadata(self):
    """Extracts info from model metadata and fills member variables.

    Raises:
      ValueError: An error occurred when:
        1. Invalid model type.
        2. Label not found.
        3. Features not found.
        4. Class names not found for {boosted_tree|random_forest}_classifier.
        6. Feature index mismatch.
        7. Invalid encode type for categorical features.
    """
    if 'model_type' not in self._model_metadata or self._model_metadata[
        'model_type'] not in [
            'boosted_tree_regressor', 'boosted_tree_classifier',
            'random_forest_regressor', 'random_forest_classifier'
        ]:
      raise ValueError('Invalid model_type in model_metadata')
    self._model_type = self._model_metadata['model_type']
    if 'label_col' not in self._model_metadata:
      raise ValueError('label_col not found in model_metadata')
    self._label_col = self._model_metadata['label_col']
    if not self._model_metadata['features']:
      raise ValueError('No feature found in model_metadata')
    self._feature_names = self._model_metadata['feature_names']
    if self._model_type in ['boosted_tree_classifier',
                            'random_forest_classifier']:
      if 'class_names' not in self._model_metadata or not self._model_metadata[
          'class_names']:
        raise ValueError('No class_names found in model_metadata')
      self._class_names = self._model_metadata['class_names']
    for feature_index in range(len(self._feature_names)):
      feature_name = self._feature_names[feature_index]
      self._feature_name_to_index_map[feature_name] = feature_index
      feature_metadata = self._model_metadata['features'][feature_name]
      if ('encode_type' not in feature_metadata) or (not feature_metadata[
          'encode_type']) or (feature_metadata[
              'encode_type'] == 'numerical_identity'):
        continue
      elif feature_metadata['encode_type'] == 'categorical_one_hot':
        if feature_index not in self._categorical_one_hot_vocab:
          raise ValueError(
              'feature_index %d missing in _categorical_one_hot_vocab' %
              feature_index)
      elif feature_metadata['encode_type'] == 'categorical_target':
        if feature_index not in self._categorical_target_vocab:
          raise ValueError(
              'feature_index %d missing in _categorical_target_vocab' %
              feature_index)
      elif (feature_metadata[
          'encode_type'] == 'categorical_label') or (feature_metadata[
              'encode_type'] == 'ohe'):
        if feature_index not in self._categorical_label_vocab:
          raise ValueError(
              'feature_index %d missing in _categorical_label_vocab' %
              feature_index)
      elif (feature_metadata[
          'encode_type'] == 'array_one_hot') or (feature_metadata[
              'encode_type'] == 'mhe'):
        if feature_index not in self._array_one_hot_vocab:
          raise ValueError('feature_index %d missing in _array_one_hot_vocab' %
                           feature_index)
      elif feature_metadata['encode_type'] == 'array_target':
        if feature_index not in self._array_target_vocab:
          raise ValueError('feature_index %d missing in _array_target_vocab' %
                           feature_index)
      elif feature_metadata['encode_type'] == 'array_struct':
        if (self._array_struct_dimension_dict and
            feature_index not in self._array_struct_dimension_dict):
          raise ValueError(
              'feature_index %d missing in _array_struct_dimension_dict' %
              feature_index)
      elif feature_metadata['encode_type'] == 'array_numerical':
        if (
            self._array_numerical_length_dict
            and feature_index not in self._array_numerical_length_dict
        ):
          raise ValueError(
              'feature_index %d missing in _array_numerical_length_dict'
              % feature_index
          )
      else:
        raise ValueError('Invalid encode_type %s for feature %s' %
                         (feature_metadata['encode_type'], feature_name))

  def _preprocess(self, data):
    """Preprocesses raw input data for prediction.

    Args:
      data: Raw input in 2d array.

    Returns:
      Preprocessed data in 2d array.

    Raises:
      ValueError: An error occurred when features in a data row are different
      from the features in the model.
    """
    self._extract_model_metadata()
    preprocessed_data = []
    for row_index in range(len(data)):
      row = data[row_index]
      sorted_data_feature_names = sorted(row.keys())
      sorted_model_feature_names = sorted(self._feature_names)
      if sorted_data_feature_names != sorted_model_feature_names:
        raise ValueError(
            'Row %d has different features %s than the model features %s' %
            (row_index, ','.join(sorted_data_feature_names),
             ','.join(sorted_model_feature_names)))
      encoded_row = []
      for feature_name in self._feature_names:
        col = row[feature_name]
        feature_index = self._feature_name_to_index_map[feature_name]
        if feature_index in self._categorical_one_hot_vocab:
          vocab = self._categorical_one_hot_vocab[feature_index]
          one_hot_list = [None] * len(vocab)
          col_value = str(col)
          if col_value in vocab:
            one_hot_list[vocab.index(col_value)] = 1.0
          encoded_row.extend(one_hot_list)
        elif feature_index in self._categorical_target_vocab:
          vocab = self._categorical_target_vocab[feature_index]
          col_value = str(col)
          # None will be automatically handled by xgboost lib.
          target_list = vocab.get(col_value,
                                  [None] * len(list(vocab.values())[0]))
          # We will treat the zero value as missing value for multi-class
          # models.
          if len(target_list) > 1:
            target_list = [None if x == 0.0 else x for x in target_list]
          encoded_row.extend(target_list)
        elif feature_index in self._categorical_label_vocab:
          vocab = self._categorical_label_vocab[feature_index]
          col_value = str(col)
          if col_value in vocab:
            encoded_row.append(float(vocab.index(col_value)))
          else:
            # unseen category.
            encoded_row.append(None)
        elif feature_index in self._array_one_hot_vocab:
          vocab = self._array_one_hot_vocab[feature_index]
          one_hot_list = [None] * len(vocab)
          try:
            for item in col:
              item_value = str(item)
              if item_value in vocab:
                one_hot_list[vocab.index(item_value)] = 1.0
            encoded_row.extend(one_hot_list)
          except ValueError:
            raise ValueError('The feature %s in row %d is not an array' %
                             (feature_name, row_index))
        elif feature_index in self._array_target_vocab:
          vocab = self._array_target_vocab[feature_index]
          target_list = [0.0] * len(list(vocab.values())[0])
          try:
            for item in col:
              item_value = str(item)
              item_target_list = vocab.get(item_value,
                                           [0.0] * len(list(vocab.values())[0]))
              item_target_list = [x / float(len(col)) for x in item_target_list]
              target_list = [sum(x) for x in zip(target_list, item_target_list)]

            # We will treat the zero value as missing value for multi-class
            # models.
            if len(target_list) > 1:
              target_list = [None if x == 0.0 else x for x in target_list]
            encoded_row.extend(target_list)
          except ValueError:
            raise ValueError('The feature %s in row %d is not an array' %
                             (feature_name, row_index))
        elif (self._array_struct_dimension_dict and
              feature_index in self._array_struct_dimension_dict):
          dimension = self._array_struct_dimension_dict[feature_index]
          array_struct_dense_vector = [None] * dimension

          for item in col:
            key = item[0]
            if key < 0:
              raise ValueError('The key of the sparse feature %s in row %d is '
                               'smaller than 0.' % (feature_name, row_index))
            if key > dimension:
              raise ValueError('The key of the sparse feature %s in row %d is '
                               'larger than the sparse feature dimension %d.' %
                               (feature_name, row_index, dimension))
            array_struct_dense_vector[item[0]] = item[1]
          encoded_row.extend(array_struct_dense_vector)
        elif (
            self._array_numerical_length_dict
            and feature_index in self._array_numerical_length_dict
        ):
          length = self._array_numerical_length_dict[feature_index]
          if len(col) != length:
            raise ValueError(
                'The length of the array numerical feature %s '
                'in row %d does not match '
                'array_numerical_length.txt.' % (feature_name, row_index)
            )
          encoded_row.extend(np.array(col).astype(np.float64))
        else:
          # Numerical feature.
          # Treat empty string as 0 as XAI use empty string as baseline.
          if col == '':
            encoded_row.append(0.0)
          else:
            try:
              encoded_row.append(float(col))
            except ValueError:
              raise ValueError(
                  'The feature %s in row %d cannot be converted to float' %
                  (feature_name, row_index))

      preprocessed_data.append(encoded_row)
    return preprocessed_data

  def predict(self, instances, **kwargs):
    """Performs prediction.

    Args:
      instances: A list of prediction input instances.
      **kwargs: A dictionary of keyword args provided as additional fields on
        the predict request body.

    Returns:
      A list of outputs containing the prediction results.
    """
    del kwargs
    encoded = self._preprocess(instances)
    # We have to convert encoded from list to numpy array, otherwise xgb will
    # take 0s as missing values.
    prediction_input = xgb.DMatrix(
        np.array(encoded, dtype=float).reshape((len(instances), -1)),
        missing=None)
    if self._model_type in ['boosted_tree_classifier',
                            'random_forest_classifier']:
      outputs = self._model.predict(prediction_input)
      final_outputs = []
      for np_output in outputs:
        output = np_output.tolist()
        final_output = {}
        final_output['predicted_{}'.format(
            self._label_col)] = self._class_names[output.index(max(output))]
        final_output['{}_values'.format(self._label_col)] = self._class_names
        final_output['{}_probs'.format(self._label_col)] = output
        final_outputs.append(final_output)
      return final_outputs
    else:
      # Boosted tree or random forest regressor.
      return self._model.predict(prediction_input).tolist()

  @classmethod
  def from_path(cls, model_dir):
    """Creates an instance of Predictor using the given path.

    Args:
      model_dir: The local directory that contains the trained XGBoost model and
        the assets including vocabularies and model metadata.

    Returns:
      An instance of 'Predictor'.
    """
    # Keep model name the same as ml::kXgboostFinalModelFilename.
    model_path = os.path.join(model_dir, 'model.bst')
    model = xgb.Booster(model_file=model_path)
    assets_path = os.path.join(model_dir, 'assets')
    model_metadata_path = os.path.join(assets_path, 'model_metadata.json')
    with open(model_metadata_path) as f:
      model_metadata = json.load(f)
    txt_list = glob.glob(assets_path + '/*.txt')
    categorical_one_hot_vocab = {}
    categorical_target_vocab = {}
    categorical_label_vocab = {}
    array_one_hot_vocab = {}
    array_target_vocab = {}
    array_struct_dimension_dict = {}
    array_numerical_length_dict = {}
    for txt_file in txt_list:
      categorical_one_hot_found = re.search(r'(\d+)_categorical_one_hot.txt',
                                            txt_file)
      categorical_target_found = re.search(r'(\d+)_categorical_target.txt',
                                           txt_file)
      categorical_label_found = re.search(r'(\d+)_categorical_label.txt',
                                          txt_file)
      categorical_label_found_legacy = re.search(r'(\d+).txt', txt_file)
      array_one_hot_found = re.search(r'(\d+)_array_one_hot.txt', txt_file)
      array_one_hot_found_legacy = re.search(r'(\d+)_array.txt', txt_file)
      array_target_found = re.search(r'(\d+)_array_target.txt', txt_file)
      array_struct_found = re.search(r'(\d+)_array_struct_dimension.txt',
                                     txt_file)
      array_numerical_found = re.search(
          r'(\d+)_array_numerical_length.txt', txt_file
      )
      if categorical_one_hot_found:
        feature_index = int(categorical_one_hot_found.group(1))
        with open(txt_file) as f:
          categorical_one_hot_vocab[feature_index] = f.read().splitlines()
      elif categorical_target_found:
        feature_index = int(categorical_target_found.group(1))
        target_dict = {}
        with open(txt_file) as f:
          split_lines = f.read().splitlines()
          for line in split_lines:
            try:
              words = line.split(',')
              target_dict[words[0]] = [float(x) for x in words[1:]]
            except ValueError:
              raise ValueError(
                  '%s does not have the right format for target encoding' %
                  (txt_file))
        categorical_target_vocab[feature_index] = target_dict
      elif categorical_label_found:
        feature_index = int(categorical_label_found.group(1))
        with open(txt_file) as f:
          categorical_label_vocab[feature_index] = f.read().splitlines()
      elif categorical_label_found_legacy:
        feature_index = int(categorical_label_found_legacy.group(1))
        with open(txt_file) as f:
          categorical_label_vocab[feature_index] = f.read().splitlines()
      elif array_one_hot_found:
        feature_index = int(array_one_hot_found.group(1))
        with open(txt_file) as f:
          array_one_hot_vocab[feature_index] = f.read().splitlines()
      elif array_one_hot_found_legacy:
        feature_index = int(array_one_hot_found_legacy.group(1))
        with open(txt_file) as f:
          array_one_hot_vocab[feature_index] = f.read().splitlines()
      elif array_target_found:
        feature_index = int(array_target_found.group(1))
        target_dict = {}
        with open(txt_file) as f:
          split_lines = f.read().splitlines()
          for line in split_lines:
            try:
              words = line.split(',')
              target_dict[words[0]] = [float(x) for x in words[1:]]
            except ValueError:
              raise ValueError(
                  '%s does not have the right format for target encoding' %
                  (txt_file))
        array_target_vocab[feature_index] = target_dict
      elif array_struct_found:
        feature_index = int(array_struct_found.group(1))
        with open(txt_file) as f:
          try:
            dimension = int(f.read().strip())
            array_struct_dimension_dict[feature_index] = dimension
          except ValueError:
            raise ValueError(
                '%s does not have the right format for array struct dimension' %
                (txt_file))
      elif array_numerical_found:
        feature_index = int(array_numerical_found.group(1))
        with open(txt_file) as f:
          try:
            length = int(f.read().strip())
            array_numerical_length_dict[feature_index] = length
          except ValueError:
            raise ValueError(
                '%s does not have the right format for array numerical length'
                % (txt_file)
            )

    return cls(
        model,
        model_metadata,
        categorical_one_hot_vocab,
        categorical_target_vocab,
        categorical_label_vocab,
        array_one_hot_vocab,
        array_target_vocab,
        array_struct_dimension_dict,
        array_numerical_length_dict,
    )
