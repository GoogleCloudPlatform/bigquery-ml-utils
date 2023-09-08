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

"""Tests for third_party.py.bqml_xgboost_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import flags
from absl.testing import absltest
from bigquery_ml_utils.inference import xgboost_predictor
import mock
import numpy as np
import xgboost as xgb

# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 9):  # `importlib.resources.files` was added in 3.9
  import importlib.resources as importlib_resources
else:
  import importlib_resources

FLAGS = flags.FLAGS


class PredictorTest(absltest.TestCase):

  def test_boosted_tree_regressor(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/boosted_tree_regressor_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 3,
        'f2': ['a']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 0
    }])
    self.assertSequenceAlmostEqual([1.0370053052902222, 1.9364699125289917],
                                   predict_output)

  def test_boosted_tree_classifier(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/boosted_tree_classifier_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 3,
        'f2': ['a']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 0
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': ''
    }])
    self.assertEqual('2', predict_output[0]['predicted_label'])
    self.assertEqual('2', predict_output[1]['predicted_label'])
    self.assertEqual(['3', '2', '1'], predict_output[0]['label_values'])
    self.assertSequenceAlmostEqual(
        [0.23010218143463135, 0.5752021670341492, 0.1946956366300583],
        predict_output[0]['label_probs'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[1]['label_probs'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[2]['label_probs'])

  def test_random_forest_regressor(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/random_forest_regressor_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'a',
        'f3': 6,
        'f2': 'a'
    }, {
        'f1': 'b',
        'f2': 'b',
        'f3': 0
    }])
    self.assertSequenceAlmostEqual([0.974166214466095, 1.5916662216186523],
                                   predict_output)

  def test_random_forest_classifier(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/random_forest_classifier_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'a',
        'f3': 6,
        'f2': 'a'
    }, {
        'f1': 'b',
        'f2': 'b',
        'f3': 0
    }])
    self.assertEqual('3', predict_output[0]['predicted_label'])
    self.assertEqual('3', predict_output[1]['predicted_label'])
    self.assertEqual(['3', '2', '1'], predict_output[0]['label_values'])
    self.assertSequenceAlmostEqual(
        [0.3333333432674408, 0.3333333432674408, 0.3333333432674408],
        predict_output[0]['label_probs'])
    self.assertSequenceAlmostEqual(
        [0.3333333432674408, 0.3333333432674408, 0.3333333432674408],
        predict_output[1]['label_probs'])

  def test_target_encode(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/target_encode_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 'c',
        'f2': ['a']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 'a'
    }])
    self.assertEqual('2', predict_output[0]['predicted_label'])
    self.assertEqual('2', predict_output[1]['predicted_label'])
    self.assertEqual(['3', '2', '1'], predict_output[0]['label_values'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[0]['label_probs'])
    self.assertSequenceAlmostEqual(
        [0.19618307054042816, 0.47606906294822693, 0.3277478516101837],
        predict_output[1]['label_probs'])

  @mock.patch('bigquery_ml_utils.inference.xgboost_predictor.predictor.xgb')
  def test_target_encode_encoded_input(self, mock_xgb):
    mock_xgb.DMatrix.return_value = xgb.DMatrix(
        np.array([[1.0], [1.0]]), missing=None)
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/target_encode_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    _ = test_predictor.predict([{
        'f1': 'b',
        'f3': 'c',
        'f2': ['d']
    }, {
        'f1': 'f',
        'f2': ['c', 'a', 'a', 'f'],
        'f3': 'a'
    }, {
        'f1': 'f',
        'f2': ['d'],
        'f3': 'a'
    }, {
        'f1': 'd',
        'f2': ['d'],
        'f3': 'a'
    }])
    encoded = [[0.3, 0.7, 1.0, None, 3.0], [None, None, 0.45, 0.3, 1.0],
               [None, None, 1.0, None, 1.0], [1.0, None, 1.0, None, 1.0]]
    np.testing.assert_array_equal(
        np.array(encoded, dtype=float), mock_xgb.DMatrix.call_args[0][0])

  def test_sparse_feature_model(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/sparse_feature_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f3': 3,
        'f2': [(1, 1.0)]
    }, {
        'f1': 'f',
        'f2': [(1, 1.0), (2, 1.0), (3, 1.0)],
        'f3': 0
    }])
    self.assertSequenceAlmostEqual([1.0370053052902222, 1.9364699125289917],
                                   predict_output)

  def test_sparse_feature_model_2(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/sparse_feature_model_2'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 'b',
        'f2': [(1, 1.0), (4, 3.0)]
    }, {
        'f1': 'f',
        'f2': [(1, 1.0), (2, 1.0), (3, 1.0)],
    }])
    self.assertSequenceAlmostEqual([1.0370053052902222, 1.9364699125289917],
                                   predict_output)

  def test_sparse_feature_model_3(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/sparse_feature_model_3'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([{
        'f1': 1.0,
        'f2': 'a',
        'f3': ['a', 'b'],
        'f4': [(1, 1.0), (3, 2.0)]
    }, {
        'f1': 2.0,
        'f2': 'b',
        'f3': ['a', 'c'],
        'f4': [(1, 2.0)]
    }])
    self.assertEqual('1', predict_output[0]['predicted_label'])
    self.assertSequenceAlmostEqual([0.11146856844425201, 0.8885313868522644],
                                   predict_output[0]['label_probs'])
    self.assertEqual('1', predict_output[1]['predicted_label'])
    self.assertSequenceAlmostEqual([0.23098896443843842, 0.7690110802650452],
                                   predict_output[1]['label_probs'])

  @mock.patch('bigquery_ml_utils.inference.xgboost_predictor.predictor.xgb')
  def test_sparse_feature_model_3_encoded_input(self, mock_xgb):
    mock_xgb.DMatrix.return_value = xgb.DMatrix(
        np.array([[1.0], [1.0]]), missing=None)
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/sparse_feature_model_3'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    _ = test_predictor.predict([{
        'f1': 1.0,
        'f2': 'a',
        'f3': ['a', 'b'],
        'f4': [(1, 1.0), (3, 2.0)]
    }, {
        'f1': 2.0,
        'f2': 'b',
        'f3': ['a', 'c'],
        'f4': [(1, 2.0)]
    }])
    encoded = [[
        1.0, 1.0, None, 1.0, 1.0, None, None, None, None, 1.0, None, 2.0
    ], [2.0, 2.0, None, 1.0, None, 1.0, None, None, None, 2.0, None, None]]
    np.testing.assert_array_equal(
        np.array(encoded, dtype=float), mock_xgb.DMatrix.call_args[0][0])

  def test_boosted_tree_regressor_mixed_features(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/'
            'boosted_tree_mixed_feature_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([
        {
            'f1': 'a',
            'f2_foo': 'k',
            'f2_bar': 'v',
            'f3': [(1, 1.0), (3, 1.0)],
            'f4': [1.0, 2.0, 3.0],
        },
        {
            'f1': 'b',
            'f2_foo': 'p',
            'f2_bar': 'q',
            'f3': [(2, 3.0), (4, 6.0)],
            'f4': [3.0, 4.0, 5.0],
        },
    ])
    self.assertEqual(0.9626126885414124, predict_output[0])
    self.assertEqual(1.872970700263977, predict_output[1])

  @mock.patch('bigquery_ml_utils.inference.xgboost_predictor.predictor.xgb')
  def test_boosted_tree_regressor_mixed_features_encoded_input(self, mock_xgb):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/'
            'boosted_tree_mixed_feature_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    _ = test_predictor.predict([
        {
            'f1': 'a',
            'f2_foo': 'k',
            'f2_bar': 'v',
            'f3': [(0, 11.0), (3, 12.0)],
            'f4': [10, 20, 30],
        },
        {
            'f1': 'b',
            'f2_foo': 'p',
            'f2_bar': 'q',
            'f3': [(1, 13.0), (4, 13.0)],
            'f4': [15, 25, 35],
        },
    ])
    encoded = [
        [1.0, 1.0, 2.0, 11.0, None, None, 12.0, None, 10.0, 20.0, 30.0],
        [2.0, 2.0, 1.0, None, 13.0, None, None, 13.0, 15.0, 25.0, 35.0],
    ]
    np.testing.assert_array_equal(
        np.array(encoded, dtype=float), mock_xgb.DMatrix.call_args[0][0]
    )

  def test_boosted_tree_cox(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/boosted_tree_cox_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([
        {
            'product_data_storage': 2048,
            'product_travel_expense': 'Free-Trial',
            'product_payroll': 'Active',
            'product_accounting': 'No',
            'csat_score': 9,
            'articles_viewed': 4,
            'smartphone_notifications_viewed': 0,
            'marketing_emails_clicked': 14,
            'social_media_ads_viewed': 1,
            'minutes_customer_support': 8.3,
            'company_size': '10-50',
            'us_region': 'West North Central',
        },
        {
            'product_data_storage': 2048,
            'product_travel_expense': 'Free-Trial',
            'product_payroll': 'Free-Trial',
            'product_accounting': 'Active',
            'csat_score': 9,
            'articles_viewed': 4,
            'smartphone_notifications_viewed': 2,
            'marketing_emails_clicked': 12,
            'social_media_ads_viewed': 1,
            'minutes_customer_support': 0.0,
            'company_size': '100-250',
            'us_region': 'South Atlantic',
        },
    ])
    self.assertSequenceAlmostEqual([1.3199291229248047, 1.4520472288131714],
                                   predict_output)

  def test_boosted_tree_aft(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/xgboost_predictor/boosted_tree_aft_model'
        )
    )
    test_predictor = xgboost_predictor.Predictor.from_path(model_path)
    predict_output = test_predictor.predict([
        {
            'product_data_storage': 2048,
            'product_travel_expense': 'Free-Trial',
            'product_payroll': 'Active',
            'product_accounting': 'No',
            'csat_score': 9,
            'articles_viewed': 4,
            'smartphone_notifications_viewed': 0,
            'marketing_emails_clicked': 14,
            'social_media_ads_viewed': 1,
            'minutes_customer_support': 8.3,
            'company_size': '10-50',
            'us_region': 'West North Central',
        },
        {
            'product_data_storage': 2048,
            'product_travel_expense': 'Free-Trial',
            'product_payroll': 'Free-Trial',
            'product_accounting': 'Active',
            'csat_score': 9,
            'articles_viewed': 4,
            'smartphone_notifications_viewed': 2,
            'marketing_emails_clicked': 12,
            'social_media_ads_viewed': 1,
            'minutes_customer_support': 0.0,
            'company_size': '100-250',
            'us_region': 'South Atlantic',
        },
    ])
    self.assertSequenceAlmostEqual([3.8022050857543945, 4.4091620445251465],
                                   predict_output)

if __name__ == '__main__':
  absltest.main()
