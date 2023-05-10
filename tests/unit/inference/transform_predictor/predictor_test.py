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

"""Tests for bqml transform predictor."""
import sys
from typing import Any

from absl import flags
from absl.testing import absltest
from bigquery_ml_utils.inference import transform_predictor
import tensorflow as tf

# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 9):  # `importlib.resources.files` was added in 3.9
  import importlib.resources as importlib_resources
else:
  import importlib_resources

FLAGS = flags.FLAGS

# Constant for float comparison.
ALMOST_EQUAL_DELTA = 1e-5


class PredictorTest(absltest.TestCase):
  """Test class for bigquery_ml_utils.inference.transform_predictor."""

  def _validate_prediction_results(self, actual_result: list[Any],
                                   expected_result: list[Any]) -> None:
    """Validates the prediction results.

    Args:
      actual_result: List[dict[str, list]], list[list[float]] or list[float].
      expected_result: Almost equal to the actual result within the given
        tolerance.
    """
    self.assertLen(actual_result, len(expected_result))
    for actual_pred, expected_pred in zip(actual_result, expected_result):
      if isinstance(actual_pred, dict):
        self.assertLen(actual_pred, len(expected_pred))
        for key, value in actual_pred.items():
          if any(isinstance(i, float) for i in value):
            self.assertSequenceAlmostEqual(
                value, expected_pred[key], delta=ALMOST_EQUAL_DELTA)
          else:
            self.assertEqual(value, expected_pred[key])
      elif isinstance(actual_pred, list):
        self.assertSequenceAlmostEqual(
            actual_pred, expected_pred, delta=ALMOST_EQUAL_DELTA)
      else:
        self.assertAlmostEqual(
            actual_pred, expected_pred, delta=ALMOST_EQUAL_DELTA)

  def test_linear_reg(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/linear_reg_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    self._validate_prediction_results(
        test_predictor.predict([{
            'f1': 1,
            'f2': 2.5,
            'f3': 'aaa',
        }, {
            'f1': 2,
            'f2': 3.5,
            'f3': 'bbb',
        }]), [[10.310742], [11.064549]])

  def test_linear_reg_custom_op(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/'
            'linear_reg_model_custom_op'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    self._validate_prediction_results(
        test_predictor.predict([
            {
                'f1': '2023-01-10 12:34:56.7 +1234',
                'f2': '2023-03-14 23:45:12.3 +1234',
            },
            {
                'f1': '2023-01-10 15:34:56.7 +1234',
                'f2': '2023-03-14 21:45:12.3 +1234',
            },
        ]),
        [[0.20000000000000004], [0.20000000000000004]],
    )

  def test_boosted_tree_classifier(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/boosted_tree_classifier_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    self._validate_prediction_results(
        test_predictor.predict([{
            'f1': 1,
            'f2': 2.5,
            'f3': 'aaa',
            'f4': ['aaa', 'noise1'],
        }, {
            'f1': 2,
            'f2': 3.5,
            'f3': 'bbb',
            'f4': ['ccc', 'noise2'],
        }]), [{
            'predicted_label_cls':
                'aaa',
            'label_cls_values': ['eee', 'ddd', 'ccc', 'bbb', 'aaa'],
            'label_cls_probs':
                [0.0522024, 0.0521628, 0.0521625, 0.0521627, 0.7913096]
        }, {
            'predicted_label_cls':
                'ccc',
            'label_cls_values': ['eee', 'ddd', 'ccc', 'bbb', 'aaa'],
            'label_cls_probs':
                [0.0522012, 0.0521616, 0.7913146, 0.0521615, 0.0521612]
        }])

  def test_dnn_classifier(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/dnn_classifier_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    self._validate_prediction_results(
        test_predictor.predict([{
            'f1': 1,
            'f2': 2.5,
            'f3': 'aaa',
            'f4': ['aaa', 'noise1'],
        }, {
            'f1': 2,
            'f2': 3.5,
            'f3': 'bbb',
            'f4': ['ccc', 'noise2'],
        }]), [{
            'class_ids': [4],
            'all_class_ids': [0, 1, 2, 3, 4],
            'all_classes': ['eee', 'ddd', 'ccc', 'bbb', 'aaa'],
            'probabilities': [0.000693, 0.004691, 0.003229, 0.006940, 0.984446],
            'logits': [-4.728594, -2.81652, -3.189965, -2.4248600, 2.529902],
            'classes': ['aaa']
        }, {
            'class_ids': [3],
            'all_class_ids': [0, 1, 2, 3, 4],
            'all_classes': ['eee', 'ddd', 'ccc', 'bbb', 'aaa'],
            'probabilities': [0.000223, 0.000475, 0.001424, 0.996830, 0.001048],
            'logits': [-4.663398, -3.908592, -2.810063, 3.7412872, -3.116024],
            'classes': ['bbb']
        }])

  def test_dnn_classifier_different_signature(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/'
            'dnn_classifier_model_predict_signature'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    self._validate_prediction_results(
        test_predictor.predict([
            {
                'f1': -4.581,
                'f2': 30.11,
                'f3': 'eee',
            },
            {
                'f1': -0.402,
                'f2': 5.87,
                'f3': 'aaa',
            },
        ]),
        [
            {
                'classes': ['eee'],
                'all_class_ids': [0, 1],
                'probabilities': [0.5004999041557312, 0.499500036239624],
                'logistic': [0.4995000660419464],
                'class_ids': [0],
                'all_classes': ['eee', 'aaa'],
                'logits': [-0.001999711152166128],
            },
            {
                'logistic': [0.7926387786865234],
                'classes': ['aaa'],
                'logits': [1.3409056663513184],
                'all_classes': ['eee', 'aaa'],
                'probabilities': [0.2073611617088318, 0.7926388382911682],
                'class_ids': [1],
                'all_class_ids': [0, 1],
            },
        ],
    )

  def test_missing_input_feature(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/linear_reg_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    with self.assertRaisesRegex(TypeError, 'missing.*required argument.*f1'):
      test_predictor.predict([{
          'f2': 2.5,
          'f3': 'a',
      }])

  def test_extra_input_feature(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/linear_reg_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    with self.assertRaisesRegex(
        ValueError, '"extra_feature" in not an input of the TRANSFORM.'):
      test_predictor.predict([{'extra_feature': 1}])

  def test_wrong_input_feature(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/linear_reg_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    with self.assertRaises(TypeError):
      test_predictor.predict([{'f1': 1.5}])

  def test_wrong_input_dimension(self):
    model_path = str(
        importlib_resources.files('bigquery_ml_utils').joinpath(
            'tests/data/inference/transform_predictor/linear_reg_model'
        )
    )
    test_predictor = transform_predictor.Predictor.from_path(model_path)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Ranks of all input tensors should match'):
      test_predictor.predict([{
          'f1': [1, 2],
          'f2': 2.5,
          'f3': 'a',
      }])


if __name__ == '__main__':
  absltest.main()
