# Copyright 2023 Google LLC
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

"""This file supports preproccesing procedure for text embedding models."""

import enum

import tensorflow as tf
import tensorflow_hub as hub
# Required for registering custom ops used by BERT.
import tensorflow_text as _  # pylint: disable=unused-import


@enum.unique
class TextEmbeddingModelType(str, enum.Enum):
  """The different text embedding model types that require signature addition."""

  NNLM = "nnlm"
  SWIVEL = "swivel"
  BERT = "bert"


@enum.unique
class TextEmbeddingModelLinks(str, enum.Enum):
  """The different text embedding model tensorflow hub links."""

  NNLM = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
  SWIVEL = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
  BERT_PREPROCESS = "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3"
  BERT_ENCODER = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4"


class Keras3HubLayer(tf.keras.layers.Layer):
  """Wrapper for hub.load to be compatible with Keras 3."""

  def __init__(self, handle, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.handle = handle
    self.output_dim = output_dim
    self.model = hub.load(handle)

  def call(self, inputs):
    return self.model(inputs)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

  def get_config(self):
    config = super().get_config()
    config.update({
        "handle": self.handle,
        "output_dim": self.output_dim,
    })
    return config


class Keras3BertWrapper(tf.keras.layers.Layer):
  """Wrapper for BERT preprocess + encoder to be compatible with Keras 3."""

  def __init__(self, pre_link, enc_link, **kwargs):
    super().__init__(**kwargs)
    self.pre_link = pre_link
    self.enc_link = enc_link
    self.pre = hub.load(pre_link)
    self.enc = hub.load(enc_link)

  def call(self, inputs):
    return self.enc(self.pre(inputs))["pooled_output"]

  def compute_output_shape(self, input_shape):
    return (input_shape[0], 768)

  def get_config(self):
    config = super().get_config()
    config.update({
        "pre_link": self.pre_link,
        "enc_link": self.enc_link,
    })
    return config


class TextEmbeddingModelGenerator:
  """Class to chain embedding models by integrating signature addition.

  It performs preprocessing for the NNLM, SWIVEL, and BERT model types.
  """

  def __init__(
      self,
      nnlm_link=TextEmbeddingModelLinks.NNLM,
      swivel_link=TextEmbeddingModelLinks.SWIVEL,
      bert_preprocess_link=TextEmbeddingModelLinks.BERT_PREPROCESS,
      bert_encoder_link=TextEmbeddingModelLinks.BERT_ENCODER,
  ):
    """Initializes a TextEmbedModelGenerator to generate text embed models.

    Args:
      nnlm_link: Path (local or hub) that links to the NNLM model.
      swivel_link: Path (local or hub) that links to the SWIVEL model.
      bert_preprocess_link: Path (local or hub) that links to BERT preprocess
        model.
      bert_encoder_link: Path (local or hub) that links to BERT encoder model.

    Returns:
      A 'Predictor' instance.
    """
    self._nnlm_link = nnlm_link
    self._swivel_link = swivel_link
    self._bert_preprocess_link = bert_preprocess_link
    self._bert_encoder_link = bert_encoder_link

  def generate_text_embedding_model(self, model_type, folder_path):
    """Generate the NNLM model from Tensorflow hub.

    Args:
      model_type: Text embedding model type (NNLM or BERT).
      folder_path: Folder path to save model in.

    Returns:
      Generated model with default signature.
    """
    if model_type.lower() == TextEmbeddingModelType.NNLM:
      model = self._generate_nnlm()
    elif model_type.lower() == TextEmbeddingModelType.SWIVEL:
      model = self._generate_swivel()
    elif model_type.lower() == TextEmbeddingModelType.BERT:
      model = self._generate_bert()
    else:
      raise ValueError(
          f'"{model_type}" is not a valid model type. Please choose one from'
          ' (NNLM, SWIVEL, BERT)'
      )
    tf.saved_model.save(
        model, folder_path, signatures=self._construct_model_signature(model)
    )

  def _generate_nnlm(self) -> tf.keras.Model:
    """Generate the NNLM model from Tensorflow hub.

    Returns:
      Generated NNLM model.
    """
    text_input = tf.keras.layers.Input(
        shape=(), dtype=tf.string, name="content"
    )
    outputs = Keras3HubLayer(self._nnlm_link, output_dim=50)(text_input)
    model = tf.keras.Model(text_input, outputs)
    return model

  def _generate_swivel(self) -> tf.keras.Model:
    """Generate the SWIVEL model from Tensorflow hub.

    Returns:
      Generated SWIVEL model.
    """
    text_input = tf.keras.layers.Input(
        shape=(), dtype=tf.string, name="content"
    )
    outputs = Keras3HubLayer(self._swivel_link, output_dim=20)(text_input)
    model = tf.keras.Model(text_input, outputs)
    return model

  def _generate_bert(self) -> tf.keras.Model:
    """Generate the BERT model from Tensorflow hub.

    Returns:
      Generated BERT model.
    """
    text_input = tf.keras.layers.Input(
        shape=(), dtype=tf.string, name="content"
    )
    outputs = Keras3BertWrapper(
        self._bert_preprocess_link, self._bert_encoder_link
    )(text_input)
    model = tf.keras.Model(text_input, outputs)
    return model

  def _construct_model_signature(self, model) -> []:
    """Constructs model signature in order to override output tensor name.

    Args:
      model: NNLM, SWIVEL, or BERT tf.keras.Model

    Returns:
      Constructed signature.
    """
    @tf.function
    def export_model_wapper(embedding_model, **feature_specs):
      return {"text_embedding": embedding_model(feature_specs)}

    tensor_spec = {
        "content": tf.TensorSpec(shape=(None,), dtype=tf.string, name="content")
    }
    signature = {
        "serving_default": export_model_wapper.get_concrete_function(
            model, **tensor_spec
        )
    }
    return signature

  def _save_model_with_path(self, model, folder_path):
    # Check if directory exists.
    if not tf.io.gfile.isdir(folder_path):
      tf.io.gfile.MkDir(folder_path)
    tf.saved_model.save(model, folder_path)
