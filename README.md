# BigQuery ML Utils

[BigQuery ML](https://cloud.google.com/bigquery-ml/docs/introduction) (aka.
BQML) lets you create and execute machine learning models in [BigQuery](https://cloud.google.com/bigquery/docs/introduction)
using standard SQL queries. The BigQuery ML Utils library is an integrated suite
of machine learning tools for building and using BigQuery ML models.


## Installation

Install this library in a [virtualenv](https://virtualenv.pypa.io/en/latest/)
using pip. [virtualenv](https://virtualenv.pypa.io/en/latest/) is a tool to
create isolated Python environments. The basic problem it addresses is one of
dependencies and versions, and indirectly permissions.

With [virtualenv](https://virtualenv.pypa.io/en/latest/), it's possible to
install this library without needing system install permissions, and without
clashing with the installed system
dependencies.

### Mac/Linux

```
    pip install virtualenv
    virtualenv <your-env>
    source <your-env>/bin/activate
    <your-env>/bin/pip install bigquery-ml-utils
```

### Windows

```
    pip install virtualenv
    virtualenv <your-env>
    <your-env>\Scripts\activate
    <your-env>\Scripts\pip.exe install bigquery-ml-utils
```

## Overview

### Inference

#### Transform Predictor

The Transform Predictor feeds input data into the BQML model trained with
TRANSFORM. It performs both preprocessing and postprocessing on the input and
output. The first argument is a [SavedModel](https://www.tensorflow.org/guide/saved_model/)
which represents the [TRANSFORM clause](https://cloud.google.com/bigquery-ml/docs/bigqueryml-transform/)
for feature preprocessing. The second argument is a
[SavedModel](https://www.tensorflow.org/guide/saved_model/) or
[XGBoost Booster](https://xgboost.readthedocs.io/en/latest/) which represents
the model logic.

#### XGBoost Predictor

The XGBoost Predictor feeds input data into the BQML XGBoost model. It performs
both preprocessing and postprocessing on the input and output. The first
argument is a [XGBoost Booster](https://xgboost.readthedocs.io/en/latest/) which
represents the model logic. The following arguments are model assets.

### Tensorflow Ops

BQML Tensorflow Custom Ops provides SQL functions ([Date functions](https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions),
[Datetime functions](https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions),
[Time functions](https://cloud.google.com/bigquery/docs/reference/standard-sql/time_functions)
and [Timestamp functions](https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions))
that are not available in TensorFlow. The implementation and function behavior
align with the [BigQuery](https://cloud.google.com/bigquery). This is part of an
effort to bridge the gap between the SQL community and the Tensorflow community.
The following example returns the same result as `TIMESTAMP_ADD(timestamp_expression, INTERVAL int64_expression date_part)`

```
>>> timestamp = tf.constant(['2008-12-25 15:30:00+00', '2023-11-11 14:30:00+00'], dtype=tf.string)
>>> interval = tf.constant([200, 300], dtype=tf.int64)
>>> result = timestamp_ops.timestamp_add(timestamp, interval, 'MINUTE')
tf.Tensor([b'2008-12-25 18:50:00.0 +0000' b'2023-11-11 19:30:00.0 +0000'], shape=(2,), dtype=string)
```

Note: `/usr/share/zoneinfo` is needed for parsing time zone which might not be
available in your OS. You will need to install `tzdata` to generate it. For
example, add the following code in your Dockerfile.

```
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" \
    TZ="America/Los_Angeles" apt-get install -y tzdata
```

### Model Generator

#### Text Embedding Model Generator

The Text Embedding Model Generator automatically loads a text embedding model
from Tensorflow hub and integrates a signature such that the resulting model can
be immediately integrated within BQML. Currently, the NNLM and BERT embedding
models can be selected.

##### NNLM Text Embedding Model

The [NNLM](https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2) model
has a model size of <150MB and is recommended for phrases, news, tweets,
reviews, etc. NNLM does not carry any default signatures because it is designed
to be utilized as a Keras layer; however, the Text Embedding Model Generator
takes care of this.

##### SWIVEL Text Embedding Model

The [SWIVEL](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1) model
has a model size of <150MB and is recommended for phrases, news, tweets,
reviews, etc. SWIVEL does not require pre-processing because the embedding model
already satisfies BQML imported model requirements. However, in order to align
signatures for NNLM, SWIVEL, and BERT, the Text Embedding Model Generator
establishes the same input label for SWIVEL.


##### BERT Text Embedding Model

The BERT model has a model size of ~200MB and is recommended for phrases, news,
tweets, reviews, paragraphs, etc. The [BERT](https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4) model does not carry any default signatures
because it is designed to be utilized as a Keras layer. The Text Embedding Model
Generator takes care of this and also integrates a [text preprocessing layer](https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3) for BERT.
