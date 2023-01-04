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
