# BigQuery Remote Inference Scripts

BigQuery supports remote models, such as Vertex AI LLMs, to perform remote inference operations on both structured and unstructured data. When using remote inference, the user needs to be aware of quotas and limits. If these limits are exceeded, it can result in retryable errors for a subset of rows. This often requires reprocessing.

In cases when a retryable error has occurred for some rows, we provide SQL scripts to iterate through the inference call until all rows have been successfully annotated.

There are two scripts based whether the input data is in an object table or a native BigQuery table. 

## Object table script
The object table script creates a target table to store successful annotations. To do this, it calls the inference in a loop. In the first iteration, a small LIMIT is set on the inference call to quickly create a table with the desired schema. The number of rows to process for each inference call can be modified through the batch_size parameter.

This script applies to the following models:
- ML.ANNOTATE_IMAGE
- ML.PROCESS_DOCUMENT
- ML.TRANSCRIBE

For the object SQL script, you need to update the following parameters at the top of the [object table script](object_table_inference_loop_generic.sql):

```
-- The name of the object table
DECLARE obj_table DEFAULT /* obj_table name */;

-- The name of the target table
DECLARE target_table DEFAULT /* target_table name */;

-- The name of the unique key column
DECLARE key_column DEFAULT "uri";

-- The name of the ML model to use for the ML operation
DECLARE ml_model DEFAULT /* model name */;

-- The SQL query on the object table to perform the desired ML operation
DECLARE ml_function DEFAULT FORMAT("""
  SELECT * FROM /* ml function name */(
  MODEL `%s`,
  TABLE `%s`,
  /* ml function options */""",
  ml_model, obj_table);

-- Name of the status column as output by the above ML operation
DECLARE ml_status_col_name DEFAULT /* status column name */;
```

We provide an example using ML.ANNOTATE_IMAGE under the [object table example script](object_table_script_inference_loop_example.sql).

## Structured table script

This script creates a target table to track all successful annotations and loops through the inference call until all rows are annotated. 

To find the rows that need to be annotated at each iteration, you need to refer to the candidate key of the table, the `key_columns` parameter in the script.

This script applies to the following models:
- ML.GENERATE_EMBEDDINGS
- ML.GENERATE_TEXT
- ML.UNDERSTAND_TEXT
- ML.TRANSLATE 

For the SQL script, you need to update the following parameters at the top of the [structured table script](structured_table_inference_loop_generic.sql):

```
-- The name of the source table
DECLARE source_table DEFAULT /* source table name */;

-- The name of the target table
DECLARE target_table DEFAULT /* target table name */;

-- The unique key columns
DECLARE key_columns DEFAULT ARRAY[/* key columns */];

-- The name of the ML model to use for the ML operation
DECLARE ml_model DEFAULT /* ml model name */;

-- The name of the ML function to use for the ML operation
DECLARE ml_function DEFAULT /* ml function name */;

-- The ML query to use for the ML operation, requires the unique key
DECLARE
  ml_query
    DEFAULT
      FORMAT(
        "SELECT %s, text AS content FROM `%s`", ARRAY_TO_STRING(key_columns, ','), source_table);

-- The ML options to use for the ML operation
DECLARE ml_options DEFAULT /* ml function options */;

-- Name of the status column as output by the above ML operation
DECLARE ml_status_col_name DEFAULT /* status column name */;
```

We provide an example using ML.GENERATE_EMBEDDING in the [structured table example script]( structured_table_script_inference_loop_example.sql).