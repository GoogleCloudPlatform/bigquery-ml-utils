-- Update the required fields below --
-- The name of the source table
DECLARE source_table DEFAULT "bigquery-public-data.bbc_news.fulltext";
-- The name of the target table
DECLARE target_table DEFAULT "target_dataset.news_body_embeddings";
-- The name of the ML model to use for the ML operation
DECLARE ml_model DEFAULT "target_dataset.embedding_model";
-- The STRING column used for embedding
DECLARE content_column DEFAULT "body";
-- The unique key columns
DECLARE key_columns DEFAULT ARRAY["filename"];

-- Updating the fields below is optional --
-- The number of rows to process per each query
DECLARE batch_size DEFAULT 80000;
DECLARE termination_time_secs DEFAULT(23 * 60 * 60);

-- Updating the fields below should be quite rare --
-- The ML query to use for the ML operation, requires the unique key
DECLARE
  ml_query
    DEFAULT
      FORMAT(
        "SELECT %s, %s AS content FROM `%s`", ARRAY_TO_STRING(key_columns, ','), content_column, source_table);

-- The ML options to use for the ML operation
DECLARE ml_options DEFAULT "STRUCT(TRUE AS flatten_json_output)";
-- Name of the status column as output by the above ML operation
DECLARE ml_status_col_name DEFAULT "ml_generate_embedding_status";
-- The filter condition for accepting the ML result into the target table
DECLARE accept_filter DEFAULT ml_status_col_name || " NOT LIKE 'A retryable error occurred:%'";
DECLARE key_cols_filter DEFAULT(
SELECT
  STRING_AGG("S." || KEY || " = T." || KEY, " AND ")
FROM
  UNNEST(key_columns) AS KEY );

-- Create the target table first if it does not exist
EXECUTE IMMEDIATE
FORMAT( """
CREATE TABLE IF NOT EXISTS `%s` AS
  (SELECT *
   FROM ML.GENERATE_EMBEDDING (MODEL `%s`,
           (SELECT *
            FROM (%s)
            LIMIT 10), %s)
   WHERE %s)""", target_table, ml_model, ml_query, ml_options, accept_filter);

-- Iteratively populate the target table
REPEAT
  DROP TABLE IF EXISTS _SESSION.new_rows;

  -- Identify new rows in the source table to generate embeddings
  -- Materialize these rows into a temp table for throughput reasons with GENERATE_EMBEDDING()
  EXECUTE IMMEDIATE
    FORMAT( """
      CREATE TEMP TABLE _SESSION.new_rows AS
      (SELECT *
          FROM (%s) AS S
          WHERE NOT EXISTS (SELECT * FROM %s AS T WHERE %s) LIMIT %d)
    """, ml_query, target_table, key_cols_filter, batch_size);
  
  -- Generate embeddings for these rows and insert them into the target table
  EXECUTE IMMEDIATE
    FORMAT( """
        INSERT `%s`
        SELECT *
            FROM ML.GENERATE_EMBEDDING (MODEL `%s`,
                    TABLE _SESSION.new_rows, %s)
            WHERE %s
        """, target_table, ml_model, ml_options, accept_filter);
      UNTIL(
  SELECT
    @@row_count) = 0
  OR TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), @@script.creation_time, SECOND) >= termination_time_secs
END
REPEAT;
