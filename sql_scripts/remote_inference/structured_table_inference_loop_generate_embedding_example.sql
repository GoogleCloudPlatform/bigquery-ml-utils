-- *** Updating the fields below is required ***
-- The name of the source table
DECLARE source_table DEFAULT 'bigquery-public-data.bbc_news.fulltext';
-- The name of the target table
DECLARE target_table DEFAULT 'target_dataset.news_body_embeddings';
-- The name of the ML model to use for the ML operation
DECLARE ml_model DEFAULT 'target_dataset.embedding_model';
-- The STRING column from the source table passed to GENERATE_EMBEDDING()
DECLARE content_column DEFAULT 'body';
-- The unique key columns from the source table. These columns are used to identify new rows present
-- in the source table and not the target table. '*' is not supported.
DECLARE key_columns DEFAULT ARRAY['filename'];
-- *** End of section ***

-- *** Updating the fields below is optional ***
-- The number of rows to process per child job. A larger value will reduce the overhead of multiple
-- child jobs, but needs to be small enough to complete in a single job run.
DECLARE batch_size DEFAULT 80000;
-- The time to wait before the script terminates
DECLARE termination_time_secs DEFAULT(23 * 60 * 60);
-- An optional where clause to apply to the source table
DECLARE where_clause DEFAULT 'TRUE';
-- The columns to project from the source table to the target table
DECLARE projection_columns DEFAULT ARRAY['*'];
-- The ML options to use for the ML operation
DECLARE ml_options DEFAULT 'STRUCT(TRUE AS flatten_json_output)';
-- *** End of section ***

-- *** Updating the fields below should be quite rare ***
-- The ML query to use for the ML operation, requires the unique key
DECLARE
  ml_query
    DEFAULT
      FORMAT(
        'SELECT %s, %s AS content FROM `%s` WHERE %s',
        ARRAY_TO_STRING(projection_columns, ','),
        content_column,
        source_table,
        where_clause);

-- The filter condition for accepting the ML result into the target table
DECLARE
  accept_filter
    DEFAULT 'ml_generate_embedding_status' || " NOT LIKE 'A retryable error occurred:%'";

DECLARE
  key_cols_filter
    DEFAULT(
      SELECT
        STRING_AGG('S.' || KEY || ' = T.' || KEY, ' AND ')
      FROM
        UNNEST(key_columns) AS KEY
    );
-- *** End of section ***

-- Create the target table first if it does not exist
EXECUTE
  IMMEDIATE
    FORMAT(
      '''
CREATE TABLE IF NOT EXISTS `%s` AS
  (SELECT *
   FROM ML.GENERATE_EMBEDDING (MODEL `%s`,
           (SELECT *
            FROM (%s)
            LIMIT 10), %s)
   WHERE %s)''',
      target_table,
      ml_model,
      ml_query,
      ml_options,
      accept_filter);

-- Iteratively populate the target table
REPEAT
DROP TABLE IF EXISTS _SESSION.embedding_batch;

-- Identify new rows in the source table to generate embeddings
-- For throughput reasons, Materialize these rows into a temp table before calling GENERATE_EMBEDDING()
EXECUTE
  IMMEDIATE
    FORMAT(
      '''
      CREATE TEMP TABLE _SESSION.embedding_batch AS
      (SELECT *
          FROM (%s) AS S
          WHERE NOT EXISTS (SELECT * FROM %s AS T WHERE %s) LIMIT %d)
    ''',
      ml_query,
      target_table,
      key_cols_filter,
      batch_size);

-- Generate embeddings for these rows and insert them into the target table
EXECUTE
  IMMEDIATE
    FORMAT(
      '''
        INSERT `%s`
        SELECT *
            FROM ML.GENERATE_EMBEDDING (MODEL `%s`,
                    TABLE _SESSION.embedding_batch, %s)
            WHERE %s
        ''',
      target_table,
      ml_model,
      ml_options,
      accept_filter);

UNTIL(
  SELECT
    @@row_count
)
= 0
OR TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), @@script.creation_time, SECOND)
  >= termination_time_secs
    END
      REPEAT;