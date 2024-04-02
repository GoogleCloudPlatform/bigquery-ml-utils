-- *** Please fill in this section ***
-- Please note that the following script needs to be filled in and will not run in the given state.

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
-- The filter condition for accepting the ML result into the target table
DECLARE accept_filter DEFAULT ml_status_col_name || " NOT LIKE 'A retryable error occurred:%'";
-- The number of rows to process per each query
DECLARE batch_size DEFAULT 10000;
-- The number of seconds elapsed to have this script terminated
DECLARE termination_time_secs DEFAULT(23 * 60 * 60);
-- *** End of section ***

-- Create the target table first if it does not exist
EXECUTE
  IMMEDIATE
    FORMAT(
      """
  CREATE TABLE IF NOT EXISTS `%s` AS
    (SELECT *
     FROM %s (MODEL `%s`,
             (SELECT *
              FROM (%s)
              LIMIT %d), %s)
     WHERE %s)""",
      target_table,
      ml_function,
      ml_model,
      ml_query,
      bucket_size,
      ml_options,
      accept_filter);

-- Iteratively populate the target table
BEGIN
  DECLARE cols_assignment STRING;

DECLARE
  key_cols_filter
    DEFAULT(
      SELECT STRING_AGG("S." || key || " = T." || key, " AND ") FROM UNNEST(key_columns) AS key
    );

EXECUTE
  IMMEDIATE
    FORMAT(
      """
    SELECT
      STRING_AGG(column_name || ' = S.' || column_name, ', ')
      FROM `%s.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '%s'""",
      LEFT(target_table, INSTR(target_table, ".", -1) - 1),
      SUBSTR(target_table, INSTR(target_table, ".", -1) + 1))
      INTO cols_assignment;

REPEAT
  EXECUTE
    IMMEDIATE
      FORMAT(
        """
      MERGE `%s` T
      USING (SELECT *
             FROM %s (MODEL `%s`,
                     (SELECT *
                      FROM (%s) AS S
                      WHERE NOT EXISTS (SELECT * FROM %s AS T WHERE %s) LIMIT %d), %s)
             WHERE %s) S
      ON %s
      WHEN NOT MATCHED THEN INSERT ROW
      WHEN MATCHED THEN UPDATE SET %s
      """,
        target_table,
        ml_function,
        ml_model,
        ml_query,
        target_table,
        key_cols_filter,
        bucket_size,
        ml_options,
        accept_filter,
        key_cols_filter,
        cols_assignment);

UNTIL(SELECT @@row_count)
= 0
OR TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), @@script.creation_time, SECOND)
  >= termination_time_secs
    END REPEAT;

END;
