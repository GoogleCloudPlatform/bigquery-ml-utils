-- The name of the object table
DECLARE obj_table DEFAULT "sample.imagesets";
-- The name of the target table, which is a native BQ table
DECLARE target_table DEFAULT "sample.annotated_imagesets";
-- The name of the unique key column
DECLARE key_column DEFAULT "uri";
-- The name of the ML model to use for the ML operation
DECLARE ml_model DEFAULT "sample.vision";
-- The SQL query on the object table to perform the desired ML operation
DECLARE ml_function DEFAULT FORMAT("""
  SELECT * FROM ML.ANNOTATE_IMAGE(
  MODEL `%s`,
  TABLE `%s`,
  STRUCT(['LABEL_DETECTION'] AS vision_features))""",
  ml_model, obj_table);
-- Name of the status column as output by the above ML operation
DECLARE ml_status_col_name DEFAULT "ml_annotate_image_status";
-- The filter condition for accepting the ML result into the target table
DECLARE accept_filter DEFAULT ml_status_col_name || " NOT LIKE 'A retryable error occurred:%'";
-- The number of rows to process per each query
DECLARE batch_size DEFAULT 500;
-- The number of seconds elapsed to have this script terminated
DECLARE termination_time_secs DEFAULT (22 * 60 * 60);


-- Incrementally perform a given ML operation over a source table
-- until the target table is fully populated or execution time
-- exceeded the termination_time_secs
BEGIN
  DECLARE cols_assignment STRING;
  DECLARE selected_keys ARRAY<STRING>;

  -- Creates the target table if it does not exist.
  --
  -- The table is created by running the ML operation and copying rows that are accepted
  -- by the filter into the target table. A small limit is used to create the table with
  -- the desired schema and to avoid spending too much time in computing the ML operation.
  EXECUTE IMMEDIATE FORMAT("""
    CREATE TABLE IF NOT EXISTS `%s`
    AS %s
    LIMIT 10""",
    target_table, ml_function);

  -- Forms the field assignment statement based on the target table column.
  -- It will be used for the subsequence MERGE operations
  EXECUTE IMMEDIATE FORMAT("""
    SELECT STRING_AGG(column_name || ' = S.' || column_name, ', ')
    FROM `%s.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '%s'""",
    LEFT(target_table, INSTR(target_table, ".", -1) - 1),
    SUBSTR(target_table, INSTR(target_table, ".", -1) + 1)
  ) INTO cols_assignment;

  -- Repeatedly performs the ML operation for objects that are not yet in
  -- the target table, or update the result for objects that
  -- have been changed since the last run.
  REPEAT
    EXECUTE IMMEDIATE FORMAT("""
      SELECT ARRAY(
        SELECT %s
        FROM `%s` AS S
        WHERE NOT EXISTS
          (SELECT * FROM `%s` AS T WHERE S.%s = T.%s)
          OR updated > (SELECT max(updated) FROM `%s`)
        LIMIT %d
      )""",
      key_column, obj_table, target_table, key_column, key_column, target_table, batch_size)
      INTO selected_keys;

    -- This statement merges the target table with the original inference call. Objects with
    -- new labels are added to the target table. Note that the USING clause passes in an
    -- identifier, which can be a variable or value. These identifiers function similarly
    -- to query parameters. Identifiers are bound to placeholders marked as "?".
    EXECUTE IMMEDIATE FORMAT("""
      MERGE %s T
      USING (%s WHERE %s IN UNNEST(?) AND %s) S
      ON S.%s = T.%s
      WHEN NOT MATCHED THEN INSERT ROW
      WHEN MATCHED THEN UPDATE SET %s""",
      target_table, ml_function, key_column, accept_filter, key_column, key_column,
      cols_assignment
    ) USING selected_keys;
    UNTIL (SELECT @@row_count) = 0
           OR TIMESTAMP_DIFF(CURRENT_TIMESTAMP(),
                             @@script.creation_time, SECOND) >= termination_time_secs
  END REPEAT;
END;
