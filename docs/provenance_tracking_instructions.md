# Provenance Tracking Instructions

This document provides detailed instructions on how to modify the code to tag additional model input and output files for provenance tracking. The goal is to ensure that all relevant files are recorded in the provenance output document, allowing for comprehensive tracking of data flow through the models.

## High-Level Process

1. **Identify Key Input/Output Points**: Determine where in the codebase files are read from or written to. These are the points where provenance tracking should be added.

2. **Use ProvenanceTracker**: Utilize the `ProvenanceTracker` class to record inputs and outputs. This class provides methods to log file paths, descriptions, and other metadata.

3. **Standardize Path Handling**: Ensure that all file paths are converted to absolute paths before recording. This helps maintain consistency and avoid issues with relative paths.

4. **Add Validation**: Before recording a file, check if it exists. Log a warning if a file is missing, and decide whether to skip recording it based on the context.

5. **Batch Recording**: Where possible, use batch recording methods to log multiple files at once. This reduces redundancy and simplifies the code.

## Implementation Steps

### Step 1: Identify Input/Output Points

- Review the code to find where files are read or written. This includes data loading, preprocessing, model execution, and postprocessing steps.
- Look for functions that handle file I/O, such as `pd.read_csv`, `pd.to_csv`, `pd.read_parquet`, and `pd.to_parquet`.

### Step 2: Add Provenance Tracking

- Import the `ProvenanceTracker` class if not already imported.
- Use the `record_input_file` and `record_output_file` methods to log files. Provide a description and, if applicable, the year or iteration number.

Example:
```python
state.record_input_file("model_name", file_path, description="Description of the input file")
state.record_output_file("model_name", file_path, year=year, description="Description of the output file")
```

### Step 3: Standardize Path Handling

- Convert file paths to absolute paths before recording them. Use `os.path.abspath(file_path)` to ensure consistency.

### Step 4: Add Validation

- Before recording, check if the file exists using `os.path.exists(file_path)`.
- Log a warning if the file is missing, and decide whether to skip recording it.

Example:
```python
if os.path.exists(file_path):
    state.record_input_file("model_name", file_path, description="Description")
else:
    logger.warning(f"File not found: {file_path}")
```

### Step 5: Use Batch Recording

- For functions that handle multiple files, use the `record_model_io_batch` method to log inputs and outputs in a single call.

Example:
```python
state.record_model_io_batch("model_name", inputs=[input1, input2], outputs=[output1, output2], year=year)
```

## Conclusion

By following these steps, you can ensure that all relevant input and output files are tagged for provenance tracking. This will enhance the traceability and reproducibility of the models, making it easier to understand data flow and debug issues.
