# PILATES Dual Storage Test Suite

This directory contains comprehensive tests for the PILATES dual storage functionality, which enables ActivitySim database input mode with preservation of both raw UrbanSim data and processed ActivitySim data.

## Test Files

### 1. `test_database_components.py` 
**Fast component tests (always run)**

Tests individual database components without requiring external H5 files:
- ✅ Database manager creation and initialization
- ✅ Dual storage table schema creation
- ✅ Data cleaning functions for ActivitySim compatibility
- ✅ Placeholder data generation
- ✅ Error handling and edge cases
- ✅ Basic database operations

**Runtime: ~1-2 seconds**

### 2. `test_dual_storage_workflow.py`
**Full workflow tests (requires H5 files)**

Tests the complete end-to-end dual storage workflow:
- 🔍 H5 structure detection (base year vs forecast year formats)
- 📤 Dual extraction (raw UrbanSim + processed ActivitySim data)
- 💾 Database upload with actual data storage
- 📄 CSV generation from H5 files (traditional path)
- 🗄️ CSV generation from database (new path)
- 🔍 Comparison between H5-generated and database-generated CSVs

**Runtime: 30 seconds - 5 minutes (depending on H5 file size)**

## Running Tests

### Quick Component Tests (Recommended for CI/CD)
```bash
# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pilates

# Run fast component tests
python -m pytest tests/test_database_components.py -v
```

### Full Workflow Tests (Requires H5 Files)
```bash
# Ensure H5 files are available in:
# - pilates/urbansim/data/*.h5
# - urbansim/data/*.h5
# - tmp/*/urbansim/data/*.h5

# Run workflow tests
python -m pytest tests/test_dual_storage_workflow.py -v -s
```

### Using the Test Runner Script
```bash
# Run all available tests
python run_dual_storage_tests.py

# Run only component tests
python run_dual_storage_tests.py --type components

# Run only workflow tests (if H5 files available)  
python run_dual_storage_tests.py --type workflow

# Verbose output
python run_dual_storage_tests.py --verbose
```

## What Gets Tested

### ✅ Dual Storage Architecture
- Raw UrbanSim data storage (`urbansim_*_raw` tables)
- Processed ActivitySim data storage (`activitysim_*` tables)
- Database schema with proper indexes and constraints
- Foreign key relationships and data integrity

### ✅ H5 Format Compatibility  
- Base year format: `/households`, `/persons`, etc.
- Forecast year format: `/2020/households`, `/2020/persons`, etc.
- Automatic structure detection and path resolution
- Robust error handling for missing or malformed data

### ✅ Data Preservation and Integrity
- Extraction of both raw and processed data from H5 files
- Database storage with actual data (not just pointers)
- Data retrieval with proper formatting for ActivitySim
- Comparison between H5-generated and database-generated outputs

### ✅ ActivitySim Integration
- Database input mode configuration
- CSV file generation with correct schemas and indexes
- Data cleaning and validation for ActivitySim compatibility
- Graceful fallbacks when data is missing

## Expected Results

### Component Tests
- **All 8 tests should PASS** in ~1-2 seconds
- Tests core database functionality without external dependencies
- Safe to run in any environment with PILATES installed

### Workflow Tests  
- **Depends on H5 file availability**
- If H5 files found: Tests should PASS or show MINOR differences
- If no H5 files: Tests will be SKIPPED with informative messages
- **Critical validation**: CSV comparison between H5 and database paths

## Interpreting Results

### ✅ Perfect Success
```
📈 Comparison Summary:
   Perfect matches: 3/3
   Tables with perfect matches: ['households', 'persons', 'land_use']
🎉 All tests PASSED! Dual storage functionality is working correctly.
```

### ⚠️ Minor Differences (Acceptable)
```
📊 Comparing households table...
     H5 shape: (1000, 7)
     DB shape: (1000, 7)  
     ✅ All common columns match perfectly
📋 Result: ✅ Perfect match
```

### ❌ Significant Issues (Needs Investigation)
```
❌ Row count mismatch: H5=1000, DB=800
⚠️ Columns missing in DB: {'important_column'}
⚠️ Column differences: {'income': 'numeric differences in 50 rows'}
```

## Troubleshooting

### No H5 Files Found
```bash
# Download a test H5 file
cd pilates/urbansim/data/
wget -O custom_mpo_06197001_model_data.h5 \
  "https://www.dropbox.com/scl/fi/l8396ztutpbcoucytywpz/custom_mpo_06197001_model_data.h5?rlkey=xyon6ck73deced7hoqlqtdass&dl=1"
```

### Database Permissions Issues
```bash
# Ensure write permissions in temp directory
export TMPDIR=/tmp/pilates_test
mkdir -p $TMPDIR
```

### Memory Issues with Large H5 Files
- Use `--type components` for faster testing
- Consider sampling data in workflow tests
- Monitor memory usage during large H5 processing

## Integration with CI/CD

For continuous integration, recommend:

1. **Always run**: `test_database_components.py` (fast, no dependencies)
2. **Optional**: `test_dual_storage_workflow.py` (if H5 files available)
3. **Test data**: Consider including a small test H5 file in the repository

```yaml
# Example GitHub Actions step
- name: Test Dual Storage
  run: |
    conda activate pilates
    python -m pytest tests/test_database_components.py -v
    python run_dual_storage_tests.py --type components
```

## Performance Benchmarks

| Test Type | Runtime | Memory | Purpose |
|-----------|---------|--------|---------|
| Components | ~1-2s | <100MB | Unit tests, schema validation |
| Small H5 (~10MB) | ~30s | ~200MB | Basic workflow validation |  
| Large H5 (~100MB) | ~5min | ~1GB | Full production validation |

---

**🎯 Goal**: Ensure that database-generated ActivitySim inputs are functionally equivalent to H5-generated inputs, enabling reliable database input mode for improved performance and cloud deployment capabilities.