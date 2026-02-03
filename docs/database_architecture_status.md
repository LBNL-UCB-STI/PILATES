# **PILATES Database Architecture: Current Status & Roadmap**

Date: November 19, 2025  
Status: Operational / Beta  
Backend: DuckDB \+ Python Integration

## **1\. Executive Summary**

The PILATES database subsystem has been re-architected from a memory-intensive, Pandas-based loading process to a **High-Performance ELT (Extract, Load, Transform) Pipeline**.  
By leveraging DuckDB's native C++ engine and Parquet metadata optimization, the system can now ingest gigabyte-scale travel model outputs (ActivitySim, UrbanSim, BEAM) with near-constant memory usage and significantly reduced storage footprints. The system automatically infers schemas, downcasts data types for efficiency, and enforces relational integrity through a strict lineage tracking system.

## **2\. System Architecture**

The pipeline consists of four distinct stages:

1. **Inference (Pre-Computation):** Before loading data, we scan file footers (Parquet) or sample rows (CSV) to calculate statistics (Min/Max, Cardinality).  
2. **Definition (DDL Generation):** We generate optimized SQL CREATE TABLE statements using the inferred stats to choose the smallest possible integer types (SMALLINT vs BIGINT) and create ENUM types for categorical strings.  
3. **Initialization:** The database is created with dependency awareness, ensuring Core Metadata tables exist before Model Output tables try to reference them via Foreign Keys.  
4. **Ingestion (Zero-Copy Loading):** Data is streamed directly from disk to the database file using INSERT INTO ... SELECT ... FROM read\_parquet. Python acts only as an orchestrator; data does not pass through Python memory.

## **3\. Deep Dive: The run\_info.json Pipeline**

The run\_info.json file serves as the "Contract" between the model run execution and the database loading process. The pipeline follows this specific sequence:

### **Step 1: Metadata Capture (Provenance)**

During a model run, the FileProvenanceTracker records every input and output file. Crucially, it invokes pilates.utils.schema\_inference to attach rich metadata to the run\_info.json record:

* **Path Tracking:** Stores relative paths to ensure portability across machines.  
* **Schema Inference:**  
  * Reads Parquet footers (Zero-Copy) to extract Min and Max values for numeric columns.  
  * Calculates cardinality for string columns to flag potential ENUMs.  
  * **Output:** A JSON block in run\_info like {"name": "household\_id", "type": "int64", "min": 1, "max": 50000}.

### **Step 2: Schema Generation (DDL)**

The pilates.database.schema\_generator script parses the run\_info.json **offline** (or post-run) to generate highly optimized SQL DDL files.

* **Type Mapping logic:**  
  * If min \> \-32k and max \< 32k → SMALLINT (2 bytes).  
  * If is\_enum is True → CREATE TYPE ... AS ENUM (...).  
  * Otherwise → BIGINT, DOUBLE, or VARCHAR.  
* **Output:** A .sql file in pilates/database/schema/generated/ (e.g., trips.sql).

### **Step 3: Database Initialization**

The DuckDBManager reads the generated .sql files to instantiate the schema structure **before** any data is loaded.

* **Order of Operations:** Core tables (runs, file\_records) are created first, followed by model tables (trips, households).  
* **Constraints:** Primary Keys and Foreign Keys are established on empty tables.

### **Step 4: Smart Data Loading**

The pilates.database.selective\_uploader performs the final ingestion using a **"Peek & Project"** strategy to maintain performance while handling schema drift.

1. **Peek:** It quickly inspects the file header using DESCRIBE SELECT \* FROM read\_parquet(...).  
2. **Project:** It constructs a dynamic SQL query to handle mismatches:  
   * *Renames:* SELECT "year" AS data\_year ... (prevents collision with metadata year).  
   * *Casts:* SELECT CAST(sector\_id AS VARCHAR) ....  
   * *Injection:* SELECT ... 'run\_abc123' AS run\_id ... (injects lineage IDs as constants).  
3. **Load:** Executes INSERT INTO target\_table SELECT ... which streams data directly from disk to the DB file.  
4. **Sort:** Appends ORDER BY run\_id, year, zone\_id to ensure physical data clustering for fast read performance.

## **4\. Key Features Implemented**

### **A. Smart Schema Inference**

* **Parquet Optimization:** Reads file metadata (footers) to extract Min/Max statistics without loading the data payload.  
* **Integer Downcasting:** Automatically detects if a column fits in SMALLINT (2 bytes) or INTEGER (4 bytes) versus the default BIGINT (8 bytes), saving \~50-75% storage for ID columns.  
* **Enum Detection:** Identifies low-cardinality string columns (e.g., tour\_mode, purpose) and converts them to strict SQL ENUM types.  
* **Numpy Compatibility:** Handles JSON serialization of NumPy types safely.

### **B. Zero-Copy Data Ingestion**

* **Native Loading:** Replaced pd.read\_parquet() with SQL-based read\_parquet() and read\_csv\_auto().  
* **Metadata Injection:** Injects lineage data (run\_id, year, iteration, file\_record\_id) directly into the SQL projection stream.  
* **Smart Projections:** Handles schema drift (e.g., renaming year \-\> data\_year, casting sector\_id \-\> VARCHAR) dynamically within the SQL query, avoiding Pandas memory overhead.

### **C. Performance Optimization**

* **Physical Clustering:** Enforces a "Golden Sort Order" (ORDER BY run\_id, year, iteration, zone\_id) during write. This activates DuckDB's **Min/Max Indexing (Zone Maps)**, allowing analytical queries to skip 90%+ of data blocks when filtering by geography.  
* **Transaction Atomicity:** All uploads are wrapped in transactions. If a 10GB upload fails at 9.9GB, the database rolls back to a clean state.

### **D. Robust Testing**

* **Unit Tests:** Validation of schema inference logic, integer range detection, and Enum identification.  
* **Integration Tests:** Full mock workflow verifying the chain from UrbanSim \-\> ATLAS \-\> ActivitySim \-\> BEAM.  
* **Constraint Testing:** Verification of Foreign Keys and Uniqueness constraints preventing bad data entry.

## **5\. Code Structure**

| Component | File | Responsibility |
| :---- | :---- | :---- |
| **Inference** | pilates/utils/schema\_inference.py | Standalone functions to read file stats. |
| **Generator** | pilates/database/schema\_generator.py | Converts JSON stats to .sql DDL files. |
| **Manager** | pilates/utils/duckdb\_manager.py | Handles connections, transactions, and dependencies. |
| **Loader** | pilates/database/selective\_uploader.py | Orchestrates the high-speed data transfer. |
| **Metadata** | pilates/generic/records.py | Data classes for lineage tracking. |

## **6\. Roadmap & Strategy**

### **Phase 1: Consumption & Usability (Immediate)**

* **Standardized Views:** Analysts shouldn't have to write joins between trips and households. We should generate standard views (e.g., v\_trips\_with\_demographics) automatically.  
* **Data Dictionary UI:** The export\_data\_dictionary HTML output is good, but hosting it alongside the run results would be better.

### **Phase 2: Advanced Analytics (Mid-Term)**

* **Scenario Comparison:** Built-in SQL functions or Python APIs to diff two runs efficiently (e.g., COMPARE\_MODE\_SHARE(run\_id\_A, run\_id\_B)).  
* **Spatial Integration:** DuckDB has a strong spatial extension. We could auto-convert lat/lon columns to GEOMETRY types for direct GIS integration.

### **Phase 3: Enterprise Scale (Long-Term)**

* **Cloud/S3 Support:** DuckDB can query Parquet files directly on S3. The current loader assumes local files; adapting it to "attach" remote buckets would allow serverless operation.  
* **MotherDuck Integration:** For collaborative sharing of results without copying .duckdb files.