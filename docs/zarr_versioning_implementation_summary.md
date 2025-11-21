# Data Versioning & Analysis Architecture Plan

**Status**: 📝 **DESIGN & PLANNING**
**Date**: November 20, 2025

## Overview

This document outlines a new plan for versioning PILATES simulation data artifacts (e.g., Zarr skims, NetCDF outputs). The architecture is designed to provide **robust, immutable storage** for data provenance and reproducibility, combined with a **powerful, flexible analysis layer** for ergonomic, cross-run comparisons.

The system is separated into two distinct components:
1.  **Storage Layer**: Prioritizes simplicity and data integrity by storing each data artifact version as a full, independent snapshot in a persistent location (e.g., cloud object storage).
2.  **Analysis Layer**: Provides a virtual, unified `xarray.Dataset` view over multiple, independent snapshots, enabling powerful `numpy`-like slicing and analysis across different runs, years, and iterations.

This approach solves the flaws of shared, mutable data stores (data corruption, concurrency issues) while providing a superior analysis experience.

## Proposed Architecture

The new architecture decouples physical storage from logical analysis, while being flexible enough to handle various data formats like Zarr and NetCDF.

```
+-----------------------------+      +---------------------------------+
|   PILATES Run Environment   |      |   Persistent Storage (e.g., S3) |
|  (e.g., Local, HPC, Cloud)  |      |                                 |
|                             |      | /data_archives/                 |
|  +-----------------------+  |      |  ├─ {snapshot_id_1}/artifact.zarr/  (or .nc) |
|  | BEAM Postprocessor    |  |      |  ├─ {snapshot_id_2}/artifact.zarr/  (or .nc) |
|  | Generates artifact    |  |      |  └─ {snapshot_id_3}/artifact.zarr/  (or .nc) |
|  +-----------------------+  |      +----------------|----------------+
|             |               |                       ^
|             v               |                       | 1. Snapshot
|  +-----------------------+  |                       |
|  |      Snapshotter      |-------------------------+
|  | (Copies full artifact)|
|  +-----------------------+
|             |
|             v 2. Write Metadata
|  +-----------------------+
|  |   Central Database    |
|  |  (e.g., PostgreSQL)   |
|  | ┌-------------------┐ |
|  | | snapshots         | |
|  | └-------------------┘ |
|  +-----------------------+
|             ^
|             | 3. Query Snapshots
|             |
|  +-----------------------+
|  |   Analysis Client     |
|  | (Jupyter, Script)   |
|  | +-------------------+ |
|  | | SnapshotAnalysisMgr | |
|  | +-------------------+ |
|  +-----------------------+

```

### 1. Storage Layer: Immutable Snapshots

-   **Principle**: Treat each generated data artifact as an immutable artifact.
-   **Workflow**: After a model run produces a final data artifact (e.g., a `skims.zarr` directory or a `data.nc` file), a "snapshot" process copies the entire artifact to a unique, persistent path in a central storage location (like S3 or a network file system).
-   **Path Scheme**: `s3://<bucket>/data_archives/{snapshot_id}/artifact.zarr` (or `artifact.nc`). The `artifact_path` column in the database will specify the exact location.
-   **Benefit**: This eliminates concurrency problems and data corruption risks. Restoration is a simple `copy` operation. Data integrity is guaranteed.

### 2. Analysis Layer: Virtual Unified View

-   **Principle**: Provide a high-level, ergonomic interface for data analysis without altering the underlying storage.
-   **Component**: A new `SnapshotAnalysisManager` class will serve as the query engine.
-   **Workflow**:
    1.  The manager queries the central database to find paths to relevant snapshots based on user criteria (e.g., run IDs, years) and their `format` (e.g., 'zarr', 'netcdf').
    2.  It uses the appropriate `xarray` function (`xarray.open_zarr()` or `xarray.open_dataset()`) to **lazily** open each of the independent data artifacts. This is fast as it only reads metadata.
    3.  It concatenates these individual datasets into a single `xarray.Dataset` and applies a **`MultiIndex`** using metadata (`run_id`, `year`, `iteration`).
-   **Benefit**: This provides the user with a single, massive-feeling dataset that can be sliced intuitively, while the underlying data remains in simple, independent archives.

## Key Components & Methods

### 1. Snapshotter (Process)
A simple, reliable process that performs two actions after a data artifact is finalized:
1.  Copies the entire artifact (directory for Zarr, file for NetCDF) to its unique, archival storage path.
2.  Writes the metadata (including the `artifact_path`, `format`, `run_id`, `year`, `chunk_manifest` (if applicable), etc.) to the `snapshots` table in the central database.

### 2. `SnapshotAnalysisManager` (Class)

**Location**: `pilates/utils/snapshot_analysis.py` (to be created)

This class will be the primary tool for post-run analysis.

#### `build_view(query_filters)`
Creates a virtual, multi-dimensional `xarray.Dataset` from multiple snapshots.

```python
# Conceptual Usage
from pilates.utils.snapshot_analysis import SnapshotAnalysisManager

# 1. Initialize manager with DB connection
analyzer = SnapshotAnalysisManager(db_connection)

# 2. Build a view based on a query
my_view = analyzer.build_view(
    run_ids=['run_A', 'run_B'],
    years=[2025, 2030]
)

# 3. Perform powerful, intuitive analysis
#    Xarray's MultiIndex handles the ragged data structure.
sov_time_run_A = my_view['SOV_TIME'].sel(run_id='run_A', year=2025)

# Compare the mean of a skim for the same year across two runs
mean_A = my_view['SOV_DIST'].sel(run_id='run_A', year=2030).mean()
mean_B = my_view['SOV_DIST'].sel(run_id='run_B', year=2030).mean()

# Data is only loaded from storage when a computation is triggered
diff = (mean_A - mean_B).compute()
```

#### `restore_snapshot(snapshot_id, target_path)`
A straightforward method to retrieve a historical artifact.

-   **Logic**:
    1.  Queries the database to find the `artifact_path` and `format` for the given `snapshot_id`.
    2.  Copies the entire artifact from that archival path to the local `target_path`.

## Database Integration

The `manifest.json` file is superseded by the central database. The `snapshots` table is the source of truth for all versioning metadata. Key columns relevant for storing different artifact types are:
-   `snapshot_id`: The primary key for a version.
-   `artifact_path`: The direct, unique path to the archived data artifact (e.g., `skims.zarr`, `data.nc`).
-   `format`: The format of the artifact (e.g., `'zarr'`, `'netcdf'`).
-   `run_id`, `year`, `iteration`: Metadata used for querying and building the analysis `MultiIndex`.
-   `chunk_manifest`: Still stored for Zarr artifacts for potential future use (see Future Work) and for deep data auditing.

## Advantages of this Architecture

1.  **Solves Data Corruption Risk**: The "last write wins" problem of a shared, mutable store is completely eliminated by using immutable, independent snapshots.
2.  **Cloud-Native & Scalable**: This design is perfectly suited for cloud object storage (S3, GCS) and avoids filesystem-specific features like hardlinks.
3.  **Simplifies Concurrency**: Multiple PILATES runs can execute and snapshot their data in parallel without interfering with each other.
4.  **Enables Powerful Analysis**: The `SnapshotAnalysisManager` provides a far more powerful and ergonomic analysis experience than previous methods, gracefully handling ragged data across runs and years.
5.  **Robust & Simple Restoration**: Restoring a snapshot is a simple, predictable bulk copy operation.
6.  **Future-Proof for Other Formats**: The design inherently supports different `xarray`-compatible data formats like Zarr and NetCDF with minimal, isolated changes.

## Trade-offs & Future Work

**Primary Trade-off**:
-   **Storage Cost**: This approach prioritizes simplicity, robustness, and developer velocity over storage efficiency. It intentionally duplicates unchanged data between snapshots. This is a conscious design choice based on the principle that storage is cheaper than complex engineering and data corruption risks.

**Future Enhancements**:
1.  **Content-Addressable Chunk Store**: If storage costs become a significant concern, the system can be evolved into a more "Git-like" model. Because we are already storing the `chunk_manifest` in the database, we have the necessary metadata to implement a central, content-addressable chunk pool for global deduplication. This would be a major engineering effort but would represent the gold standard for storage efficiency.
2.  **Analysis View Optimization**: For very large queries (spanning terabytes of skim data), the `SnapshotAnalysisManager` could be optimized to use distributed computing backends like Dask.
3.  **Automated Cleanup**: Implement retention policies to automatically archive or delete old snapshots from storage based on rules defined in the database.

## Conclusion

This two-layer architecture provides a clear path forward. It establishes a robust and simple foundation for data versioning while enabling a highly sophisticated and flexible analysis workflow. It addresses the key limitations of the previous plan and is well-suited for a scalable, multi-run simulation environment.
