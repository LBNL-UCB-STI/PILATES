# PILATES Zone ID and Skim Alignment Management

## 1. Overview

This document outlines the authoritative process for defining, managing, and using zone identifiers (e.g., TAZ IDs) across the PILATES framework. The primary goal is to ensure that all models—UrbanSim, ActivitySim, ATLAS, and BEAM—operate on a consistent and verifiably correct zone ordering. This prevents silent data corruption and eliminates the need for inefficient, on-the-fly data realignment during model runs.

The entire process is anchored by a **"single source of truth"**: a canonical zone geometry file that defines all valid zones and their identifiers. All model inputs that depend on zone ordering are derived from this single source.

## 2. The Authoritative Data Flow

The following steps describe the end-to-end process of how zone IDs are managed, from the initial source file to their use in different models.

### Step 1: The Canonical Zone Source File

The foundation of the system is a single geometry file (e.g., a Shapefile, GeoJSON, or GeoPackage) that contains the master list of all zones for the simulation region.

-   **Configuration:** The path to this file is specified in `settings.yaml` under `shared.geography.zones.source_file`.
-   **Requirement:** This file **must** contain a column with unique, canonical identifiers for each zone. The name of this column is specified in `shared.geography.zones.canonical_id_col`.

### Step 2: Loading the Canonical Order

All zone information is loaded and processed through a single, authoritative function: `pilates.utils.zone_utils.load_canonical_zones`.

-   **Action:** This function reads the source geometry file specified in the settings.
-   **Validation:** It performs critical validation checks, ensuring the canonical ID column exists and contains no duplicate values.
-   **Sorting:** It sets the canonical ID column as the DataFrame index and **sorts the entire GeoDataFrame by this index**.
-   **Output:** It returns a **sorted GeoDataFrame**, where the index contains the canonical zone IDs in a stable, authoritative order. This sorted DataFrame is the in-memory "source of truth" for all downstream processes.

### Step 3: Preparing Inputs for ActivitySim

The ActivitySim preprocessor is responsible for creating the `land_use.csv` and `skims.omx` files that ActivitySim consumes.

-   **`land_use.csv` Creation:** This process is orchestrated by the `create_asim_data_from_h5` function within the ActivitySim preprocessor.
    1.  **Load Canonical Zones:** It first calls `pilates.utils.zone_utils.load_canonical_zones` to establish the authoritative, sorted list of zones, which forms the base for the `land_use` table.
    2.  **Get Block-to-Zone Mapping:** It then generates a lookup table by calling `pilates.utils.zone_utils.get_block_to_zone_mapping`. This function performs a geometric intersection to map UrbanSim's fine-grained block IDs to the coarser canonical zone IDs (e.g., TAZs). While this mapping is generated and used in-memory, its application is reflected in the output `land_use.csv`, which is tracked for provenance.
    3.  **Read UrbanSim Data:** The preprocessor reads raw `households`, `persons`, `jobs`, and `blocks` data directly from the UrbanSim H5 datastore.
    4.  **Map Block IDs:** The `blocks` DataFrame is enriched by adding a new column (e.g., `TAZ`) where each block ID is mapped to its corresponding canonical zone ID using the lookup table generated in the previous step.
    5.  **Aggregate and Join Data:** The `_create_land_use_table` function then aggregates demographic (persons, households) and employment (jobs) data at the canonical zone level. This aggregated data is joined to the canonically ordered zones DataFrame.
    6.  **Calculate Metrics:** Various density metrics, parking costs, and area types are calculated and added to the table.
    7.  **Output:** The resulting comprehensive DataFrame, now representing the `land_use` table with data aligned to the canonical zone order, is written to `land_use.csv`.

-   **Skim Creation:** The preprocessor calls the `create_skims_from_beam` function, which in turn uses the `zone_order` helper function. This function also calls `load_canonical_zones` to get the exact same sorted list of zone IDs. This list is used to construct the final skim matrices, ensuring they are created with the correct canonical ordering from the start.

### Step 4: Preparing Inputs for BEAM

The BEAM preprocessor (`pilates.beam.preprocessor.prepare_beam_zone_shapefile`) prepares the zone file that BEAM uses for its network and skim generation.

-   **Action:** It calls `load_canonical_zones` to get the sorted GeoDataFrame.
-   **Output:** It saves this sorted data to a new shapefile. This shapefile, now sorted canonically, is passed to the BEAM model run.
-   **Configuration:** It also updates the BEAM configuration (`beam.agentsim.taz.tazIdFieldName`) to ensure BEAM reads the correct ID column from the shapefile.

### Step 5: Skim Generation in BEAM

The BEAM model, running in its own container, now uses the canonically sorted zone shapefile.

-   **BEAM's `ActivitySimZarrWriter`:** When BEAM generates skims in the Zarr format, the `ActivitySimZarrWriter` receives the list of zone IDs in the same sorted order as the input shapefile.
-   **Zarr Coordinate Indexing (CRITICAL):** It is **critical** that the `otaz` (origin TAZ) and `dtaz` (destination TAZ) coordinate arrays within the generated Zarr file are **0-based contiguous integers** (i.e., `[0, 1, 2, ..., N-1]`, where N is the total number of zones). This ensures compatibility with ActivitySim's expectations and the PILATES postprocessor's validation. The `ActivitySimZarrWriter` must be configured to produce these 0-based coordinates, even if the original canonical zone IDs are 1-based.
-   **Metadata:** It writes this list of original, sorted zone IDs as a metadata attribute (`original_zone_ids`) in the root of the Zarr file. This embeds the "ground truth" ordering directly into the skim artifact.

### Step 6: Verification in the BEAM Postprocessor

After the BEAM run is complete, the BEAM postprocessor performs a critical verification step before merging the new skims.

-   **`verify_skim_zone_order`:** This function is called automatically during the `_merge_beam_skims_to_zarr` process.
-   **Action:** It performs two key checks:
    1.  It compares the `original_zone_ids` attribute from the new BEAM Zarr skims against the canonical order loaded via `load_canonical_zones`.
    2.  **It verifies that the `otaz` and `dtaz` coordinate arrays within the Zarr file are 0-based contiguous integers.** This is a crucial check to ensure that the skims are correctly indexed for downstream models like ActivitySim.
-   **Safeguard:** If any of these checks fail (e.g., orders do not match, or `otaz`/`dtaz` are not 0-based), it raises a `ValueError` and stops the simulation, preventing misaligned or incorrectly indexed data from being used.

### Step 7: Preparing Inputs for UrbanSim

The UrbanSim preprocessor ensures that any skim data it prepares for the UrbanSim model is aligned with the canonical order.

-   **Skim Processing:** The `_load_raw_skims` function now uses `load_canonical_zones` to get the authoritative zone order.
-   **Alignment:** When it reads a skim matrix from an OMX or Zarr file, it uses this canonical order to construct the DataFrame's index and columns. This guarantees that the skim data passed to UrbanSim is explicitly aligned, rather than trusting the internal (and potentially incorrect) mapping of the source file.

### Step 8: Preparing Inputs for ATLAS

The ATLAS preprocessor, when calculating accessibility measures, now fully aligns with the canonical workflow.

-   **Dynamic Mapping:** The `compute_accessibility` function no longer uses a hardcoded `geoid_to_zone.csv` file. It now calls `get_block_to_zone_mapping` to receive a dynamic mapping based on the authoritative zone definitions.
-   **Skim Alignment:** The function also retrieves the canonical zone order via `load_canonical_zones`. When it reads the skim matrices, it wraps them in a pandas DataFrame using the canonical order for the index and columns, ensuring all subsequent calculations are correctly aligned.

### Step 9: Final Model Execution

-   **Execution:** All models (ActivitySim, UrbanSim, ATLAS) now receive input data (land use tables, skims, accessibility measures) that is guaranteed to be aligned to the same canonical zone order.
-   **Result:** This prevents data mismatch errors, improves simulation stability, and ensures the correctness of model results.

## 3. Summary for Developers

-   **Single Source of Truth:** The zone geometry file defined in `settings.yaml` is the master record. All zone-related logic starts here.
-   **Authoritative Function:** Always use `pilates.utils.zone_utils.load_canonical_zones` to get the list of zones. This guarantees you have a validated and canonically sorted list. The older `get_canonical_zones` is deprecated.
-   **Data Alignment:** The system is designed to produce aligned `land_use` and `skim` artifacts from the start for all models. Runtime re-indexing in ActivitySim should not be necessary.
-   **Zarr Coordinate Indexing:** When generating Zarr skims (e.g., in BEAM's `ActivitySimZarrWriter`), the `otaz` and `dtaz` coordinate arrays **must be 0-based contiguous integers**. This is crucial for compatibility and is explicitly validated by the PILATES postprocessor.
-   **Verification is Key:** The `verify_skim_zone_order` check in the BEAM postprocessor acts as a critical safeguard to catch any unexpected changes or bugs in BEAM's output ordering, including incorrect Zarr coordinate indexing.
-   **Consistency:** The canonical ID column from the source file is consistently used (and renamed to `activitysim_index_col`, e.g., "TAZ") across all models to refer to a zone.
