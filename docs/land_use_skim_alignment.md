# Land Use Index and Skim Alignment in ActivitySim

## Overview
This document explains **how ActivitySim determines the ordering of zones in the `land_use` table** and **how skim matrices are aligned** to that ordering.  The goal is to make clear:

1. What the code does when reading the land‑use file.
2. How (and when) the zone IDs are recoded to a contiguous zero‑based index.
3. Where the mapping between original IDs and the recoded index is stored.
4. How the skim loader checks the alignment and re‑indexes the skim datasets if necessary.
5. What happens when the land‑use file is 1‑based but *out of order*.

The explanation references the relevant source files:
- `activitysim/core/input.py`
- `activitysim/core/skim_dataset.py`

---

## 1. Loading the `land_use` Table (`input.py`)

### 1.1 Canonical Index Determination
```python
# activitysim/core/input.py

def canonical_table_index_name(table_name):
    from activitysim.abm.models.util import canonical_ids
    table_index_names = canonical_ids.CANONICAL_TABLE_INDEX_NAMES
    return table_index_names and table_index_names.get(table_name, None)
```
- For the **land_use** table the canonical index name is **"TAZ"**.
- `read_input_table()` obtains the table definition from the settings file (`input_table_list`).
- `read_from_table_info()` then decides which column to use as the DataFrame index:
  * If `index_col` is explicitly set in the settings it **must** equal the canonical name.
  * Otherwise `canonical_table_index_name(tablename)` is used.

### 1.2 Setting the Index
```python
# activitysim/core/input.py (excerpt)
if index_col is not None:
    if index_col in df.columns:
        df.set_index(index_col, inplace=True)
```
- The CSV (or parquet/HDF5) file is read **as‑is** – no sorting occurs.
- The DataFrame’s index now contains the raw zone identifiers exactly in the order the rows appeared in the file.

### 1.3 Optional Recoding (Zero‑Based Contiguous IDs)
If the configuration asks for recoding (`recode_columns` contains `land_use.TAZ`), the following block runs:
```python
remapper = {j: i for i, j in enumerate(sorted(set(df[colname]))) }
df[f"_original_{colname}"] = df[colname]
df[colname] = df[colname].apply(remapper.get)
```
- `sorted(set(...))` forces the **original IDs to be sorted** before mapping.
- The *new* column (`TAZ`) now holds a **contiguous 0‑based integer index**.
- The original IDs are preserved in a new column named `_original_TAZ`.
- A flag `state.settings.offset_preprocessing` is turned on for legacy skim handling.

**Result after recoding**
| Column | Meaning |
|--------|----------|
| `TAZ` (index) | Zero‑based contiguous IDs (0, 1, 2 …). |
| `_original_TAZ` | Original (possibly 1‑based, possibly out‑of‑order) IDs. |

If recoding is **not** requested, the DataFrame retains the original IDs as its index and **no `_original_…` column** is created.

---

## 2. Building the Remapper for Skim Loading (`skim_dataset.py`)
When the skim dataset is loaded, ActivitySim needs to know how to map **original zone IDs** to the **contiguous index** used by the land‑use DataFrame.
```python
# activitysim/core/skim_dataset.py (excerpt)
if f"_original_{land_use.index.name}" in land_use:
    land_use_zone_ids = land_use[f"_original_{land_use.index.name}"]
    remapper = dict(zip(land_use_zone_ids, land_use_zone_ids.index))
else:
    remapper = None
```
- If the `_original_<index>` column exists (i.e., recoding happened), `remapper` becomes a **dictionary**:
  - **Key** = original zone ID (e.g., 23, 5, 12 …).
  - **Value** = the new zero‑based index position (0, 1, 2 …).
- If the column does not exist, `remapper` stays `None` – the land‑use index already contains the IDs used by the skims.

---

## 3. Loading Skim Matrices (`skim_dataset.py`)
### 3.1 Reading OMX / ZARR Files
```python
omx_file_handles = [openmatrix.open_file(f, mode="r") for f in omx_file_paths]

# Create a sharrow Dataset from the OMX files
d = sh.dataset.from_omx_3d(
    omx_file_handles,
    index_names=("otaz", "dtaz", "time_period"),
    time_periods=time_periods,
    max_float_precision=max_float_precision,
    ignore=state.settings.omx_ignore_patterns,
)
```
- The dimensions are named `otaz` (origin TAZ) and `dtaz` (destination TAZ).
- The values in these dimensions **directly come from the OMX file**; they are the **canonical zone IDs** that the modeler used when creating the skims (normally the same IDs appearing in the land‑use CSV).

### 3.2 Aligning Skims with the Land‑Use Index
After the skim dataset is created, ActivitySim verifies that the skim axes line up with the land‑use table.
```python
if d["otaz"].attrs.get("preprocessed") != "zero-based-contiguous":
    try:
        np.testing.assert_array_equal(land_use_zone_id, d.otaz)
    except AssertionError:
        logger.info(f"otaz realignment required\n{err}")
        d = d.reindex(otaz=land_use_zone_id)
        dask_required = True
    d["otaz"] = land_use.index.to_numpy()
    d["otaz"].attrs["preprocessed"] = "zero-based-contiguous"
```
- `land_use_zone_id` is obtained **either** from the `_original_TAZ` column **or** directly from the index when that column does not exist.
- If the skim’s origin axis does **not** already contain a zero‑based‑contiguous index, the code:
  1. **Compares** the two arrays (`land_use_zone_id` vs. `d.otaz`).
  2. **Re‑indexes** the skim with `d.reindex(otaz=land_use_zone_id)` to reorder the rows to match the land‑use order.
  3. **Replaces** the `otaz` coordinate with `land_use.index.to_numpy()`, turning it into a contiguous zero‑based index.
- The same process is repeated for the destination axis (`dtaz`).
- After this block finishes, the skim dimensions **exactly match** the land‑use DataFrame index (both length and order).

---

## 4. What Happens When the Land‑Use File Is 1‑Based but Out‑of‑Order?
### Scenario A – **No recoding** (default)
1. **Reading** – The DataFrame index holds the original TAZ values **in the file order** (e.g., `[23, 5, 12, …]`).
2. **Remapper** – No `_original_TAZ` column → `remapper = None`.
3. **Skim Load** – Skim dimensions contain the same original IDs (the OMX creator used the same IDs).
4. **Alignment** – The check in `load_skim_dataset_to_shared_memory()` compares the land‑use IDs (`land_use_zone_id = land_use.index`) with the skim axis. If the order differs, `d.reindex(otaz=land_use_zone_id)` re‑orders the skim rows to **match the exact order of the land‑use index**.
5. **Result** – The skim matrix is reordered to the out‑of‑order layout; the mapping is implicit – the *position* of a zone in the DataFrame is its index in the skim.

### Scenario B – **Recoding enabled** (zero‑based contiguous)
1. **Reading** – Same as above, but then the recoding block runs.
2. **Sorting** – The `sorted(set(df[colname]))` call sorts the unique TAZ IDs, producing a *canonical sorted list*.
3. **Mapping** – `remapper` maps each original TAZ to a new contiguous index (0, 1, 2 …). The original order is **lost** in the index but saved in `_original_TAZ`.
4. **Skim Load** – The `_original_TAZ` column exists, so a `remapper` dict is built (`{orig → new_index}`).
5. **Alignment** – The code uses `land_use_zone_id = land_use['_original_TAZ']` for comparison, then re‑indexes skims to that order, and finally sets `otaz`/`dtaz` to the new contiguous index.
6. **Result** – The final skim axes are **sorted** (0‑based contiguous) regardless of the original file order. The original IDs are still accessible via `_original_TAZ`.

---

## 5. Where Is the Mapping Stored?
| Artifact | Content |
|----------|----------|
| `land_use` DataFrame index | After loading (no recoding) – the original TAZ IDs in the file order. After recoding – **contiguous 0‑based IDs**.
| `_original_TAZ` column (if recoding) | The **original** zone IDs, preserving their values even if the index is sorted.
| `remapper` dictionary (temporary) | Built in `load_skim_dataset_to_shared_memory()`; maps **original → new index**. Used only while loading skims.
| Skim dataset (`d`) – dimensions `otaz` / `dtaz` | After alignment they hold the **same contiguous index** as `land_use.index`. Their original values are overwritten, but the original ordering is reflected in the earlier re‑index step.

---

## 6. Summary Checklist
- **Read land‑use** → index set to the column named by the canonical index (`TAZ`). No sorting.
- **Optional recoding** → sorted unique IDs → new contiguous index + `_original_TAZ` column.
- **Create remapper** → `{original_TAZ: new_index}` if `_original_TAZ` exists.
- **Load skim** → raw OMX axes contain original IDs.
- **Align skims** → compare skim axes with `land_use` IDs, re‑index skims if they differ, then replace skim axes with the contiguous land‑use index.
- **Result** – All downstream look‑ups (`skim_dataset.otaz[orig, dest]`) are guaranteed to reference the same zone ordering as the land‑use table.

---

## 7. References
- `activitysim/core/input.py` – reading and recoding of input tables.
- `activitysim/core/skim_dataset.py` – loading skims, building remapper, and alignment logic.
- `activitysim/abm/models/util/canonical_ids.py` – defines `CANONICAL_TABLE_INDEX_NAMES`.

---

*This document is intended for developers and modelers working with ActivitySim who need to understand how zone identifiers are handled and how skim matrices stay in sync with the land‑use data.*