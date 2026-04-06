from __future__ import annotations

import logging
import os
import shutil
import glob
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace

import numpy as np
import pandas as pd

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord, sanitize_artifact_key
from pilates.atlas.inputs import atlas_selected_scenario, atlas_static_input_relpaths
from pilates.atlas.outputs import AtlasPreprocessOutputs
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.utils.path_utils import find_project_root
from pilates.utils.settings_helper import get as get_setting

logger = logging.getLogger(__name__)

_ATLAS_RESTART_START_YEAR_CSVS: Tuple[str, ...] = (
    "households.csv",
    "blocks.csv",
    "persons.csv",
    "residential.csv",
    "jobs.csv",
)
_ATLAS_RESTART_PRIOR_SUBYEAR_CSVS: Tuple[str, ...] = (
    "households.csv",
    "blocks.csv",
    "persons.csv",
    "grave.csv",
    "residential.csv",
    "jobs.csv",
)
_ATLAS_RESTART_PRIOR_SUBYEAR_RDATA: Tuple[str, ...] = (
    "vehicles_output.RData",
    "households_output.RData",
)


def _get_usim_datastore_fname(settings, io, year=None):
    # reference: asim postprocessor
    if io == "output":
        datastore_name = get_setting(settings, "urbansim.output_file_template").format(
            year=year
        )
    elif io == "input":
        region = get_setting(settings, "run.region")
        region_id = get_setting(
            settings, "urbansim.region_mappings.region_to_region_id"
        )[region]
        usim_base_fname = get_setting(settings, "urbansim.input_file_template")
        datastore_name = usim_base_fname.format(region_id=region_id)

    return datastore_name


def _prepare_atlas_table_for_export(
    table_data: pd.DataFrame,
    *,
    table_name_in_h5: str,
    expected_index_name: str,
) -> pd.DataFrame:
    """
    Ensure Atlas export tables have the expected logical identifier as index.

    UrbanSim often stores key identifiers (e.g., household_id, block_id) as the
    DataFrame index rather than a normal column. Atlas CSV exports rely on that
    logical key being present and named consistently.
    """
    if table_data.index.name == expected_index_name:
        return table_data

    if expected_index_name in table_data.columns:
        logger.warning(
            "[AtlasPreprocessor] Table %s uses %s as a column; promoting it to index "
            "for CSV export.",
            table_name_in_h5,
            expected_index_name,
        )
        return table_data.set_index(expected_index_name, drop=True)

    raise ValueError(
        f"ATLAS table {table_name_in_h5!r} is missing expected logical key "
        f"{expected_index_name!r} as index or column."
    )


def _export_atlas_table_to_csv(
    table_data: pd.DataFrame,
    *,
    table_name_in_h5: str,
    expected_index_name: str,
    output_csv_path: str,
) -> None:
    """
    Export a table to CSV with explicit, validated index semantics.
    """
    prepared = _prepare_atlas_table_for_export(
        table_data,
        table_name_in_h5=table_name_in_h5,
        expected_index_name=expected_index_name,
    )
    # Keep identifier index in output explicitly so downstream schema alignment
    # is deterministic and does not depend on pandas defaults.
    prepared.to_csv(
        output_csv_path,
        index=True,
        index_label=expected_index_name,
    )


def _first_existing_path(*paths: Optional[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _resolve_existing_artifact_path(
    value: Any, *, workspace: "Workspace"
) -> Optional[str]:
    return artifact_to_existing_path(value, workspace=workspace)


def _restart_required_atlas_input_years(
    *,
    start_year: int,
    atlas_year: int,
) -> List[int]:
    """
    Return the minimal ATLAS year-input directories required on restart.

    Dynamic ATLAS subyear runs rely on the workflow start-year seed inputs and,
    for later subyears, the immediately preceding ATLAS evolution-year inputs.
    """
    required_years = [start_year]
    if atlas_year > start_year:
        prior_subyear = atlas_year - 2
        if prior_subyear >= start_year and prior_subyear not in required_years:
            required_years.append(prior_subyear)
    return required_years


def restart_required_atlas_input_paths(
    *,
    atlas_input_root: str,
    start_year: int,
    atlas_year: int,
) -> Dict[str, str]:
    """
    Return the restart-critical ATLAS files that should exist locally.

    Native restart recovery should hydrate concrete files rather than treating
    whole year directories as first-class artifacts.
    """
    atlas_input_root = os.path.realpath(atlas_input_root)
    required: Dict[str, str] = {}

    start_year_dir = os.path.join(atlas_input_root, f"year{start_year}")
    for filename in _ATLAS_RESTART_START_YEAR_CSVS:
        artifact_name = filename.rsplit(".", 1)[0]
        required[f"atlas_restart_seed::{start_year}::{artifact_name}"] = os.path.join(
            start_year_dir,
            filename,
        )

    if atlas_year > start_year:
        prior_subyear = atlas_year - 2
        if prior_subyear >= start_year:
            prior_year_dir = os.path.join(atlas_input_root, f"year{prior_subyear}")
            for filename in _ATLAS_RESTART_PRIOR_SUBYEAR_CSVS:
                artifact_name = filename.rsplit(".", 1)[0]
                required[
                    f"atlas_restart_prior::{prior_subyear}::{artifact_name}"
                ] = os.path.join(prior_year_dir, filename)
            for filename in _ATLAS_RESTART_PRIOR_SUBYEAR_RDATA:
                artifact_name = filename.replace(".", "_")
                required[
                    f"atlas_restart_prior::{prior_subyear}::{artifact_name}"
                ] = os.path.join(prior_year_dir, filename)

    return required


def _restore_restart_atlas_year_inputs(
    *,
    previous_run_dir: str,
    workspace: "Workspace",
    start_year: int,
    atlas_year: int,
) -> None:
    """
    Rehydrate restart-critical ATLAS year directories from the previous run.

    For reruns of later ATLAS subyears, restoring only the workflow start year
    is insufficient. The dynamic container also expects the immediately
    preceding ATLAS subyear input directory to exist.
    """
    required_paths = restart_required_atlas_input_paths(
        atlas_input_root=workspace.get_atlas_mutable_input_dir(),
        start_year=start_year,
        atlas_year=atlas_year,
    )
    missing_paths = [
        path for path in required_paths.values() if not os.path.exists(path)
    ]
    if not missing_paths:
        return

    for required_year in _restart_required_atlas_input_years(
        start_year=start_year,
        atlas_year=atlas_year,
    ):
        old_year_input_path = os.path.join(
            previous_run_dir, "atlas", "atlas_input", f"year{required_year}"
        )
        new_year_input_path = os.path.join(
            workspace.get_atlas_mutable_input_dir(), f"year{required_year}"
        )
        year_missing_paths = [
            path
            for key, path in required_paths.items()
            if f"::{required_year}::" in key and not os.path.exists(path)
        ]
        if not year_missing_paths:
            continue
        if not os.path.exists(old_year_input_path):
            logger.warning(
                "[AtlasPreprocessor] Restart requires prior ATLAS input directory "
                "year%s, but it was missing from previous run: %s",
                required_year,
                old_year_input_path,
            )
            continue
        logger.info(
            "[AtlasPreprocessor] Copying restart-required ATLAS inputs from previous run: %s",
            old_year_input_path,
        )
        shutil.copytree(
            old_year_input_path,
            new_year_input_path,
            dirs_exist_ok=True,
            symlinks=True,
        )


def _record_restart_chained_rdata_inputs(
    *,
    prepared_inputs: Dict[str, Path],
    atlas_input_root: str,
    start_year: int,
    atlas_year: int,
) -> None:
    if atlas_year <= start_year:
        return

    prior_subyear = atlas_year - 2
    if prior_subyear < start_year:
        return

    prior_year_dir = os.path.join(atlas_input_root, f"year{prior_subyear}")
    for filename in _ATLAS_RESTART_PRIOR_SUBYEAR_RDATA:
        path = os.path.join(prior_year_dir, filename)
        if not os.path.exists(path):
            continue
        artifact_name = filename.replace(".", "_")
        prepared_inputs[
            f"atlas_restart_prior::{prior_subyear}::{artifact_name}"
        ] = Path(path)


def _discover_global_atlas_input_files(global_source_dir: str) -> List[Tuple[str, str]]:
    """
    Discover top-level static ATLAS global inputs that must be present in mutable input.

    Returns tuples of (absolute_source_path, label_for_logging).
    """
    patterns = (
        ("*.csv", "CSV"),
        ("*.RData", "RData"),
        ("*.Rdat", "Rdat"),
        ("*.rdata", "RData"),
        ("*.rdat", "Rdat"),
    )
    discovered: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for pattern, label in patterns:
        for path in glob.glob(os.path.join(global_source_dir, pattern)):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            discovered.append((real, label))
    return discovered


def _atlas_static_input_metadata(
    *,
    relpath: str,
    settings,
    source_origin: str,
    source_path: str,
) -> Dict[str, object]:
    normalized_relpath = relpath.replace("\\", "/")
    filename = os.path.basename(normalized_relpath)
    input_group = "global"
    selected_scenario = atlas_selected_scenario(settings)
    input_year = None
    compact_key = normalized_relpath.replace("/", "_")
    compact_stem = os.path.splitext(compact_key)[0]

    if normalized_relpath.startswith("adopt/"):
        input_group = "adopt"
        parts = normalized_relpath.split("/")
        if len(parts) >= 2:
            selected_scenario = parts[1]
    elif compact_key.startswith("vehicle_type_mapping_"):
        input_group = "vehicle_type_mapping"
        if "baseline" in compact_key:
            selected_scenario = "baseline"
        elif "evMandForced2" in compact_key:
            selected_scenario = "zev_mandate"
        elif "ESS_const_220_price" in compact_key:
            selected_scenario = "ess_cons"

    tail = compact_stem.rsplit("_", 1)
    if len(tail) == 2 and len(tail[1]) == 4 and tail[1].isdigit():
        input_year = int(tail[1])

    metadata: Dict[str, object] = {
        "atlas_static_input": True,
        "atlas_relpath": normalized_relpath,
        "atlas_source_origin": source_origin,
        "atlas_source_path": os.path.realpath(source_path),
        "atlas_input_group": input_group,
    }
    if selected_scenario:
        metadata["atlas_scenario"] = selected_scenario
    if input_year is not None:
        metadata["atlas_input_year"] = input_year
    if filename.lower().endswith(".csv"):
        metadata["profile_file_schema"] = True
    return metadata


def _resolve_atlas_h5_table_key(
    store: pd.HDFStore, *, year: int, table: str, is_start_year: bool
) -> str:
    """
    Resolve the HDF5 key for an ATLAS-required table.

    Prefer ``/{year}/{table}`` when present, regardless of whether the current
    ATLAS sub-year is the first sub-year in the interval. Some start-subyear
    UrbanSim snapshots are still year-scoped (for example ``/2023/households``
    inside ``model_data_2023.h5``) rather than root-scoped.

    Fall back to root-level ``{table}`` to support merged/current datastores
    that do not keep per-year table prefixes.
    """
    year_key = f"/{year}/{table}"
    if year_key in store:
        return year_key

    if table in store:
        if is_start_year:
            return table
        logger.warning(
            "[AtlasPreprocessor] Year-specific table %s not found; falling back "
            "to root table %s.",
            year_key,
            table,
        )
        return table

    # Some UrbanSim outputs may carry only year-scoped tables (e.g. /2023/*)
    # without root aliases. For ATLAS subyear runs, fall back to the nearest
    # available year-scoped table for this table name.
    suffix = f"/{table}"
    year_scoped_candidates: list[tuple[int, str]] = []
    for key in store.keys():
        if not key.endswith(suffix):
            continue
        parts = key.strip("/").split("/")
        if len(parts) != 2:
            continue
        year_token, table_token = parts
        if table_token != table or not year_token.isdigit():
            continue
        year_scoped_candidates.append((int(year_token), key))

    if year_scoped_candidates:
        prior_or_equal = [entry for entry in year_scoped_candidates if entry[0] <= year]
        if prior_or_equal:
            selected_year, selected_key = max(prior_or_equal, key=lambda x: x[0])
        else:
            selected_year, selected_key = min(year_scoped_candidates, key=lambda x: x[0])
        logger.warning(
            "[AtlasPreprocessor] Year-specific table %s and root table %s were missing; "
            "falling back to nearest available year-scoped table %s (year=%s).",
            year_key,
            table,
            selected_key,
            selected_year,
        )
        return selected_key

    return year_key


def _resolve_atlas_static_sources(
    settings,
) -> Tuple[str, str]:
    source_dir = get_setting(
        settings, "atlas.host_input_folder", "pilates/atlas/atlas_input"
    )
    project_root = find_project_root(start_path=os.path.dirname(__file__))
    if not project_root:
        project_root = os.path.realpath(os.getcwd())
        logger.warning(
            "[NOT IDEAL] Could not locate PILATES project root via markers; "
            "falling back to cwd='%s'.",
            project_root,
        )

    if not os.path.isabs(source_dir):
        source_dir = os.path.join(project_root, source_dir)
    source_dir = os.path.realpath(source_dir)

    default_source_dir = os.path.realpath(
        os.path.join(project_root, "pilates/atlas/atlas_input")
    )
    return source_dir, default_source_dir


def _stage_atlas_static_inputs(
    *,
    settings,
    output_dir: str,
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, object]]]:
    source_dir, default_source_dir = _resolve_atlas_static_sources(settings)
    source_dirs = [source_dir]
    if default_source_dir not in source_dirs:
        source_dirs.append(default_source_dir)

    scenario = get_setting(settings, "atlas.scenario")
    adscen = get_setting(settings, "atlas.adscen")
    selected_scenario = atlas_selected_scenario(settings)
    scenario_key = selected_scenario or (str(scenario).lower() if scenario else "")
    if scenario and scenario_key not in {"baseline", "ess_cons", "zev_mandate"}:
        logger.warning(
            "[AtlasPreprocessor] Unknown atlas.scenario=%s; using deterministic static input fallback.",
            scenario,
        )
    if adscen and scenario and adscen != scenario:
        logger.warning(
            "[AtlasPreprocessor] atlas.adscen=%s differs from atlas.scenario=%s; "
            "using atlas.adscen for input selection to match runner behavior.",
            adscen,
            scenario,
        )

    required_relpaths = atlas_static_input_relpaths(settings)
    logger.info(
        "[AtlasPreprocessor] Copying ATLAS static files to mutable input "
        "(primary=%s fallback=%s)",
        source_dir,
        default_source_dir,
    )

    staged_paths: Dict[str, Path] = {}
    metadata_by_key: Dict[str, Dict[str, object]] = {}
    missing_required_relpaths: List[str] = []
    fallback_copy_count = 0

    for relpath in required_relpaths:
        normalized_relpath = relpath.replace("\\", "/")
        source_path = None
        source_base = None
        for base_dir in source_dirs:
            candidate = os.path.realpath(os.path.join(base_dir, normalized_relpath))
            if os.path.exists(candidate):
                source_path = candidate
                source_base = base_dir
                break

        if source_path is None:
            missing_required_relpaths.append(normalized_relpath)
            continue

        if source_base is not None and source_base != source_dir:
            fallback_copy_count += 1
            logger.warning(
                "[AtlasPreprocessor] Required file missing in primary source, "
                "using fallback: %s",
                normalized_relpath,
            )

        dest_path = os.path.realpath(os.path.join(output_dir, normalized_relpath))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)

        rel_no_ext = os.path.splitext(normalized_relpath)[0]
        rel_key = rel_no_ext
        short_name = sanitize_artifact_key(rel_key) or rel_key
        staged_paths[short_name] = Path(dest_path)
        metadata_by_key[short_name] = _atlas_static_input_metadata(
            relpath=normalized_relpath,
            settings=settings,
            source_origin=(
                "fallback"
                if source_base is not None and source_base != source_dir
                else "primary"
            ),
            source_path=source_path,
        )

    if missing_required_relpaths:
        preview = ", ".join(sorted(missing_required_relpaths)[:8])
        raise RuntimeError(
            "Missing required ATLAS static input files "
            f"(count={len(missing_required_relpaths)}). "
            f"Preview: {preview}"
        )

    if fallback_copy_count:
        logger.warning(
            "[AtlasPreprocessor] Copied %s files via fallback source.",
            fallback_copy_count,
        )

    logger.info(
        "[AtlasPreprocessor] Finished copying %s static ATLAS inputs.",
        len(staged_paths),
    )
    return staged_paths, metadata_by_key


class AtlasPreprocessor(GenericPreprocessor):
    """
    ATLAS-specific preprocessor that consolidates all preprocessing steps for the ATLAS vehicle ownership model.
    This includes extracting UrbanSim outputs, formatting them as ATLAS inputs, and (optionally) calculating accessibility
    using BEAM skims.
    """

    @staticmethod
    def declared_expected_inputs(
        settings: "PilatesConfig", state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this preprocessor expects without disk checks.
        """
        usim_input_fname = _get_usim_datastore_fname(
            settings,
            io="input" if state.is_start_year() else "output",
            year=state.forecast_year,
        )
        return {
            "atlas_mutable_input_dir": workspace.get_atlas_mutable_input_dir(),
            "usim_datastore_h5": os.path.join(
                workspace.get_usim_mutable_data_dir(), usim_input_fname
            ),
        }

    @staticmethod
    def runtime_expected_inputs(
        settings: "PilatesConfig", state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare runtime expected inputs, including filesystem presence checks.
        """
        usim_input_fname = _get_usim_datastore_fname(
            settings,
            io="input" if state.is_start_year() else "output",
            year=state.forecast_year,
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "atlas_mutable_input_dir": workspace.get_atlas_mutable_input_dir(),
            "usim_datastore_h5": (
                usim_input_path if os.path.exists(usim_input_path) else None
            ),
        }

    @staticmethod
    def expected_inputs(
        settings: "PilatesConfig", state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        return AtlasPreprocessor.runtime_expected_inputs(settings, state, workspace)

    @staticmethod
    def expected_outputs(
        settings: "PilatesConfig", state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this preprocessor produces.

        Notes
        -----
        Output keys
            - ``atlas_mutable_input_dir``: ATLAS mutable input directory with
              prepared configs and data.
        Related docs
            - See `pilates/atlas/inputs.py` for the corresponding input
              descriptions used by ATLAS and downstream models.
        """
        return {
            "atlas_mutable_input_dir": workspace.get_atlas_mutable_input_dir(),
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_data = ["usim_datastore_h5"]

    def copy_data_to_mutable_location(
        self,
        settings,
        output_dir,
        workspace: Optional["Workspace"] = None,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy ATLAS input files from the production directory to the run's mutable input directory.
        """
        staged_paths, metadata_by_key = _stage_atlas_static_inputs(
            settings=settings,
            output_dir=output_dir,
        )
        input_records = []
        output_records = []
        for short_name, dest_path in staged_paths.items():
            source_path = metadata_by_key[short_name]["atlas_source_path"]
            filename = os.path.basename(str(dest_path))
            input_records.append(
                FileRecord(
                    file_path=str(source_path),
                    description=f"ATLAS input file: {filename}",
                    short_name=short_name,
                    metadata=metadata_by_key[short_name],
                )
            )
            output_records.append(
                FileRecord(
                    file_path=str(dest_path),
                    description=f"Mutable ATLAS input file: {filename}",
                    short_name=short_name,
                )
            )
        return RecordStore(recordList=input_records), RecordStore(recordList=output_records)

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
        final_skims_omx: Optional[Any] = None,
    ) -> AtlasPreprocessOutputs:
        """
        Prepares all data needed to run ATLAS, including extracting UrbanSim outputs
        and formatting them as ATLAS inputs. Handles provenance tracking.

        Steps:
        1. Collect required input paths (UrbanSim HDF5, BEAM skims if needed).
        2. Extract UrbanSim HDF5 tables and write them as CSVs for ATLAS.
        3. If enabled, compute accessibility using BEAM skims.
        """
        logger.info("[AtlasPreprocessor] Starting preprocessing for ATLAS.")
        settings = self.state.full_settings
        prepared_inputs, prepared_input_meta = _stage_atlas_static_inputs(
            settings=settings,
            output_dir=workspace.get_atlas_mutable_input_dir(),
        )

        # --- Ensure global ATLAS input files are present for every year ---
        # Source for global files (e.g., cpi.csv, RData/Rdat files)
        global_source_dir = "pilates/atlas/atlas_input"
        if not os.path.isabs(global_source_dir):
            project_root = find_project_root(start_path=os.path.dirname(__file__))
            if not project_root:
                project_root = os.path.realpath(os.getcwd())
                logger.warning(
                    "[NOT IDEAL] Could not locate PILATES project root via markers; "
                    "falling back to cwd='%s'.",
                    project_root,
                )
            global_source_dir = os.path.join(project_root, global_source_dir)

        # Destination for global files in the current run's mutable directory
        current_atlas_mutable_input_root = workspace.get_atlas_mutable_input_dir()

        # Copy global top-level ATLAS files, including legacy *.Rdat.
        for f, label in _discover_global_atlas_input_files(global_source_dir):
            dest_path = os.path.realpath(
                os.path.join(current_atlas_mutable_input_root, os.path.basename(f))
            )
            if not os.path.exists(dest_path):
                shutil.copy2(f, dest_path)
                logger.info(
                    f"[AtlasPreprocessor] Copied global {label} file: {f} to {dest_path}"
                )
            else:
                logger.debug(
                    f"[AtlasPreprocessor] Global {label} file already exists: {dest_path}"
                )

        # --- End Global File Handling ---

        # --- Restart Logic for year-specific data ---
        is_restart_run = getattr(self.state, "is_restart_run", None)
        if is_restart_run is None:
            is_restart_run = bool(self.state.run_info_path)
        if (
            is_restart_run
            and self.state.run_info_path
            and os.path.exists(self.state.run_info_path)
        ):
            # This is a restarted run
            previous_run_dir = os.path.dirname(self.state.run_info_path)
            logger.info(
                f"[AtlasPreprocessor] Restarted run detected. Using previous run's output path from {previous_run_dir}"
            )

            # 1. Copy restart-critical ATLAS year inputs from previous run.
            _restore_restart_atlas_year_inputs(
                previous_run_dir=previous_run_dir,
                workspace=workspace,
                start_year=self.state.start_year,
                atlas_year=self.state.year,
            )

            # 2. Set path for UrbanSim output
            urbansim_output_path = os.path.join(previous_run_dir, "urbansim", "data")
        else:
            # This is a fresh run
            urbansim_output_path = workspace.get_usim_mutable_data_dir()

        artifact_current_h5 = getattr(self.state, "atlas_usim_datastore_h5", None)
        artifact_base_h5 = getattr(self.state, "atlas_usim_datastore_base_h5", None)
        current_h5_path = _resolve_existing_artifact_path(
            artifact_current_h5,
            workspace=workspace,
        )
        base_h5_path = _resolve_existing_artifact_path(
            artifact_base_h5,
            workspace=workspace,
        )
        preferred_h5 = _first_existing_path(
            base_h5_path if self.state.is_start_year() else current_h5_path,
            current_h5_path if self.state.is_start_year() else base_h5_path,
        )

        if preferred_h5 is not None:
            urbansim_output = preferred_h5
        else:
            if self.state.is_start_year():
                urbansim_output_fname = _get_usim_datastore_fname(settings, io="input")
            else:
                urbansim_output_fname = _get_usim_datastore_fname(
                    settings, io="output", year=self.state.forecast_year
                )
            urbansim_output = os.path.join(urbansim_output_path, urbansim_output_fname)
            logger.warning(
                "[AtlasPreprocessor] Falling back to template-resolved UrbanSim H5. "
                "Preferred artifacts were unavailable: current=%s base=%s fallback=%s",
                artifact_current_h5,
                artifact_base_h5,
                urbansim_output,
            )

        atlas_input_path = os.path.join(
            workspace.get_atlas_mutable_input_dir(),
            "year{}".format(self.state.year),
        )
        _record_restart_chained_rdata_inputs(
            prepared_inputs=prepared_inputs,
            atlas_input_root=workspace.get_atlas_mutable_input_dir(),
            start_year=self.state.start_year,
            atlas_year=self.state.year,
        )

        # Record BEAM skims as input if needed
        beamac = settings.atlas.beamac
        if beamac > 0:
            if final_skims_omx is not None and Path(final_skims_omx).exists():
                expected_beam_skims_path = str(Path(final_skims_omx))
                logger.info(
                    "[AtlasPreprocessor] Using explicit final_skims_omx artifact: %s",
                    expected_beam_skims_path,
                )
            else:
                if final_skims_omx is not None:
                    logger.warning(
                        "[AtlasPreprocessor] Explicit final_skims_omx artifact did not "
                        "resolve to an existing path: %s. Falling back to legacy BEAM skims discovery.",
                        final_skims_omx,
                    )
                beam_output_dir = workspace.get_beam_output_dir()
                expected_beam_skims_path = os.path.join(
                    beam_output_dir, settings.shared.skims.fname
                )
            if os.path.exists(expected_beam_skims_path):
                logger.info(
                    f"[AtlasPreprocessor] Recording BEAM skims as input: {expected_beam_skims_path}"
                )
                prepared_inputs["beam_skims_input"] = Path(expected_beam_skims_path)
            else:
                logger.warning(
                    f"[AtlasPreprocessor] BEAM skims file not found: {expected_beam_skims_path}"
                )
        else:
            # FIX ATLAS ISSUE 3: Track .RData accessibility files when NOT using BEAM skims
            logger.info(
                "[AtlasPreprocessor] atlas_beamac=0, looking for .RData accessibility files"
            )
            # Look for .RData files in the root input directory
            year_input_dir = workspace.get_atlas_mutable_input_dir()
            if os.path.exists(year_input_dir):
                rdata_files = glob.glob(os.path.join(year_input_dir, "*.RData"))
                for rdata_file in rdata_files:
                    if "access" in os.path.basename(rdata_file).lower():
                        logger.info(
                            f"[AtlasPreprocessor] Recording accessibility .RData file: {rdata_file}"
                        )
                        prepared_inputs["atlas_rdata_accessibility"] = Path(rdata_file)

        # --- Write ATLAS input CSVs and record as outputs ---
        if not os.path.exists(urbansim_output):
            logger.error("[AtlasPreprocessor] UrbanSim input H5 was not found.")
            raise RuntimeError(
                "ATLAS preprocess cannot continue: UrbanSim input H5 was not found "
                f"for year {self.state.year} (resolved path: {urbansim_output})"
            )

        with pd.HDFStore(urbansim_output, mode="r") as data:

            missing_required_tables: List[str] = []

            def process_table(
                table_name_in_h5,
                output_csv_name,
                output_short_name,
                output_description,
                expected_index_name,
                table_name_root,
                required=True,
            ):
                resolved_table_name = _resolve_atlas_h5_table_key(
                    data,
                    year=self.state.year,
                    table=table_name_root,
                    is_start_year=self.state.is_start_year(),
                )
                try:
                    table_data = data[resolved_table_name]
                    cr.log_h5_table(
                        urbansim_output,
                        key=f"atlas_preprocess_usim_{table_name_root}_table_input",
                        table_path=resolved_table_name,
                        direction="input",
                        description=(
                            f"UrbanSim {table_name_root} table consumed by ATLAS preprocess"
                        ),
                        profile_file_schema=True,
                        h5_table_name=table_name_root,
                    )
                    output_csv_path = f"{atlas_input_path}/{output_csv_name}.csv"
                    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                    _export_atlas_table_to_csv(
                        table_data,
                        table_name_in_h5=resolved_table_name,
                        expected_index_name=expected_index_name,
                        output_csv_path=output_csv_path,
                    )
                    prepared_inputs[output_short_name] = Path(output_csv_path)
                except KeyError:
                    if required:
                        missing_required_tables.append(table_name_in_h5)
                        logger.error(
                            "[AtlasPreprocessor] Required table '%s' not found in HDF5 file: %s",
                            table_name_in_h5,
                            urbansim_output,
                        )
                    else:
                        logger.warning(
                            "[AtlasPreprocessor] Optional table '%s' not found in HDF5 file.",
                            table_name_in_h5,
                        )
                except Exception as e:
                    logger.error(
                        f"[AtlasPreprocessor] Error processing table {table_name_in_h5}: {e}"
                    )

            year_prefix = (
                f"/{self.state.year}" if not self.state.is_start_year() else ""
            )

            process_table(
                f"{year_prefix}/households",
                "households",
                "atlas_households_csv",
                "ATLAS households input CSV",
                "household_id",
                "households",
                required=True,
            )
            process_table(
                f"{year_prefix}/blocks",
                "blocks",
                "atlas_blocks_csv",
                "ATLAS blocks input CSV",
                "block_id",
                "blocks",
                required=True,
            )
            process_table(
                f"{year_prefix}/persons",
                "persons",
                "atlas_persons_csv",
                "ATLAS persons input CSV",
                "person_id",
                "persons",
                required=True,
            )
            # Dynamic ATLAS evolution years need grave.csv whenever the subyear
            # is beyond the global simulation start year, including the first
            # subyear in a later forecast interval (e.g. 2023 in a 2017->2029 run).
            if int(self.state.year) > int(self.state.start_year):
                process_table(
                    f"{year_prefix}/graveyard",
                    "grave",
                    "atlas_grave_csv",
                    "ATLAS graveyard input CSV",
                    "person_id",
                    "graveyard",
                    required=True,
                )
            process_table(
                f"{year_prefix}/residential_units",
                "residential",
                "atlas_residential_csv",
                "ATLAS residential units input CSV",
                "unit_id",
                "residential_units",
                required=True,
            )
            process_table(
                f"{year_prefix}/jobs",
                "jobs",
                "atlas_jobs_csv",
                "ATLAS jobs input CSV",
                "job_id",
                "jobs",
                required=True,
            )

            if missing_required_tables:
                missing_msg = ", ".join(sorted(set(missing_required_tables)))
                raise RuntimeError(
                    "ATLAS preprocess missing required UrbanSim tables for "
                    f"year {self.state.year}: {missing_msg} "
                    f"(source H5: {urbansim_output})"
                )

            logger.info(
                f"[AtlasPreprocessor] Prepared ATLAS Year {self.state.year} input from UrbanSim output."
            )

        # --- Accessibility calculation (BEAM skims) ---
        if beamac > 0:
            logger.info(
                "[AtlasPreprocessor] Calculating accessibility using BEAM skims for ATLAS."
            )
            path_list = [
                "WLK_COM_WLK",
                "WLK_EXP_WLK",
                "WLK_HVY_WLK",
                "WLK_LOC_WLK",
                "WLK_LRF_WLK",
            ]
            measure_list = ["WACC", "IWAIT", "XWAIT", "TOTIVT", "WEGR"]
            # compute_accessibility expects (path_list, measure_list, settings, year)
            compute_accessibility(
                path_list,
                measure_list,
                settings,
                self.state.forecast_year,
                workspace,
            )
            logger.info("[AtlasPreprocessor] Accessibility calculation complete.")

            # Record accessibility output file
            accessibility_csv = "{}/accessibility_{}_tract.csv".format(
                atlas_input_path, self.state.forecast_year
            )
            if os.path.exists(accessibility_csv):
                prepared_inputs["atlas_accessibility_csv"] = Path(accessibility_csv)

        logger.info("[AtlasPreprocessor] ATLAS preprocessing complete.")
        return AtlasPreprocessOutputs(
            atlas_mutable_input_dir=Path(workspace.get_atlas_mutable_input_dir()),
            prepared_inputs=prepared_inputs,
            prepared_input_meta=prepared_input_meta,
        )

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
    ) -> AtlasPreprocessOutputs:
        """Prepare ATLAS inputs and return typed outputs."""
        self.state.set_sub_stage_progress("preprocessor")
        return self._preprocess(workspace, previous_records)


def compute_accessibility(
    path_list, measure_list, settings, year, workspace, threshold=500
):
    # set where to put atlas csv inputs (processed from urbansim outputs)
    atlas_input_path = os.path.join(
        workspace.get_atlas_mutable_input_dir(), f"year{year}"
    )
    os.makedirs(atlas_input_path, exist_ok=True)

    # --- Get Canonical Zone Information ---
    from pilates.utils.zone_utils import (
        load_canonical_zones,
        get_block_to_zone_mapping,
    )

    canonical_zones_df = load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.values

    # for each OD, compute minimum time taken by public transit
    # inf means no public transit available; unit = minute
    ODmatrix_df = _get_time_ODmatrix(
        settings, path_list, measure_list, threshold, workspace, canonical_order
    )

    # assign values = 1 if time taken by public transit <= 30min; 0 if not
    ODmatrix = ODmatrix_df <= 30

    # read and format geoid_to_zoneid mapping list
    mapping = get_block_to_zone_mapping(settings, year, workspace)

    # read in jobs data (keep low_memory=False to solve dtypeerror)
    jobs = pd.read_csv(
        "{}/jobs.csv".format(atlas_input_path),
        low_memory=False,
    )

    # map jobs geoid to zone id in OD matrix
    jobs["zone_id"] = jobs["block_id"].astype(str).map(mapping)

    # Drop jobs that couldn't be mapped to a zone
    jobs.dropna(subset=["zone_id"], inplace=True)

    # count number of jobs for each block_id
    jobs_vector = (
        jobs.groupby("block_id")
        .agg({"job_id": "size", "zone_id": "max"})
        .rename(columns={"job_id": "access_sum"})
    )

    # average # of jobs per block for each taz
    jobs_vector = jobs_vector.groupby("zone_id").agg({"access_sum": "mean"})

    # make sure every zone id has a row in jobs_vector
    jobs_vector = jobs_vector.reindex(canonical_order, fill_value=0)

    # multiply OD matrix (o*d) with jobs vector (d*1)
    # to get number of jobs accessible by public transit within 30min
    accessibility = np.matmul(ODmatrix, jobs_vector)
    accessibility.index.name = "zone_id"

    # # calculate taz-level zscore
    # accessibility['access_zscore'] = (accessibility['access_sum'] - accessibility['access_sum'].mean())/accessibility['access_sum'].std()

    # # write taz-level accessibility data
    # accessibility.to_csv('{}/accessibility_{}_taz.csv'.format(atlas_input_path, year))

    # read in taz_to_tract conversion matrix (1454*1588)
    taz_to_tract = pd.read_csv(
        "{}/taz_to_tract_{}.csv".format(
            settings.atlas.host_input_folder, settings.run.region
        ),
        index_col=0,
    )

    # convert taz- to tract-level accessibility data
    accessibility_tract = np.matmul(
        np.transpose(accessibility), np.array(taz_to_tract.values)
    )
    accessibility_tract.columns = taz_to_tract.columns
    accessibility_tract = accessibility_tract.transpose()

    # calculate tract-level zscore
    accessibility_tract["access_zscore"] = (
        accessibility_tract["access_sum"] - accessibility_tract["access_sum"].mean()
    ) / accessibility_tract["access_sum"].std()

    # format before writing
    accessibility_tract.index.name = "tract"
    accessibility_tract["urban_cbsa"] = 1  ## all sfbay tracts belong to cbsa

    # write tract-level accessibility data
    accessibility_tract.to_csv(
        "{}/accessibility_{}_tract.csv".format(atlas_input_path, year)
    )


def _get_time_ODmatrix(
    settings, path_list, measure_list, threshold, workspace, canonical_order
):
    # open skims file
    import openmatrix as omx

    skims_dir = workspace.get_asim_mutable_data_dir()
    skims = omx.open_file(os.path.join(skims_dir, "skims.omx"), mode="r")

    # find the path with minimum time for each o-d
    ODmatrix = np.ones(skims.shape()) * np.inf

    for path in path_list:
        tmp_path = np.zeros(skims.shape())

        # sum total time taken for each specific path
        for measure in measure_list:
            tmp_measure = np.zeros(skims.shape())

            # extract data from skims.omx
            key = "{}_{}__AM".format(path, measure)
            try:
                tmp_measure = np.array(skims[key])
            except KeyError:
                tmp_measure = np.zeros(skims.shape())
                # logger.error('{} not found in skims'.format(key))

            # sum up time taken for each path
            tmp_path = tmp_path + tmp_measure

        # filter out paths with unreasonable TOTIVT (no available transit)
        tmp_path[tmp_path <= threshold] = 1e6

        # find the path with minimum total time taken
        ODmatrix = np.minimum(ODmatrix, tmp_path)

        # divide by 100 to get minute values before returning
    ODmatrix = ODmatrix / 100

    skims.close()
    return pd.DataFrame(ODmatrix, index=canonical_order, columns=canonical_order)
