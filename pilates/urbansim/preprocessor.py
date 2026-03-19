from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING, Dict, Any
import os
import shutil
import logging
import pandas as pd
import openmatrix as omx
import numpy as np

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs
from pilates.utils.path_utils import find_project_root

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)

skim_dtypes = {
    "timePeriod": str,
    "pathType": str,
    "origin": int,
    "destination": int,
    "TIME_minutes": float,
    "TOTIVT_IVT_minutes": float,
    "VTOLL_FAR": float,
    "DIST_meters": float,
    "WACC_minutes": float,
    "WAUX_minutes": float,
    "WEGR_minutes": float,
    "DTIM_minutes": float,
    "DDIST_meters": float,
    "KEYIVT_minutes": float,
    "FERRYIVT_minutes": float,
    "BOARDS": float,
    "DEBUG_TEXT": str,
}

_OPTIONAL_URBANSIM_INPUT_FILENAMES = (
    ("schools_2010.csv", "schools"),
    ("blocks_school_districts_2010.csv", "school_districts"),
)


def _urbansim_source_data_dir(settings: PilatesConfig) -> Path:
    project_root = find_project_root(start_path=os.path.dirname(__file__))
    if not project_root:
        project_root = os.path.realpath(os.getcwd())
        logger.warning(
            "[NOT IDEAL] Could not locate PILATES project root via markers; falling back to cwd='%s'. "
            "This is error-prone in production and may affect inputs:// vs workspace:// URI labeling.",
            project_root,
        )
    data_dir = (
        settings.urbansim.local_data_input_folder
        if os.path.isabs(settings.urbansim.local_data_input_folder)
        else os.path.join(project_root, settings.urbansim.local_data_input_folder)
    )
    return Path(data_dir)


def _beam_input_root(settings: PilatesConfig) -> Path:
    project_root = find_project_root(start_path=os.path.dirname(__file__))
    if not project_root:
        project_root = os.path.realpath(os.getcwd())
    beam_input_dir = (
        settings.beam.local_input_folder
        if os.path.isabs(settings.beam.local_input_folder)
        else os.path.join(project_root, settings.beam.local_input_folder)
    )
    return Path(beam_input_dir)


def _archive_fallback_path(
    *,
    state: "WorkflowState",
    workspace: "Workspace",
    local_path: Path,
) -> Optional[Path]:
    run_info_path = getattr(state, "run_info_path", None)
    if not run_info_path:
        return None
    archive_run_dir = Path(run_info_path).expanduser().resolve().parent
    local_root = Path(workspace.full_path).expanduser().resolve()
    try:
        rel = local_path.expanduser().resolve().relative_to(local_root)
    except Exception:
        return None
    return archive_run_dir / rel


def _restore_missing_mutable_urbansim_supporting_inputs(
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
) -> Dict[str, Path]:
    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    mutable_dir = Path(workspace.get_usim_mutable_data_dir())
    source_dir = _urbansim_source_data_dir(settings)
    beam_inputs_root = _beam_input_root(settings)

    required_files: Dict[str, Tuple[Path, Path]] = {
        "omx_skims": (
            mutable_dir / f"skims_mpo_{region_id}.omx",
            beam_inputs_root / region / settings.shared.skims.fname,
        ),
        "hh_size": (
            mutable_dir / f"hsize_ct_{region_id}.csv",
            source_dir / f"hsize_ct_{region_id}.csv",
        ),
        "income_rates": (
            mutable_dir / f"income_rates_{region_id}.csv",
            source_dir / f"income_rates_{region_id}.csv",
        ),
        "relmap": (
            mutable_dir / f"relmap_{region_id}.csv",
            source_dir / f"relmap_{region_id}.csv",
        ),
        "schools": (
            mutable_dir / "schools_2010.csv",
            source_dir / "schools_2010.csv",
        ),
        "school_districts": (
            mutable_dir / "blocks_school_districts_2010.csv",
            source_dir / "blocks_school_districts_2010.csv",
        ),
    }

    restored: Dict[str, Path] = {}
    for key, (dest_path, source_path) in required_files.items():
        if dest_path.exists():
            continue

        archive_path = _archive_fallback_path(
            state=state,
            workspace=workspace,
            local_path=dest_path,
        )
        candidate = None
        if archive_path is not None and archive_path.exists():
            candidate = archive_path
        elif source_path.exists():
            candidate = source_path

        if candidate is None:
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, dest_path)
        restored[key] = dest_path
        logger.info(
            "[UrbansimPreprocessor] Restored missing mutable input %s from %s",
            dest_path,
            candidate,
        )

    return restored


def _mutable_urbansim_input_paths(
    settings: PilatesConfig,
    workspace: "Workspace",
) -> Dict[str, Path]:
    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    mutable_dir = Path(workspace.get_usim_mutable_data_dir())

    prepared_inputs: Dict[str, Path] = {}
    candidate_paths = {
        "usim_datastore_h5": mutable_dir
        / settings.urbansim.input_file_template.format(region_id=region_id),
        "omx_skims": mutable_dir / f"skims_mpo_{region_id}.omx",
        "hh_size": mutable_dir / f"hsize_ct_{region_id}.csv",
        "income_rates": mutable_dir / f"income_rates_{region_id}.csv",
        "relmap": mutable_dir / f"relmap_{region_id}.csv",
        "geoid_to_zone": mutable_dir / "geoid_to_zone.csv",
    }
    for filename, key in _OPTIONAL_URBANSIM_INPUT_FILENAMES:
        candidate_paths[key] = mutable_dir / filename

    for key, path in candidate_paths.items():
        if path.exists():
            prepared_inputs[key] = path
    return prepared_inputs


def _load_raw_skims(settings, asim_data_dir, usim_data_dir, skim_format, workspace):
    """
    Load raw skims and format for UrbanSim, ensuring canonical zone order.
    """
    from pilates.utils.zone_utils import load_canonical_zones

    # Get the authoritative, sorted list of zone IDs
    canonical_zones_df = load_canonical_zones(settings, workspace)
    canonical_order = canonical_zones_df.index.values

    skims_fname = settings.shared.skims.fname

    try:
        if skim_format == "beam":
            if skims_fname.endswith("csv"):
                raise NotImplementedError("DEMOS requires skims in omx format, not csv")

            elif skims_fname.endswith("omx"):
                skims_fname = "skims.omx"
                mutable_skims_location = os.path.join(asim_data_dir, skims_fname)
                # This copy seems redundant if the file is already in the right place,
                # but preserving original logic for now.
                region_id = settings.urbansim.region_mappings["region_to_region_id"][
                    settings.run.region
                ]
                input_skims_location = "pilates/urbansim/data/skims_mpo_{0}.omx".format(
                    region_id
                )
                logger.info(
                    "Copying skims from {0} to {1} for urbansim".format(
                        mutable_skims_location, input_skims_location
                    )
                )
                shutil.copyfile(mutable_skims_location, input_skims_location)

                with omx.open_file(mutable_skims_location, "r") as skims:
                    # Use the canonical order for the DataFrame index and columns
                    index = pd.Index(canonical_order, name="from_zone_id", dtype=str)
                    columns = pd.Index(canonical_order, name="to_zone_id", dtype=str)

                    # The skim matrix data is assumed to be in the canonical order
                    # as enforced by the ActivitySim preprocessor.
                    travel_time_mins = np.array(skims["SOV_TIME__AM"])
                    out = (
                        pd.DataFrame(travel_time_mins, index=index, columns=columns)
                        .stack()
                        .rename("SOV_AM_IVT_mins")
                    )
                    return out.to_frame()

            elif skims_fname.endswith("zarr"):
                beam_output_dir = settings.beam.local_output_folder
                skims_fname = settings.shared.skims.fname
                mutable_skims_location = os.path.join(beam_output_dir, skims_fname)
                region_id = settings.urbansim.region_mappings["region_to_region_id"][
                    settings.run.region
                ]
                input_skims_location = os.path.join(
                    usim_data_dir, "skims_mpo_{0}.zarr".format(region_id)
                )
                logger.info(
                    "Copying skims from {0} to {1} for urbansim".format(
                        mutable_skims_location, input_skims_location
                    )
                )

                if os.path.exists(input_skims_location):
                    shutil.rmtree(input_skims_location)
                shutil.copytree(mutable_skims_location, input_skims_location)

                import xarray as xr

                with xr.open_zarr(mutable_skims_location) as ds:
                    # Use the canonical order for the DataFrame index and columns
                    index = pd.Index(canonical_order, name="from_zone_id", dtype=str)
                    columns = pd.Index(canonical_order, name="to_zone_id", dtype=str)

                    # Get AM time period data
                    am_idx = list(ds.coords["time_period"].values).index("AM")
                    travel_time_data = ds["SOV_TIME"].isel(time_period=am_idx).values

                    travel_time_mins = np.array(travel_time_data)
                    out = (
                        pd.DataFrame(travel_time_mins, index=index, columns=columns)
                        .stack()
                        .rename("SOV_AM_IVT_mins")
                    )
                    return out.to_frame()
            else:
                raise NotImplementedError(
                    "Invalid skim format {0}".format(skims_fname.split(".")[-1])
                )
        elif skim_format == "polaris":
            raise NotImplementedError("DEMOS requires skims in omx format, not polaris")

    except KeyError:
        raise KeyError("Couldn't find input skims named {0}".format(skims_fname))

    raise NotImplementedError("Invalid skims format {0}".format(skim_format))


class UrbansimPreprocessor(GenericPreprocessor):
    """
    UrbanSim-specific preprocessor that consolidates all preprocessing steps for the UrbanSim land use model.
    This includes copying input files to a mutable location and preparing any additional
    data needed for the UrbanSim run.
    """

    @staticmethod
    def declared_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this preprocessor expects without disk checks.
        """
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
        return {
            "usim_source_data_dir": settings.urbansim.local_data_input_folder,
            "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir(),
            "usim_datastore_h5": os.path.join(
                workspace.get_usim_mutable_data_dir(), usim_input_fname
            ),
        }

    @staticmethod
    def runtime_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare runtime expected inputs, including filesystem presence checks.
        """
        project_root = find_project_root(start_path=os.path.dirname(__file__))
        if not project_root:
            project_root = os.path.realpath(os.getcwd())
        source_dir = (
            settings.urbansim.local_data_input_folder
            if os.path.isabs(settings.urbansim.local_data_input_folder)
            else os.path.join(project_root, settings.urbansim.local_data_input_folder)
        )
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "usim_source_data_dir": source_dir,
            "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir(),
            "usim_datastore_h5": (
                usim_input_path if os.path.exists(usim_input_path) else None
            ),
        }

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        return UrbansimPreprocessor.runtime_expected_inputs(settings, state, workspace)

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: "Workspace"
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this preprocessor produces.

        Notes
        -----
        Output keys
            - ``usim_mutable_data_dir``: UrbanSim mutable data directory populated
              with inputs for the runner.
            - ``usim_datastore_h5``: Base-year UrbanSim datastore staged into the
              mutable directory.
        Related docs
            - See `pilates/urbansim/inputs.py` for the corresponding input
              descriptions used by UrbanSim and downstream models.
        """
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_input_fname = settings.urbansim.input_file_template.format(
            region_id=region_id
        )
        usim_input_path = os.path.join(
            workspace.get_usim_mutable_data_dir(), usim_input_fname
        )
        return {
            "usim_mutable_data_dir": workspace.get_usim_mutable_data_dir(),
            "usim_datastore_h5": usim_input_path,
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_data = ["usim_data_reference"]

    def copy_data_to_mutable_location(
        self,
        settings: PilatesConfig,
        output_dir,
        workspace: Optional["Workspace"] = None,
    ) -> Tuple[RecordStore, RecordStore]:
        """
        Copy UrbanSim input files from production to mutable location.

        Returns:
            Tuple[RecordStore, RecordStore]: (inputs, outputs) as RecordStores
        """
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]

        # Get the input file templates
        year_template = settings.urbansim.input_file_template_year
        if year_template:
            year_specific_model_data_fname = year_template.format(
                region_id=region_id, start_year=settings.run.start_year
            )
        else:
            year_specific_model_data_fname = ""

        base_template = settings.urbansim.input_file_template
        if base_template:
            model_data_fname = base_template.format(region_id=region_id)
        else:
            model_data_fname = ""
        data_dir = str(_urbansim_source_data_dir(settings))

        # Validate we have a filename
        if not model_data_fname:
            raise ValueError(
                "UrbanSim input file template is not configured. "
                "Please set 'urbansim.input_file_template' or 'usim_formattable_input_file_name' in settings."
            )

        if os.path.exists(os.path.join(data_dir, year_specific_model_data_fname)) and (
            settings.urbansim.input_file_template_year is not None
        ):
            src = os.path.join(data_dir, year_specific_model_data_fname)
        else:
            src = os.path.join(data_dir, model_data_fname)
        dest = os.path.join(output_dir, model_data_fname)

        logger.info(
            f"[UrbansimPreprocessor] Copying input urbansim data from {src} to {dest}"
        )
        if os.path.exists(src):
            shutil.copyfile(src, dest)
        else:
            # Create an empty HDF5 file if the source does not exist
            with pd.HDFStore(dest, "w"):
                pass
            logger.warning(
                f"[UrbansimPreprocessor] Source UrbanSim HDF5 file not found at {src}. Created empty HDF5 at {dest}."
            )
        inputs = [
            FileRecord(
                file_path=src,
                description="Reference urbanSim model data",
                short_name="usim_data_reference",
            )
        ]
        outputs = [
            FileRecord(
                file_path=dest,
                description="UrbanSim model data",
                short_name="usim_datastore_h5",
            )
        ]

        beam_inputs_root = str(_beam_input_root(settings))
        skims_src = os.path.abspath(
            os.path.join(
                beam_inputs_root, settings.run.region, settings.shared.skims.fname
            )
        )
        skims_target = os.path.join(output_dir, "skims_mpo_{0}.omx".format(region_id))

        inputs.append(
            FileRecord(
                file_path=skims_src,
                short_name="omx_skims",
                description="Raw BEAM OD skims",
            )
        )
        shutil.copyfile(skims_src, skims_target)

        outputs.append(
            FileRecord(
                file_path=skims_target,
                short_name="omx_skims",
                description="Raw BEAM OD skims for USim",
            )
        )

        logger.info(
            f"[UrbansimPreprocessor] Copying input beam skims data from {skims_src} to {skims_target}"
        )

        other_data_fnames = {
            f"hsize_ct_{region_id}.csv": "hh_size",
            f"income_rates_{region_id}.csv": "income_rates",
            f"relmap_{region_id}.csv": "relmap",
            "schools_2010.csv": "schools",
            "blocks_school_districts_2010.csv": "school_districts",
        }
        for fname, short_name in other_data_fnames.items():
            src = os.path.join(data_dir, fname)
            dest = os.path.join(output_dir, fname)
            if os.path.exists(src):
                logger.info(
                    f"[UrbansimPreprocessor] Copying input urbansim file from {src} to {dest}"
                )
                shutil.copyfile(src, dest)
                inputs.append(
                    FileRecord(
                        file_path=src,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
                outputs.append(
                    FileRecord(
                        file_path=dest,
                        description=f"UrbanSim input file: {fname}",
                        short_name=short_name,
                    )
                )
        logger.info("[UrbansimPreprocessor] Finished copying UrbanSim input files.")
        return RecordStore(recordList=inputs), RecordStore(recordList=outputs)

    def preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
        final_skims_omx: Optional[Any] = None,
    ) -> UrbanSimPreprocessOutputs:
        """Prepare UrbanSim inputs and return typed outputs."""
        self.state.set_sub_stage_progress("preprocessor")
        return self._preprocess(
            workspace,
            previous_records,
            final_skims_omx=final_skims_omx,
        )

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
        final_skims_omx: Optional[Any] = None,
    ) -> UrbanSimPreprocessOutputs:
        """
        Preprocess UrbanSim data, including copying necessary skims.
        """
        logger.info("[UrbansimPreprocessor] Preprocessing for UrbanSim.")
        settings = self.state.full_settings

        # Ensure the mutable data directory exists, especially on restarts when Initialization is skipped.
        usim_mutable_data_dir = workspace.get_usim_mutable_data_dir()
        os.makedirs(usim_mutable_data_dir, exist_ok=True)
        _restore_missing_mutable_urbansim_supporting_inputs(
            settings,
            self.state,
            workspace,
        )

        updated_skims_path: Optional[Path] = None

        try:
            # Generate the block-to-zone mapping for UrbanSim
            logger.info("Generating block-to-zone mapping for UrbanSim.")
            from pilates.utils.zone_utils import get_block_to_zone_mapping

            mapping = get_block_to_zone_mapping(
                settings, self.state.start_year, workspace
            )

            # Save the mapping to a CSV file in the mutable data directory
            geoid_to_zone_fname = "geoid_to_zone.csv"
            geoid_to_zone_path = os.path.join(
                workspace.get_usim_mutable_data_dir(), geoid_to_zone_fname
            )
            (
                pd.DataFrame.from_dict(mapping, orient="index", columns=["zone_id"])
                .rename_axis("GEOID")
                .to_csv(geoid_to_zone_path)
            )

            # If not the first iteration, check if BEAM is enabled and copy updated skims
            if (
                self.state.current_year > settings.run.start_year
                or self.state.iteration > 0
            ):
                if settings.run.models.travel == "beam":
                    logger.info(
                        "Updating skims from BEAM mutable output for subsequent iteration."
                    )
                    if final_skims_omx is not None and Path(final_skims_omx).exists():
                        source_skims_path = str(Path(final_skims_omx))
                        logger.info(
                            "[UrbansimPreprocessor] Using explicit final_skims_omx artifact: %s",
                            source_skims_path,
                        )
                    else:
                        if final_skims_omx is not None:
                            logger.warning(
                                "[UrbansimPreprocessor] Explicit final_skims_omx artifact "
                                "did not resolve to an existing path: %s. Falling back to "
                                "legacy BEAM skims discovery.",
                                final_skims_omx,
                            )
                        is_restart_run = getattr(self.state, "is_restart_run", None)
                        if is_restart_run is None:
                            is_restart_run = bool(self.state.run_info_path)
                        if (
                            is_restart_run
                            and self.state.run_info_path
                            and os.path.exists(self.state.run_info_path)
                        ):
                            logger.info(
                                f"[UrbansimPreprocessor] Restarted run detected. Using previous run's output path from {self.state.run_info_path}"
                            )
                            previous_run_dir = os.path.dirname(self.state.run_info_path)
                            beam_mutable_data_dir = os.path.join(
                                previous_run_dir, "beam", "input"
                            )
                        else:
                            beam_mutable_data_dir = workspace.get_beam_mutable_data_dir()

                        source_skims_path = os.path.join(
                            beam_mutable_data_dir,
                            settings.run.region,
                            settings.shared.skims.fname,
                        )

                    region_id = settings.urbansim.region_mappings[
                        "region_to_region_id"
                    ][settings.run.region]
                    dest_skims_fname = f"skims_mpo_{region_id}.omx"
                    dest_skims_path = os.path.join(
                        workspace.get_usim_mutable_data_dir(), dest_skims_fname
                    )

                    if os.path.exists(source_skims_path):
                        logger.info(
                            f"Copying skims from {source_skims_path} to {dest_skims_path}"
                        )
                        shutil.copy(source_skims_path, dest_skims_path)
                        updated_skims_path = Path(dest_skims_path)
                    else:
                        logger.warning(
                            f"Skims file not found at source: {source_skims_path}"
                        )

        except Exception as e:
            logger.error(f"Error during UrbanSim preprocessing: {e}")
            raise

        prepared_inputs = _mutable_urbansim_input_paths(settings, workspace)
        if updated_skims_path is not None and updated_skims_path.exists():
            prepared_inputs["usim_skims_input_updated"] = updated_skims_path

        return UrbanSimPreprocessOutputs(
            usim_mutable_data_dir=Path(usim_mutable_data_dir),
            prepared_inputs=prepared_inputs,
        )
