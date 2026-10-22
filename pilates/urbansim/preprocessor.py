from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING, Dict, Any
import os
import shutil
import logging
import pandas as pd

from pilates.config import PilatesConfig
from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore, FileRecord
from pilates.runtime.archive_paths import archive_fallback_path
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs
from pilates.utils.beam import get_beam_omx_skims_name
from pilates.utils.path_utils import find_project_root
from pilates.utils.zone_utils import resolve_canonical_zone_source_path
from pilates.workflows.input_authority import requires_prior_beam_skim_handoff

if TYPE_CHECKING:
    from workflow_state import WorkflowState
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)

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


def selected_urbansim_static_input_sources(
    settings: PilatesConfig,
    workspace: "Workspace",
) -> Tuple[Tuple[str, Path], ...]:
    """Resolve immutable sources that contribute to an UrbanSim run's identity.

    The relative names describe the stable model-facing role of each source;
    absolute host paths remain values only. These files are identity inputs for
    Consist, not semantic coupler roles or outputs recovered from an older run.
    """

    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    source_dir = _urbansim_source_data_dir(settings)
    beam_inputs_root = _beam_input_root(settings)
    canonical_zone_source = Path(
        resolve_canonical_zone_source_path(settings, workspace)
    )

    selected_sources = (
        (
            f"skims_mpo_{region_id}.omx",
            beam_inputs_root / region / get_beam_omx_skims_name(settings),
        ),
        (f"hsize_ct_{region_id}.csv", source_dir / f"hsize_ct_{region_id}.csv"),
        (
            f"income_rates_{region_id}.csv",
            source_dir / f"income_rates_{region_id}.csv",
        ),
        (f"relmap_{region_id}.csv", source_dir / f"relmap_{region_id}.csv"),
        ("schools_2010.csv", source_dir / "schools_2010.csv"),
        (
            "blocks_school_districts_2010.csv",
            source_dir / "blocks_school_districts_2010.csv",
        ),
        (
            f"canonical_zones/{canonical_zone_source.name}",
            canonical_zone_source,
        ),
    )
    missing = [str(path) for _name, path in selected_sources if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Missing configured UrbanSim static input files: " + ", ".join(missing)
        )
    return selected_sources


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
            beam_inputs_root / region / get_beam_omx_skims_name(settings),
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

        archive_path = archive_fallback_path(
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
    mutable_data_dir: Path,
) -> Dict[str, Path]:
    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    mutable_dir = Path(mutable_data_dir)

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


def _stage_declared_urbansim_datastore(
    *,
    settings: PilatesConfig,
    mutable_data_dir: Path,
    usim_datastore_h5: Path,
) -> Path:
    """Stage the bound datastore at UrbanSim's model-visible input path.

    Native execution receives a Consist-managed snapshot.  UrbanSim itself
    still consumes the conventional mutable-data filename, so this is the one
    explicit handoff from the immutable binding into that model-visible path.
    It intentionally overwrites a stale workspace copy rather than treating it
    as an alternate source.
    """
    source = Path(usim_datastore_h5)
    if not source.is_file():
        raise RuntimeError(
            f"UrbanSim preprocess requires the bound datastore H5 to exist: {source}"
        )

    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    destination = mutable_data_dir / (
        settings.urbansim.input_file_template.format(region_id=region_id)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def stage_urbansim_run_workspace(
    *,
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: "Workspace",
    usim_datastore_h5: Path,
    final_skims_omx: Path | None,
    mutable_data_dir: Path | None = None,
) -> UrbanSimPreprocessOutputs:
    """Stage UrbanSim's mutable workspace immediately before its container run.

    The workflow supplies immutable, already-materialized artifacts. UrbanSim's
    container continues to read its conventional mutable-data filenames, so this
    helper is the sole handoff between those two representations.
    """
    logger.info("[UrbansimRunner] Staging mutable UrbanSim run workspace.")
    mutable_dir = (
        Path(mutable_data_dir)
        if mutable_data_dir is not None
        else Path(workspace.get_usim_mutable_data_dir())
    )
    mutable_dir.mkdir(parents=True, exist_ok=True)

    _stage_declared_urbansim_datastore(
        settings=settings,
        mutable_data_dir=mutable_dir,
        usim_datastore_h5=Path(usim_datastore_h5),
    )
    for relative_name, source_path in selected_urbansim_static_input_sources(
        settings,
        workspace,
    ):
        if relative_name.startswith("canonical_zones/"):
            continue
        destination = mutable_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    from pilates.utils.zone_utils import get_block_to_zone_mapping

    mapping = get_block_to_zone_mapping(settings, state.start_year, workspace)
    geoid_to_zone_path = mutable_dir / "geoid_to_zone.csv"
    (
        pd.DataFrame.from_dict(mapping, orient="index", columns=["zone_id"])
        .rename_axis("GEOID")
        .to_csv(geoid_to_zone_path)
    )

    updated_skims_path: Optional[Path] = None
    explicit_skims_path = (
        Path(final_skims_omx)
        if final_skims_omx is not None and Path(final_skims_omx).is_file()
        else None
    )
    requires_final_skims = requires_prior_beam_skim_handoff(
        settings=settings,
        state=state,
    )
    if explicit_skims_path is None and requires_final_skims:
        raise FileNotFoundError(
            "UrbanSim run requires the resolved final_skims_omx handoff "
            "after BEAM has run."
        )
    if final_skims_omx is not None and explicit_skims_path is None:
        logger.warning(
            "[UrbansimRunner] Explicit final_skims_omx artifact did not resolve "
            "to an existing path: %s. Keeping the configured base OMX.",
            final_skims_omx,
        )
    if explicit_skims_path is not None:
        logger.info(
            "[UrbansimRunner] Using explicit final_skims_omx artifact: %s",
            explicit_skims_path,
        )
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        destination = mutable_dir / f"skims_mpo_{region_id}.omx"
        logger.info("Copying skims from %s to %s", explicit_skims_path, destination)
        shutil.copy2(explicit_skims_path, destination)
        updated_skims_path = destination

    prepared_inputs = _mutable_urbansim_input_paths(settings, mutable_dir)
    if updated_skims_path is not None:
        prepared_inputs["usim_skims_input_updated"] = updated_skims_path
    return UrbanSimPreprocessOutputs(
        usim_mutable_data_dir=mutable_dir,
        prepared_inputs=prepared_inputs,
    )


def stage_synthetic_population(
    settings: PilatesConfig,
    output_dir: str,
) -> Tuple[RecordStore, RecordStore]:
    """
    Stage the base synthetic population HDF5 datastore into the mutable workspace.

    This helper copies ``custom_mpo_*.h5`` (and its ``.metadata.json`` sidecar if
    present) into ``<workspace>/urbansim/data/`` without invoking full UrbanSim
    land-use preprocessing or staging UrbanSim-only static/skim inputs.
    """
    urbansim_cfg = getattr(settings, "urbansim", None)
    if urbansim_cfg is None:
        return RecordStore(), RecordStore()

    region = settings.run.region
    region_id = urbansim_cfg.region_mappings["region_to_region_id"][region]

    year_template = urbansim_cfg.input_file_template_year
    if year_template:
        year_specific_model_data_fname = year_template.format(
            region_id=region_id, start_year=settings.run.start_year
        )
    else:
        year_specific_model_data_fname = ""

    base_template = urbansim_cfg.input_file_template
    if base_template:
        model_data_fname = base_template.format(region_id=region_id)
    else:
        model_data_fname = ""
    data_dir = str(_urbansim_source_data_dir(settings))

    if not model_data_fname:
        raise ValueError(
            "UrbanSim input file template is not configured. "
            "Please set 'urbansim.input_file_template' or 'usim_formattable_input_file_name' in settings."
        )

    if (
        year_specific_model_data_fname
        and os.path.exists(os.path.join(data_dir, year_specific_model_data_fname))
        and urbansim_cfg.input_file_template_year is not None
    ):
        src = os.path.join(data_dir, year_specific_model_data_fname)
    else:
        src = os.path.join(data_dir, model_data_fname)
    dest = os.path.join(output_dir, model_data_fname)

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Synthetic population HDF5 file not found at configured path: {src}"
        )

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Copying synthetic population HDF5 data from {src} to {dest}")
    shutil.copy2(src, dest)

    sidecar_src = f"{src}.metadata.json"
    sidecar_dest = f"{dest}.metadata.json"
    if os.path.exists(sidecar_src):
        shutil.copy2(sidecar_src, sidecar_dest)

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
    return RecordStore(recordList=inputs), RecordStore(recordList=outputs)


class UrbansimPreprocessor(GenericPreprocessor):
    """
    UrbanSim-specific preprocessor that consolidates all preprocessing steps for the UrbanSim land use model.
    This includes copying input files to a mutable location and preparing any additional
    data needed for the UrbanSim run.
    """

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
        del workspace
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        data_dir = str(_urbansim_source_data_dir(settings))

        in_store, out_store = stage_synthetic_population(settings, output_dir)
        inputs = list(in_store.all_records())
        outputs = list(out_store.all_records())

        beam_inputs_root = str(_beam_input_root(settings))
        skims_src = os.path.abspath(
            os.path.join(
                beam_inputs_root,
                settings.run.region,
                get_beam_omx_skims_name(settings),
            )
        )

        skims_target = os.path.join(output_dir, "skims_mpo_{0}.omx".format(region_id))
        if os.path.islink(skims_target) and not os.path.exists(skims_target):
            try:
                os.unlink(skims_target)
            except OSError:
                pass

        if os.path.exists(skims_src):
            inputs.append(
                FileRecord(
                    file_path=skims_src,
                    short_name="omx_skims",
                    description="Raw BEAM OD skims",
                )
            )
            shutil.copy2(skims_src, skims_target)

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
        else:
            raise FileNotFoundError(
                f"[UrbansimPreprocessor] UrbanSim skims file not found at {skims_src}"
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
                shutil.copy2(src, dest)
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
        usim_datastore_h5: Optional[Path] = None,
        final_skims_omx: Optional[Any] = None,
        allow_workspace_skim_fallback: bool = True,
    ) -> UrbanSimPreprocessOutputs:
        """Prepare UrbanSim inputs and return typed outputs."""
        if usim_datastore_h5 is None:
            raise TypeError(
                "UrbansimPreprocessor.preprocess requires usim_datastore_h5"
            )
        self.state.set_sub_stage_progress("preprocessor")
        return self._preprocess(
            workspace,
            previous_records,
            usim_datastore_h5=usim_datastore_h5,
            final_skims_omx=final_skims_omx,
            allow_workspace_skim_fallback=allow_workspace_skim_fallback,
        )

    def _preprocess(
        self,
        workspace: "Workspace",
        previous_records: Optional[RecordStore] = None,
        usim_datastore_h5: Optional[Path] = None,
        final_skims_omx: Optional[Any] = None,
        allow_workspace_skim_fallback: bool = True,
    ) -> UrbanSimPreprocessOutputs:
        """Retain the legacy preprocessor surface without staging runner inputs."""
        del (
            previous_records,
            usim_datastore_h5,
            final_skims_omx,
            allow_workspace_skim_fallback,
        )
        usim_mutable_data_dir = Path(workspace.get_usim_mutable_data_dir())
        usim_mutable_data_dir.mkdir(parents=True, exist_ok=True)
        return UrbanSimPreprocessOutputs(
            usim_mutable_data_dir=usim_mutable_data_dir,
        )
