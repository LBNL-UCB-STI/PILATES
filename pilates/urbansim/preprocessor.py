from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING, Dict
import os
import shutil
import logging
import pandas as pd

from pilates.config import PilatesConfig
from pilates.urbansim.outputs import UrbanSimPreprocessOutputs
from pilates.utils.admission_helpers import resolve_local_source_uri
from pilates.utils.beam import get_beam_omx_skims_name
from pilates.utils.path_utils import find_project_root
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.zone_utils import resolve_canonical_zone_source_path
from pilates.workflows.input_authority import requires_prior_beam_skim_handoff

if TYPE_CHECKING:
    from pilates.workflows.state import WorkflowState
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


def resolve_urbansim_initial_datastore_source_path(
    settings: PilatesConfig,
) -> Optional[Path]:
    """Resolve the configured static UrbanSim initial HDF5 datastore path."""
    raw_source_uri = get_setting(
        settings,
        "urbansim.admission.initial_datastore.expectation.source_uri",
    )
    source_uri_path = resolve_local_source_uri(
        str(raw_source_uri) if raw_source_uri is not None else None
    )
    if source_uri_path is not None and source_uri_path.is_file():
        return source_uri_path

    urbansim_cfg = get_setting(settings, "urbansim")
    if urbansim_cfg is None:
        return source_uri_path

    region = get_setting(settings, "run.region")
    if not region:
        return source_uri_path

    region_map = (
        get_setting(urbansim_cfg, "region_mappings.region_to_region_id", {}) or {}
    )
    region_id = region_map.get(region) if isinstance(region_map, dict) else None
    if region_id is None:
        return source_uri_path

    if not get_setting(urbansim_cfg, "local_data_input_folder"):
        return source_uri_path

    data_dir = _urbansim_source_data_dir(settings)
    start_year = get_setting(settings, "run.start_year")
    year_template = get_setting(urbansim_cfg, "input_file_template_year")
    if year_template and start_year is not None:
        year_specific_fname = year_template.format(
            region_id=region_id,
            year=start_year,
            start_year=start_year,
        )
        year_candidate = data_dir / year_specific_fname
        if year_candidate.is_file():
            return year_candidate

    base_template = get_setting(urbansim_cfg, "input_file_template")
    if not base_template:
        return source_uri_path
    model_data_fname = base_template.format(region_id=region_id)
    base_candidate = data_dir / model_data_fname
    if base_candidate.is_file() or source_uri_path is None:
        return base_candidate
    return source_uri_path
