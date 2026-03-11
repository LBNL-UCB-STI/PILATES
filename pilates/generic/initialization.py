from typing import Optional, Iterator, Tuple, Dict, Any, TYPE_CHECKING

from pilates.config import PilatesConfig
from pilates.generic.model import Model
from pilates.generic.records import FileRecord, RecordStore, sanitize_artifact_key
from pilates.utils.path_utils import find_project_root
from pilates.workspace import Workspace
from pilates.generic.model_factory import ModelFactory
from pilates.utils import consist_runtime as cr

import os
import logging
import shutil

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from workflow_state import WorkflowState


_BOOTSTRAP_DIRECTION_KEY = "bootstrap_direction"


def _counts_by_model_from_records(
    copied_records: Optional[RecordStore],
    *,
    direction: str,
) -> Dict[str, int]:
    if not isinstance(copied_records, RecordStore):
        return {}
    counts: Dict[str, int] = {}
    for record in copied_records.all_records():
        metadata = getattr(record, "metadata", None) or {}
        if metadata.get(_BOOTSTRAP_DIRECTION_KEY) != direction:
            continue
        model_name = metadata.get("model")
        if not model_name:
            continue
        counts[model_name] = counts.get(model_name, 0) + 1
    return counts


def build_bootstrap_artifact_summary(
    workspace: Workspace, copied_records: Optional[RecordStore] = None
) -> Dict[str, Any]:
    """
    Build a compact summary of initialization artifacts staged into workspace.

    This summary is intentionally lightweight for Phase 1 bootstrap reporting.
    """
    input_counts = _counts_by_model_from_records(
        copied_records,
        direction="input",
    )
    output_counts = _counts_by_model_from_records(
        copied_records,
        direction="output",
    )

    models = sorted(set(input_counts.keys()) | set(output_counts.keys()))
    input_total = sum(input_counts.values())
    output_total = sum(output_counts.values())
    copied_total = len(copied_records.all_records()) if isinstance(copied_records, RecordStore) else 0

    return {
        "models": models,
        "input_records_by_model": input_counts,
        "output_records_by_model": output_counts,
        "input_records_total": input_total,
        "output_records_total": output_total,
        "copied_records_total": copied_total,
    }


def _tag_record_store(
    record_store: RecordStore,
    model_name: Optional[str],
    *,
    direction: Optional[str] = None,
) -> None:
    if not model_name:
        return
    for record in record_store.all_records():
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("model", model_name)
            if direction is not None:
                metadata.setdefault(_BOOTSTRAP_DIRECTION_KEY, direction)


def _iter_unique_records(record_store: RecordStore) -> Iterator[Tuple[str, object]]:
    used_keys = set()
    for record in record_store.all_records():
        key = getattr(record, "short_name", None) or getattr(
            record, "unique_id", None
        )
        if not key:
            logger.warning(
                "Initialization record missing short_name/unique_id; skipping."
            )
            continue
        sanitized = sanitize_artifact_key(key)
        if sanitized is None:
            fallback = getattr(record, "unique_id", None)
            if not fallback:
                logger.warning(
                    "Initialization record key '%s' could not be sanitized; skipping.",
                    key,
                )
                continue
            logger.warning(
                "Initialization record key '%s' invalid; using unique_id '%s'.",
                key,
                fallback,
            )
            key = fallback
        else:
            if sanitized != key:
                logger.warning(
                    "Initialization record key '%s' sanitized to '%s'.",
                    key,
                    sanitized,
                )
            key = sanitized

        if key in used_keys:
            metadata = getattr(record, "metadata", {}) or {}
            model_name = metadata.get("model")
            if model_name:
                namespaced = f"{model_name}/{key}"
                namespaced = sanitize_artifact_key(namespaced) or namespaced
                if namespaced not in used_keys:
                    logger.warning(
                        "Duplicate initialization key '%s' detected; using namespaced key '%s'.",
                        key,
                        namespaced,
                    )
                    key = namespaced
                else:
                    fallback = getattr(record, "unique_id", None)
                    if not fallback or fallback in used_keys:
                        logger.warning(
                            "Duplicate initialization key '%s' with no safe fallback; skipping.",
                            key,
                        )
                        continue
                    logger.warning(
                        "Duplicate initialization key '%s' detected; using unique_id '%s'.",
                        key,
                        fallback,
                    )
                    key = fallback
            else:
                fallback = getattr(record, "unique_id", None)
                if not fallback or fallback in used_keys:
                    logger.warning(
                        "Duplicate initialization key '%s' with no safe fallback; skipping.",
                        key,
                    )
                    continue
                logger.warning(
                    "Duplicate initialization key '%s' detected; using unique_id '%s'.",
                    key,
                    fallback,
                )
                key = fallback

        used_keys.add(key)
        yield key, record


def _log_record_store(
    record_store: RecordStore,
    *,
    log_fn,
    workspace: Workspace,
    direction: str,
) -> None:
    def _h5_table_filter_from_list(tables_used):
        normalized = {name if name.startswith("/") else f"/{name}" for name in tables_used if name}

        def _filter(table_name: str) -> bool:
            if any(tok in table_name for tok in ("_axis", "_block", "_level", "_label")):
                return False
            return table_name in normalized

        return _filter

    schema_keys = {"canonical_zones_source", "omx_skims"}
    for key, record in _iter_unique_records(record_store):
        path = record.get_absolute_path(base_path=workspace.full_path)
        if not path:
            logger.warning(
                "Initialization %s record '%s' missing path; skipping.",
                direction,
                key,
            )
            continue
        if not os.path.exists(path):
            logger.warning(
                "Initialization %s path does not exist for key '%s': %s",
                direction,
                key,
                path,
            )
            continue
        meta = {}
        if key in schema_keys:
            meta["profile_file_schema"] = "if_changed"
        tables_used = getattr(record, "h5_tables_used", None)
        if tables_used:
            normalized_tables = sorted(
                {
                    name if name.startswith("/") else f"/{name}"
                    for name in tables_used
                    if name
                }
            )
            if normalized_tables:
                meta["h5_table_paths"] = normalized_tables
                meta["h5_table_count"] = len(normalized_tables)
            artifact = cr.log_h5_container(
                path,
                key=key,
                direction=direction,
                table_filter=_h5_table_filter_from_list(tables_used),
                description=getattr(record, "description", None),
                **meta,
            )
        else:
            artifact = log_fn(
                path,
                key=key,
                description=getattr(record, "description", None),
                **meta,
            )
        if artifact is not None and hasattr(record, "content_hash"):
            record_hash = getattr(artifact, "hash", None)
            if record_hash:
                record.content_hash = record_hash

def _accumulate_copy_result(
    *,
    result: Optional[Tuple[RecordStore, RecordStore]],
    model_name: str,
    initialization_records_in: RecordStore,
    initialization_records_out: RecordStore,
) -> bool:
    """
    Tag and append copy outputs, optionally storing them on workspace caches.

    Returns
    -------
    bool
        True when records were added; False when `result` was empty.
    """
    if not result:
        return False

    rec_in, rec_out = result
    _tag_record_store(rec_in, model_name, direction="input")
    _tag_record_store(rec_out, model_name, direction="output")
    initialization_records_in += rec_in
    initialization_records_out += rec_out
    return True


class Initialization(Model):
    """
    A dedicated class to handle the initialization of mutable data and
    record creation. This class consolidates input data copying for
    all models into one single initialization job. It does not conform to the
    GenericPreprocessor interface.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)

    def run(self, settings: PilatesConfig, workspace: Workspace) -> RecordStore:
        """
        Execute the initialization process:
          - Copy all necessary input data from production directories to mutable locations.
        """
        initialization_records_in = RecordStore()
        initialization_records_out = RecordStore()
        have_not_copied_usim_data = True
        urbansim_enabled = settings.run.models.land_use == "urbansim"
        model_factory = ModelFactory()

        try:
            # BEAM model initialization
            if settings.run.models.travel == "beam":
                beam_preprocessor = model_factory.get_preprocessor("beam", self.state)

                beam_input_dir = workspace.get_beam_mutable_data_dir()
                result = beam_preprocessor.copy_data_to_mutable_location(
                    settings, beam_input_dir
                )
                _accumulate_copy_result(
                    result=result,
                    model_name="beam",
                    initialization_records_in=initialization_records_in,
                    initialization_records_out=initialization_records_out,
                )

            if settings.run.models.travel == "beam":
                zones_config = settings.shared.geography.zones
                if (
                    zones_config is not None
                    and settings.run.models.activity_demand is not None
                ):
                    project_root = find_project_root(start_path=os.path.dirname(__file__))
                    if not project_root:
                        project_root = os.path.realpath(os.getcwd())
                        logger.warning(
                            "[NOT IDEAL] Could not locate PILATES project root via markers; "
                            "falling back to cwd='%s'.",
                            project_root,
                        )
                    zone_source_path = zones_config.source_file
                    if not os.path.isabs(zone_source_path):
                        zone_source_path = os.path.join(project_root, zone_source_path)

                    asim_input_dir = workspace.get_asim_mutable_data_dir()
                    os.makedirs(asim_input_dir, exist_ok=True)
                    if os.path.exists(zone_source_path):
                        zone_fname = os.path.basename(zone_source_path)
                        dest_path = os.path.join(asim_input_dir, zone_fname)
                        if os.path.abspath(zone_source_path) != os.path.abspath(
                            dest_path
                        ):
                            logger.info(
                                "Copying canonical zones from %s to %s",
                                zone_source_path,
                                dest_path,
                            )
                            shutil.copy(zone_source_path, dest_path)
                        else:
                            logger.info(
                                "Canonical zones already at destination: %s",
                                dest_path,
                            )
                        rec_in = RecordStore(
                            recordList=[
                                FileRecord(
                                    file_path=zone_source_path,
                                    short_name="canonical_zones_source",
                                )
                            ]
                        )
                        rec_out = RecordStore(
                            recordList=[
                                FileRecord(
                                    file_path=dest_path,
                                    short_name="canonical_zones",
                                )
                            ]
                        )
                        _accumulate_copy_result(
                            result=(rec_in, rec_out),
                            model_name="activitysim",
                            initialization_records_in=initialization_records_in,
                            initialization_records_out=initialization_records_out,
                        )
                    else:
                        logger.warning(
                            "Canonical zone source file not found at %s, skipping copy.",
                            zone_source_path,
                        )
                elif zones_config is None:
                    logger.info(
                        "No zones configured in shared.geography.zones; skipping zone setup."
                    )
                else:
                    logger.info(
                        "ActivitySim not enabled; skipping activitysim directory setup."
                    )

            # Other models
            model_map = {
                "activity_demand": settings.run.models.activity_demand,
                "vehicle_ownership": settings.run.models.vehicle_ownership,
                "land_use": settings.run.models.land_use,
            }

            for model_key, model_name in model_map.items():
                if not model_name:
                    continue

                # UrbanSim data copy (once)
                if model_name == "urbansim" or (
                    model_name == "activitysim"
                    and have_not_copied_usim_data
                    and not urbansim_enabled
                ):

                    output_dir = workspace.get_usim_mutable_data_dir()
                    os.makedirs(output_dir, exist_ok=True)
                    usim_preprocessor = model_factory.get_preprocessor(
                        "urbansim", self.state
                    )
                    result = usim_preprocessor.copy_data_to_mutable_location(
                        settings, output_dir
                    )
                    _accumulate_copy_result(
                        result=result,
                        model_name=model_name,
                        initialization_records_in=initialization_records_in,
                        initialization_records_out=initialization_records_out,
                    )
                    have_not_copied_usim_data = False

                # Atlas data copy
                if model_name == "atlas":
                    input_dir = workspace.get_atlas_mutable_input_dir()
                    os.makedirs(input_dir, exist_ok=True)
                    atlas_preprocessor = model_factory.get_preprocessor(
                        "atlas", self.state
                    )
                    result = atlas_preprocessor.copy_data_to_mutable_location(
                        settings, input_dir
                    )
                    _accumulate_copy_result(
                        result=result,
                        model_name=model_name,
                        initialization_records_in=initialization_records_in,
                        initialization_records_out=initialization_records_out,
                    )
                    os.makedirs(workspace.get_atlas_output_dir(), exist_ok=True)

                # ActivitySim config copy
                if model_name == "activitysim":

                    activitysim_preprocessor = model_factory.get_preprocessor(
                        model_name, self.state
                    )
                    asim_input_dir = workspace.get_asim_mutable_data_dir()
                    rec_in, rec_out = (
                        activitysim_preprocessor.copy_data_to_mutable_location(
                            settings, asim_input_dir
                        )
                    )
                    _accumulate_copy_result(
                        result=(rec_in, rec_out),
                        model_name=model_name,
                        initialization_records_in=initialization_records_in,
                        initialization_records_out=initialization_records_out,
                    )

            # You can add further model-specific blocks (e.g., for urbansim, atlas) as needed

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise e

        _log_record_store(
            initialization_records_in,
            log_fn=cr.log_input,
            workspace=workspace,
            direction="input",
        )
        _log_record_store(
            initialization_records_out,
            log_fn=cr.log_output,
            workspace=workspace,
            direction="output",
        )

        combined_records = RecordStore()
        combined_records += initialization_records_in
        combined_records += initialization_records_out
        return combined_records
