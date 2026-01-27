from typing import Optional, Iterator, Tuple

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
        model_factory = ModelFactory()

        try:
            # BEAM model initialization
            if settings.run.models.travel == "beam":
                beam_preprocessor = model_factory.get_preprocessor("beam", self.state)

                beam_input_dir = workspace.get_beam_mutable_data_dir()
                result = beam_preprocessor.copy_data_to_mutable_location(
                    settings, beam_input_dir
                )
                if result:
                    rec_in, rec_out = result
                    initialization_records_in += rec_in
                    initialization_records_out += rec_out
                    # Make available to later preprocess()
                    workspace.input_data["beam"] = rec_in
                    workspace.output_data["beam"] = rec_out

            if (
                settings.run.models.travel == "beam"
                and settings.run.models.activity_demand is None
            ):
                asim_input_dir = workspace.get_asim_mutable_data_dir()
                os.makedirs(asim_input_dir, exist_ok=True)
                project_root = find_project_root(start_path=os.path.dirname(__file__))
                if not project_root:
                    project_root = os.path.realpath(os.getcwd())
                    logger.warning(
                        "[NOT IDEAL] Could not locate PILATES project root via markers; "
                        "falling back to cwd='%s'.",
                        project_root,
                    )
                zone_source_path = settings.shared.geography.zones.source_file
                if not os.path.isabs(zone_source_path):
                    zone_source_path = os.path.join(project_root, zone_source_path)
                if os.path.exists(zone_source_path):
                    zone_fname = os.path.basename(zone_source_path)
                    asim_zones_path = os.path.join(asim_input_dir, zone_fname)
                    logger.info(
                        "Copying canonical zones from %s to %s",
                        zone_source_path,
                        asim_zones_path,
                    )
                    shutil.copy(zone_source_path, asim_zones_path)
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
                                file_path=asim_zones_path,
                                short_name="canonical_zones",
                            )
                        ]
                    )
                    initialization_records_in += rec_in
                    initialization_records_out += rec_out
                else:
                    logger.warning(
                        "Canonical zone source file not found at %s, skipping copy.",
                        zone_source_path,
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
                    model_name == "activitysim" and have_not_copied_usim_data
                ):

                    output_dir = workspace.get_usim_mutable_data_dir()
                    os.makedirs(output_dir, exist_ok=True)
                    usim_preprocessor = model_factory.get_preprocessor(
                        "urbansim", self.state
                    )
                    result = usim_preprocessor.copy_data_to_mutable_location(
                        settings, output_dir
                    )
                    if result:
                        rec_in, rec_out = result
                        if model_name in workspace.input_data:
                            workspace.input_data[model_name] += rec_in
                        else:
                            workspace.input_data[model_name] = rec_in
                        if model_name in workspace.output_data:
                            workspace.output_data[model_name] += rec_out
                        else:
                            workspace.output_data[model_name] = rec_out
                    have_not_copied_usim_data = False
                    if result:
                        rec_in, rec_out = result
                        initialization_records_in += rec_in
                        initialization_records_out += rec_out

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
                    if result:
                        rec_in, rec_out = result
                        if model_name in workspace.input_data:
                            workspace.input_data[model_name] += rec_in
                        else:
                            workspace.input_data[model_name] = rec_in
                        if model_name in workspace.output_data:
                            workspace.output_data[model_name] += rec_out
                        else:
                            workspace.output_data[model_name] = rec_out
                        initialization_records_in += rec_in
                        initialization_records_out += rec_out
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
                    initialization_records_in += rec_in
                    initialization_records_out += rec_out
                    if model_name in workspace.input_data:
                        workspace.input_data[model_name] += rec_in
                    else:
                        workspace.input_data[model_name] = rec_in
                    if model_name in workspace.output_data:
                        workspace.output_data[model_name] += rec_out
                    else:
                        workspace.output_data[model_name] = rec_out

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
