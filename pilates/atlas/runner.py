from typing import Tuple
import logging
import os

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker
from pilates.atlas.preprocessor import get_atlas_docker_vols, get_atlas_cmd

logger = logging.getLogger(__name__)


class AtlasRunner(GenericRunner):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def run(
        self,
        store: RecordStore,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        settings = state.full_settings
        freq = settings.get("vehicle_ownership_freq", False)
        npe = settings.get("atlas_num_processes", False)
        nsample = settings.get("atlas_sample_size", False)
        beamac = settings.get("atlas_beamac", 0)
        mod = settings.get("atlas_mod", 1)
        adscen = settings.get("atlas_adscen", False)
        rebfactor = settings.get("atlas_rebfactor", 0)
        taxfactor = settings.get("atlas_taxfactor", 0)
        discIncent = settings.get("atlas_discIncent", 0)

        client = None
        if settings.get("container_manager") == "docker":
            try:
                client = self.initialize_docker_client(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")

        model_name = self.model_name
        atlas_image = settings[f"{settings['container_manager']}_images"][model_name]
        atlas_docker_vols = get_atlas_docker_vols(settings, workspace)
        atlas_cmd = get_atlas_cmd(
            settings,
            freq,
            state.forecast_year,
            npe,
            nsample,
            beamac,
            mod,
            adscen,
            rebfactor,
            taxfactor,
            discIncent,
        )

        model_run_hash = provenance_tracker.start_model_run(
            model_name,
            state.current_year,
            state.current_inner_iter,
            description="ATLAS run",
            inputs=store,
        )

        success = self.run_container(
            client=client,
            settings=settings,
            image=atlas_image,
            volumes=atlas_docker_vols,
            command=atlas_cmd,
            model_name=model_name,
            working_dir="/",
        )

        provenance_tracker.complete_model_run(
            run_hash=model_run_hash, status="completed" if success else "failed"
        )

        output_store = RecordStore()

        return output_store, provenance_tracker.run_info.model_runs.get(model_run_hash)
