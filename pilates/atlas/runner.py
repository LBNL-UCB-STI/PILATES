import sys
from typing import Tuple
import logging
import os

from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker

# Fix missing imports
from pilates.urbansim import postprocessor as usim_post
from pilates.atlas import preprocessor as atlas_pre
from pilates.atlas import postprocessor as atlas_post

logger = logging.getLogger(__name__)


## Atlas: evolve household vehicle ownership
# def run_atlas(
#     settings,
#     state: WorkflowState,
#     client,
#     workspace: Workspace,
#     provenance_tracker: FileProvenanceTracker,
#     warm_start_atlas,
#     forecast=False,
#     atlas_run_count=1,
# ):
#     # warm_start: warm_start_atlas = True, output_year = year = start_year
#     # asim_no_usim: warm_start_atlas = True, output_year = year (should  = start_year)
#     # normal: warm_start_atlas = False, output_year = forecast_year
#
#     if forecast:
#         yr = state.forecast_year
#     else:
#         yr = state.start_year
#
#     # 1. PARSE SETTINGS
#     vehicle_ownership_model, atlas_image = GenericRunner.get_model_and_image(
#         settings, "vehicle_ownership_model"
#     )
#     freq = settings.get("vehicle_ownership_freq", False)
#     npe = settings.get("atlas_num_processes", False)
#     nsample = settings.get("atlas_sample_size", False)
#     beamac = settings.get("atlas_beamac", 0)
#     mod = settings.get("atlas_mod", 1)
#     adscen = settings.get("atlas_adscen", False)
#     rebfactor = settings.get("atlas_rebfactor", 0)
#     taxfactor = settings.get("atlas_taxfactor", 0)
#     discIncent = settings.get("atlas_discIncent", 0)
#     atlas_docker_vols = get_atlas_docker_vols(settings, workspace)
#     atlas_cmd = get_atlas_cmd(
#         settings,
#         freq,
#         yr,
#         npe,
#         nsample,
#         beamac,
#         mod,
#         adscen,
#         rebfactor,
#         taxfactor,
#         discIncent,
#     )
#     docker_stdout = settings.get("docker_stdout", False)
#
#     # 2. PREPARE ATLAS DATA
#     if warm_start_atlas:
#         print_str = "Preparing input data for warm start vehicle ownership simulation for {0}.".format(
#             yr
#         )
#     else:
#         print_str = (
#             "Preparing input data for vehicle ownership simulation for {0}.".format(yr)
#         )
#     logger.info(print_str)
#
#     # prepare atlas inputs from urbansim h5 output
#     # preprocessed csv input files saved in "atlas/atlas_inputs/year{}/"
#     if forecast:
#         yrs = [y + 2 for y in range(state.year, yr, 2)]
#     else:
#         yrs = [yr]
#
#     # Record inputs to Atlas preprocessing
#     usim_output_store_name = usim_post.get_usim_datastore_fname(
#         settings, io="output", year=state.forecast_year
#     )
#     usim_output_store_path = os.path.join(
#         workspace.get_usim_mutable_data_dir(),
#         usim_output_store_name,
#     )
#
#     atlas_pre_hash = provenance_tracker.start_model_run(
#         f"{vehicle_ownership_model}_preprocessor", yr, description="Atlas preprocessing"
#     )
#
#     provenance_tracker.record_input_file(
#         f"{vehicle_ownership_model}_preprocessor",
#         usim_output_store_path,
#         description="UrbanSim output for Atlas input preparation",
#         model_run_id=atlas_pre_hash,
#     )
#
#     for yr_it in yrs:
#         atlas_pre.preprocess(
#             state,
#             workspace,
#             provenance_tracker,
#         )
#
#     # calculate accessibility if beamac != 0
#     if beamac > 0:
#         # Record inputs to accessibility calculation (BEAM skims)
#         beam_output_dir = workspace.get_beam_output_dir()
#         expected_beam_skims_path = os.path.join(
#             beam_output_dir, settings["skims_fname"]
#         )
#         provenance_tracker.record_input_file(
#             f"{vehicle_ownership_model}_preprocessor",
#             expected_beam_skims_path,
#             description="BEAM skims for Atlas accessibility calculation",
#             model_run_id=atlas_pre_hash,
#         )
#
#         path_list = [
#             "WLK_COM_WLK",
#             "WLK_EXP_WLK",
#             "WLK_HVY_WLK",
#             "WLK_LOC_WLK",
#             "WLK_LRF_WLK",
#         ]
#         measure_list = ["WACC", "IWAIT", "XWAIT", "TOTIVT", "WEGR"]
#         # compute_accessibility expects (path_list, measure_list, settings, year)
#         atlas_pre.compute_accessibility(
#             path_list,
#             measure_list,
#             settings,
#             state.forecast_year,
#         )
#
#     provenance_tracker.complete_model_run(atlas_pre_hash)
#
#     # 3. RUN ATLAS via docker container client
#     print_str = (
#         "Simulating vehicle ownership for {0} "
#         "with frequency {1}, npe {2} nsample {3} beamac {4}".format(
#             yr, freq, npe, nsample, beamac
#         )
#     )
#     logger.info(print_str)
#
#     # Start Atlas run and get model_run_hash
#     atlas_run_hash = provenance_tracker.start_model_run(
#         vehicle_ownership_model, yr, description="Atlas run", iteration=atlas_run_count
#     )
#
#     success = GenericRunner.run_container(
#         client=client,
#         settings=settings,
#         image=atlas_image,
#         volumes=atlas_docker_vols,
#         command=atlas_cmd,
#         model_name=vehicle_ownership_model,
#         working_dir="/",
#     )
#
#     # Record Atlas run completion
#     provenance_tracker.complete_model_run(
#         atlas_run_hash, status="completed" if success else "failed"
#     )
#     if not success:
#         logger.error("Atlas run failed.")
#         # Don't exit immediately here, let run_atlas_auto handle retries
#
#     # 4. ATLAS OUTPUT -> UPDATE USIM OUTPUT CARS & HH_CARS
#     atlas_post_hash = provenance_tracker.start_model_run(
#         f"{vehicle_ownership_model}_postprocessor",
#         yr,
#         description="Atlas postprocessing",
#     )
#     # Use the postprocess method of AtlasPostprocessor
#     postprocessor = atlas_post.AtlasPostprocessor()
#     postprocessor.postprocess(
#         RecordStore(),
#         None,
#         state,
#         workspace,
#         provenance_tracker,
#         atlas_post_hash,
#     )
#     provenance_tracker.complete_model_run(atlas_post_hash)
#
#     return success


# def run_atlas_auto(
#     settings,
#     state: WorkflowState,
#     client,
#     workspace: Workspace,
#     provenance_tracker: FileProvenanceTracker,
#     warm_start_atlas,
#     forecast,
# ) -> bool:
#     """
#     Run Atlas with automatic retries on failure.
#     """
#     max_retries = settings.get("atlas_max_retries", 3)
#     for i in range(max_retries):
#         success = run_atlas(
#             settings,
#             state,
#             client,
#             workspace,
#             provenance_tracker,
#             warm_start_atlas,
#             forecast=forecast,
#             atlas_run_count=i + 1,
#         )
#         if success:
#             return True
#         else:
#             logger.warning(
#                 f"Atlas run failed on attempt {i + 1}. Retrying... ({max_retries - i - 1} retries left)"
#             )
#     return False

## Atlas vehicle ownership model volume mount defintion, equivalent to
## docker run -v atlas_host_input_folder:atlas_container_input_folder
def get_atlas_docker_vols(settings, workspace: Workspace):
    atlas_host_input_folder = workspace.get_atlas_mutable_input_dir()
    atlas_host_output_folder = workspace.get_atlas_output_dir()

    atlas_container_input_folder = os.path.abspath(
        settings["atlas_container_input_folder"]
    )
    atlas_container_output_folder = os.path.abspath(
        settings["atlas_container_output_folder"]
    )
    atlas_docker_vols = {
        atlas_host_input_folder: {  ## source location, aka "local"
            "bind": atlas_container_input_folder,  ## destination loc, aka "remote", "client"
            "mode": "rw",
        },
        atlas_host_output_folder: {"bind": atlas_container_output_folder, "mode": "rw"},
    }
    return atlas_docker_vols




## For Atlas container command
def get_atlas_cmd(
    settings,
    freq,
    output_year,
    npe,
    nsample,
    beamac,
    mod,
    adscen,
    rebfactor,
    taxfactor,
    discIncent,
):
    basedir = settings.get("basedir", "/")
    codedir = settings.get("codedir", "/")
    formattable_atlas_cmd = settings["atlas_formattable_command"]
    atlas_cmd = formattable_atlas_cmd.format(
        freq,
        output_year,
        npe,
        nsample,
        basedir,
        codedir,
        beamac,
        mod,
        adscen,
        rebfactor,
        taxfactor,
        discIncent,
    )
    return atlas_cmd


class AtlasRunner(GenericRunner):
    """
    Runner for the ATLAS vehicle ownership model.
    This class is responsible for running the ATLAS containerized model.
    All preprocessing and postprocessing should be handled outside this method,
    following the BEAM/ActivitySim pattern.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def run(
        self,
        store: RecordStore,
        state: WorkflowState,
        workspace: Workspace,
        provenance_tracker: FileProvenanceTracker,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Executes the ATLAS model run.

        Args:
            store (RecordStore): The input data generated by the preprocessor.
            state (WorkflowState): The workflow state.
            workspace (Workspace): The workspace object for path management.
            provenance_tracker (FileProvenanceTracker): The provenance tracker.

        Returns:
            tuple: A tuple containing:
                - data (RecordStore): The raw output files that have been prepared to run the model
                - model_run_info (ModelRunInfo): Information about the model run

        Notes:
            - All preprocessing and postprocessing should be handled outside this method.
            - This method only runs the ATLAS container and records provenance for the run.
        """
        logger.info("[AtlasRunner] Starting ATLAS model run for year %s", state.current_year)
        settings = state.full_settings
        client = None
        if settings.get("container_manager") == "docker":
            try:
                client = self.initialize_docker_client(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")

        model_name = self.model_name
        atlas_image = settings[f"{settings['container_manager']}_images"][model_name]
        atlas_docker_vols = get_atlas_docker_vols(settings, workspace)
        freq = settings.get("vehicle_ownership_freq", False)
        npe = settings.get("atlas_num_processes", False)
        nsample = settings.get("atlas_sample_size", False)
        beamac = settings.get("atlas_beamac", 0)
        mod = settings.get("atlas_mod", 1)
        adscen = settings.get("atlas_adscen", False)
        rebfactor = settings.get("atlas_rebfactor", 0)
        taxfactor = settings.get("atlas_taxfactor", 0)
        discIncent = settings.get("atlas_discIncent", 0)
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

        # Read input files from the preprocessor's output RecordStore
        # (store: RecordStore) and log them as inputs to the run
        input_records = []
        for record in store.all_records():
            if hasattr(record, "file_path"):
                input_records.append(
                    provenance_tracker.record_input_file(
                        model_name,
                        record.file_path,
                        description=record.description,
                        short_name=record.short_name,
                    )
                )

        model_run_hash = provenance_tracker.start_model_run(
            model_name,
            state.current_year,
            state.current_inner_iter,
            description="ATLAS run",
            inputs=store,
        )

        logger.info("[AtlasRunner] Running Atlas vehicle ownership model container for year %s", state.current_year)
        success = GenericRunner.run_container(
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
        run_info = provenance_tracker.run_info.model_runs.get(model_run_hash)
        logger.info("[AtlasRunner] ATLAS model run complete for year %s (success=%s)", state.current_year, success)
        return output_store, run_info

