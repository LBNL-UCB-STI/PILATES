from typing import Tuple, Optional
import logging
import os

from pilates.generic.model import provenance_logging
from pilates.generic.runner import GenericRunner
from pilates.generic.records import RecordStore, ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker


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
#     freq = get_setting(settings, "run.vehicle_ownership_freq", False)
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
#             beam_output_dir, get_setting(settings, "shared.skims.fname")
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
        settings.atlas.container_input_folder
    )
    atlas_container_output_folder = os.path.abspath(
        settings.atlas.container_output_folder
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
    basedir = settings.atlas.basedir
    codedir = settings.atlas.codedir
    formattable_atlas_cmd = settings.atlas.command_template
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

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: FileProvenanceTracker,
        major_stage: Optional["WorkflowState.Stage"] = None,
    ):
        super().__init__(model_name, state, provenance_tracker, major_stage)

    @provenance_logging
    def _run(
        self,
        store: RecordStore,
        workspace: Workspace,
    ) -> Tuple[RecordStore, ModelRunInfo]:
        """
        Executes the ATLAS model run.

        Args:
            store (RecordStore): The input data generated by the preprocessor.
            workspace (Workspace): The workspace object for path management.

        Returns:
            tuple: A tuple containing:
                - data (RecordStore): The raw output files that have been prepared to run the model
                - model_run_info (ModelRunInfo): Information about the model run

        Notes:
            - All preprocessing and postprocessing should be handled outside this method.
            - This method only runs the ATLAS container and records provenance for the run.
        """
        logger.info(
            "[AtlasRunner] Starting ATLAS model run for year %s",
            self.state.current_year,
        )
        settings = self.state.full_settings
        self.setup_container_cache_dirs(settings)

        # Get container configuration
        vehicle_ownership_model, atlas_image = self.get_model_and_image(
            settings, "vehicle_ownership_model"
        )

        atlas_docker_vols = get_atlas_docker_vols(settings, workspace)
        freq = settings.run.vehicle_ownership_freq
        npe = settings.atlas.num_processes
        container_input_dir = settings.atlas.container_input_folder
        container_output_dir = settings.atlas.container_output_folder
        sample_size = settings.atlas.sample_size
        # BEAMAC is an integer flag that controls accessibility calculation
        # 0 = no accessibility calculation
        # 1 = use BEAM skims to calculate accessibility
        # 2 = use BEAM skims to calculate accessibility, but only for the first year
        beamac = settings.atlas.beamac
        mod = settings.atlas.mod
        adscen = settings.atlas.adscen
        rebfactor = settings.atlas.rebfactor
        taxfactor = settings.atlas.taxfactor
        discIncent = settings.atlas.discIncent
        atlas_cmd = get_atlas_cmd(
            settings,
            freq,
            self.state.forecast_year,
            npe,
            sample_size,
            beamac,
            mod,
            adscen,
            rebfactor,
            taxfactor,
            discIncent,
        )

        # Initialize container client
        client = None
        if settings.infrastructure.container_manager == "docker":
            try:
                client = self.initialize_docker_client(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                raise

        # Execute container
        try:
            logger.info(
                "[AtlasRunner] Running Atlas vehicle ownership model container for year %s",
                self.state.current_year,
            )
            max_retries = settings.atlas.max_retries
            success = False

            for i in range(max_retries):
                success = self.run_container(
                    client=client,
                    settings=settings,
                    image=atlas_image,
                    volumes=atlas_docker_vols,
                    command=atlas_cmd,
                    model_name=self.model_name,
                    working_dir="/",
                    provenance_tracker=self.provenance_tracker,
                    output_paths=[workspace.get_atlas_output_dir()],
                )

                if not success:
                    logger.error(f"ATLAS container execution failed in attempt {i + 1}")
                else:
                    logger.info(f"ATLAS container execution succeeded in attempt {i + 1}")
                    break

            if not success:
                raise RuntimeError("ATLAS container execution failed after all retry attempts")

        except Exception as e:
            logger.error(f"ATLAS container execution error: {e}")
            raise

        # Collect outputs
        output_records = []
        atlas_output_dir = workspace.get_atlas_output_dir()
        output_year = self.state.forecast_year

        # TODO: Add comprehensive output file detection
        # ATLAS generates multiple output files that should be recorded
        expected_outputs = [
            f"householdv_{output_year}.csv",
            f"vehicles_{output_year}.csv",
        ]

        for output_file in expected_outputs:
            output_path = os.path.join(atlas_output_dir, output_file)
            if os.path.exists(output_path):
                # Output file provenance is automatically tracked by @provenance_logging decorator
                from pilates.generic.records import FileRecord

                output_record = FileRecord(
                    file_path=output_path,
                    models=[self.model_name],
                    year=output_year,
                    description=f"ATLAS {output_file} output for year {output_year}",
                    short_name=output_file.replace(".csv", ""),
                )
                output_records.append(output_record)
                logger.info(f"Recorded ATLAS output: {output_path}")
            else:
                logger.warning(f"Expected ATLAS output file not found: {output_path}")

        # Prepare runtime metadata
        runtime_metadata = {
            "container_command": atlas_cmd,
            "runtime_parameters": {
                "freq": freq,
                "output_year": self.state.forecast_year,
                "npe": npe,
                "nsample": sample_size,
                "beamac": beamac,
                "mod": mod,
                "adscen": adscen,
                "rebfactor": rebfactor,
                "taxfactor": taxfactor,
                "discIncent": discIncent,
            },
            "container_image": atlas_image,
            "container_manager": settings.infrastructure.container_manager,
            "working_directory": "/",
        }

        logger.info(
            "[AtlasRunner] ATLAS model run complete for year %s (outputs=%d)",
            self.state.current_year,
            len(output_records),
        )

        return RecordStore(recordList=output_records), ModelRunInfo(
            model=self.model_name, year=self.state.current_year
        )
