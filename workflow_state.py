from datetime import datetime
from enum import Enum
from typing import Optional, List
import os
import yaml
import json
import logging
from pilates.activitysim import preprocessor as asim_pre
from pilates.urbansim import preprocessor as usim_pre
from pilates.beam import preprocessor as beam_pre
from pilates.atlas import preprocessor as atlas_pre
from pilates.utils.provenance import ProvenanceTracker  # Import the new class
import uuid

logger = logging.getLogger(__name__)


class WorkflowState:
    # Define all possible stages
    Stage = Enum(
        "WorkflowStage",
        [
            "initialize_data",
            "land_use",
            "vehicle_ownership_model",
            "supply_demand_loop",
            "activity_demand",
            "activity_demand_directly_from_land_use",
            "traffic_assignment",
            "postprocessing",
        ],
    )

    def __init__(
        self,
        start_year: int,
        end_year: int,
        travel_model_freq: int,
        land_use_enabled: bool,
        vehicle_ownership_model_enabled: bool,
        activity_demand_enabled: bool,
        traffic_assignment_enabled: bool,
        replanning_enabled: bool,
        year: int | None,
        major_stage: Stage | None,
        inner_iter: int,
        sub_stage: Stage | None,
        output_path: str | None,
        folder_name: str | None,
        file_loc: str,
        asim_compiled: bool,
        run_id: str | None = None,
    ):

        # Store basic simulation parameters
        self.start_year = start_year
        self.end_year = end_year
        self.travel_model_freq = travel_model_freq

        # State variables for tracking progress
        self.current_year = year or start_year
        self.current_major_stage = major_stage
        self.current_inner_iter = inner_iter
        self.current_sub_stage = sub_stage

        self.forecast_year = None
        self.output_path = os.path.abspath(output_path) if output_path else None
        self.folder_name = folder_name
        self.file_loc = file_loc
        self.run_id = run_id

        self.__asim_compiled = asim_compiled
        self.initial_step = 7 if self.current_year == 2010 else None

        # Initialize provenance tracker if output path and run_id are available
        self.provenance_tracker: Optional[ProvenanceTracker] = None
        if self.output_path and self.run_id:
            # Ensure the output directory exists before initializing tracker
            if self.folder_name:
                tracker_output_path = os.path.join(self.output_path, self.folder_name)
            else:
                tracker_output_path = (
                    self.output_path
                )  # Should not happen if folder_name is None with output_path
            os.makedirs(tracker_output_path, exist_ok=True)
            self.provenance_tracker = ProvenanceTracker(
                self.run_id, self.output_path, self.folder_name
            )

        # Store settings for access by methods that need them
        self._settings = {
            "end_year": end_year,
            "supply_demand_iters": 1,  # Default, will be updated in from_settings
            "land_use_enabled": land_use_enabled,
            "vehicle_ownership_model_enabled": vehicle_ownership_model_enabled,
            "activity_demand_enabled": activity_demand_enabled,
            "traffic_assignment_enabled": traffic_assignment_enabled,
        }

        # Determine what stages are enabled
        self.enabled_stages = set()

        if land_use_enabled:
            self.enabled_stages.add(self.Stage.land_use)
        if vehicle_ownership_model_enabled:
            self.enabled_stages.add(self.Stage.vehicle_ownership_model)
        if activity_demand_enabled:
            self.enabled_stages.add(self.Stage.activity_demand)
        elif land_use_enabled:  # Only if land use is enabled but activity demand is not
            self.enabled_stages.add(self.Stage.activity_demand_directly_from_land_use)
        if traffic_assignment_enabled:
            self.enabled_stages.add(self.Stage.traffic_assignment)

        # Define the order of major stages
        self.major_stage_order = [
            self.Stage.land_use,
            self.Stage.vehicle_ownership_model,
            self.Stage.supply_demand_loop,
            self.Stage.postprocessing,  # Added postprocessing as a major stage
        ]

        # If we have activity demand OR traffic assignment, we need the supply-demand loop
        if activity_demand_enabled or traffic_assignment_enabled:
            self.enabled_stages.add(self.Stage.supply_demand_loop)
            if self.Stage.supply_demand_loop not in self.major_stage_order:
                self.major_stage_order.append(self.Stage.supply_demand_loop)
                logger.debug("Added supply_demand_loop to major_stage_order")

        # Define what happens inside the supply-demand loop
        self.loop_substages = []
        if activity_demand_enabled:
            self.loop_substages.append(self.Stage.activity_demand)
        elif self.Stage.activity_demand_directly_from_land_use in self.enabled_stages:
            self.loop_substages.append(
                self.Stage.activity_demand_directly_from_land_use
            )
        if traffic_assignment_enabled:
            self.loop_substages.append(self.Stage.traffic_assignment)

        # For backward compatibility
        self.enabled_individual_stages = self.enabled_stages
        self.MAJOR_STAGE_SEQUENCE = self.major_stage_order
        self.SUB_STAGE_SEQUENCE = self.loop_substages

    @property
    def settings(self):
        return (
            self._settings
            if hasattr(self, "_settings")
            else {"end_year": self.end_year}
        )

    @property
    def full_path(self):
        """
        Returns the full path where mutable data should be stored.
        If output_path is None, returns the current working directory (for in-place updates).
        Otherwise, returns the path to the run-specific directory.
        """
        if self.output_path is None:
            # No output directory specified - work in place (current directory)
            return os.getcwd()
        elif self.folder_name:
            # Output directory with folder name (no run ID subdirectory)
            return os.path.join(self.output_path, self.folder_name)
        else:
            # Just the base output path (should ideally have a folder_name)
            return self.output_path

    @property
    def asim_compiled(self):
        if self.current_year is not None:
            if self.__asim_compiled:
                logger.info(
                    "ActivitySim already compiled in year %s", self.current_year
                )
            else:
                logger.info(
                    "ActivitySim not compiled in year %s, so running compilation (this will take longer)",
                    self.current_year,
                )
        return self.__asim_compiled

    def compile_asim(self):
        self.__asim_compiled = True
        logger.info("Completed compiling activitysim in year %s", self.current_year)
        self.write_state()

    @property
    def year(self):
        return self.current_year

    @property
    def stage(self):
        return self.current_sub_stage or self.current_major_stage

    @stage.setter
    def stage(self, value):
        if value is None:
            self.current_sub_stage = None
        else:
            self.current_sub_stage = value

    @property
    def iteration(self):
        return self.current_inner_iter

    @iteration.setter
    def iteration(self, value):
        self.current_inner_iter = value

    def enabled(self, stage):
        return stage in self.enabled_stages

    def _create_output_dir(self, settings: dict):
        """
        Creates output directory structure and copies input data to mutable locations.
        Only called when output_directory is specified in settings and it's a new run.
        """
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_loc = os.path.expandvars(settings["output_directory"])
        run_name = settings["output_run_name"]
        folder_name = "{0}-{1}-{2}".format(settings["region"], run_name, dt)
        base_folder_path = os.path.join(base_loc, folder_name)

        self.run_id = str(uuid.uuid4())
        os.makedirs(base_folder_path, exist_ok=True)

        self.output_path = base_loc
        self.folder_name = folder_name

        logger.info(
            f"Created output directory structure for run {self.run_id}: {base_folder_path}"
        )

        # Initialize provenance tracker for the new run
        self.provenance_tracker = ProvenanceTracker(
            self.run_id, self.output_path, self.folder_name
        )
        self.provenance_tracker.initialize_from_settings(settings)

        # Create subdirectories for models and copy input data to mutable locations
        # Record initial input files as they are copied
        have_not_copied_usim_data = True
        for model_key in [
            "travel_model",
            "activity_demand_model",
            "vehicle_ownership_model",
            "land_use_model",
        ]:
            model_name = settings.get(model_key)
            if model_name:
                model_output_base = os.path.join(base_folder_path, model_name)
                os.makedirs(model_output_base, exist_ok=True)

                if (model_name == "urbansim") or (
                    (model_name == "activitysim") and have_not_copied_usim_data
                ):
                    output_dir = os.path.join(
                        base_folder_path, settings["usim_local_mutable_data_folder"]
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    # Record UrbanSim input files before copying
                    input_dir = settings["usim_local_data_input_folder"]
                    if os.path.exists(input_dir):
                        for file in os.listdir(input_dir):
                            input_path = os.path.join(self.full_path, input_dir, file)
                            if os.path.isfile(input_path):
                                logger.info(f"Recording input file for {model_name}: {input_path}")
                                self.record_input_file(
                                    model_name,
                                    input_path,
                                    description="Base UrbanSim input data",
                                )

                    usim_pre.copy_data_to_mutable_location(settings, output_dir, self)
                    have_not_copied_usim_data = False

                if model_name == "beam":
                    input_dir = os.path.join(
                        base_folder_path, settings["beam_local_mutable_data_folder"]
                    )
                    os.makedirs(input_dir, exist_ok=True)

                    # Record BEAM input files before copying
                    beam_input_dir = os.path.join(self.full_path, settings["beam_local_input_folder"], settings["region"])
                    if os.path.exists(beam_input_dir):
                        if self.provenance_tracker.is_git_repo(beam_input_dir):
                            repo_name = os.path.basename(beam_input_dir)
                            git_hash = self.provenance_tracker.get_git_hash(beam_input_dir)
                            self.record_input_file(
                                model_name,
                                beam_input_dir,
                                description=f"Git repo {repo_name} at {git_hash}",
                            )
                        # Always copy data to mutable location
                        beam_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(
                        base_folder_path, settings["beam_local_output_folder"]
                    )
                    os.makedirs(output_dir, exist_ok=True)

                if model_name == "atlas":
                    input_dir = os.path.join(
                        base_folder_path, settings["atlas_host_mutable_input_folder"]
                    )
                    os.makedirs(input_dir, exist_ok=True)

                    # Record Atlas input files before copying
                    atlas_input_dir = os.path.join(self.full_path, settings.get("atlas_host_input_folder"))
                    if atlas_input_dir and os.path.exists(atlas_input_dir):
                        for root, dirs, files in os.walk(atlas_input_dir):
                            for file in files:
                                input_path = os.path.join(root, file)
                                if os.path.isfile(input_path):
                                    self.record_input_file(
                                        model_name,
                                        input_path,
                                        description="Atlas input file",
                                    )

                    atlas_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(
                        base_folder_path, settings["atlas_host_output_folder"]
                    )
                    os.makedirs(output_dir, exist_ok=True)

                if model_name == "activitysim":
                    # Record ActivitySim config files before copying
                    asim_config_dir = settings.get("asim_local_configs_folder")
                    if asim_config_dir and os.path.exists(asim_config_dir):
                        for root, dirs, files in os.walk(asim_config_dir):
                            for file in files:
                                input_path = os.path.join(root, file)
                                if os.path.isfile(input_path):
                                    self.record_input_file(
                                        model_name,
                                        input_path,
                                        description="ActivitySim configuration file",
                                    )

                    # Check if the ActivitySim configs folder is a git repository
                    asim_config_dir = os.path.join(settings['asim_local_configs_folder'], settings['region'])
                    if os.path.exists(asim_config_dir):
                        if self.provenance_tracker.is_git_repo(asim_config_dir):
                            repo_name = os.path.basename(asim_config_dir)
                            git_hash = self.provenance_tracker.get_git_hash(asim_config_dir)
                            self.provenance_tracker.record_repo_input(
                                model_name,
                                asim_config_dir,
                                description=f"ActivitySim configuration repository",
                                git_hash=git_hash,
                            )
                        # Always copy data to mutable location
                        asim_pre.copy_data_to_mutable_location(settings, base_folder_path)
                    output_dir = os.path.join(
                        base_folder_path, settings["asim_local_output_folder"]
                    )
                    os.makedirs(output_dir, exist_ok=True)

    def record_model_start(
        self, model: str, year: int = None, iteration: int = None
    ) -> int:
        """Record the start of a model run and return the run index."""
        if self.provenance_tracker:
            return self.provenance_tracker.start_model_run(model, year, iteration)
        return -1  # Return -1 if tracker is not initialized

    def record_model_completion(self, run_index: int, status: str = "completed"):
        """Record the completion of a model run."""
        if self.provenance_tracker and run_index >= 0:
            self.provenance_tracker.complete_model_run(run_index, status)

    def record_input_file(
        self,
        model: str,
        file_path: str,
        source_run_id: str = None,
        description: str = None,
    ):
        """Record an input file for provenance tracking."""
        if self.provenance_tracker:
            # Ensure file_path is absolute before passing to tracker
            abs_file_path = os.path.abspath(file_path)
            logger.debug(f"Recording input file: {file_path}")
            if os.path.exists(abs_file_path):
                logger.debug(f"File exists: {abs_file_path}")
                self.provenance_tracker.record_input_file(
                    model, abs_file_path, source_run_id, description
                )
            else:
                logger.warning(f"Input file not found: {abs_file_path}")

    def record_model_io_batch(
        self,
        model: str,
        inputs: List[str] = None,
        outputs: List[str] = None,
        year: int = None,
        input_descriptions: List[str] = None,
        output_descriptions: List[str] = None,
    ):
        """Record multiple input and output files for a model in batch."""
        if self.provenance_tracker:
            self.provenance_tracker.record_model_io_batch(
                model, inputs, outputs, year, input_descriptions, output_descriptions
            )

    def record_output_file(
        self, model: str, file_path: str, year: int = None, description: str = None
    ):
        """Record an output file for provenance tracking."""
        if self.provenance_tracker:
            # Ensure file_path is absolute before passing to tracker
            abs_file_path = os.path.abspath(file_path)
            if os.path.exists(abs_file_path):
                self.provenance_tracker.record_output_file(
                    model, abs_file_path, year, description
                )
            else:
                logger.warning(f"Output file not found: {abs_file_path}")

    @classmethod
    def write_stage(
        cls,
        year: int,
        current_stage: Stage,
        file_loc,
        path,
        folder_name,
        iteration,
        asim_compiled,
        run_id: str | None,
    ):
        to_save = {
            "year": year,
            "stage": current_stage.name if current_stage else None,
            "path": path,
            "folder_name": folder_name,
            "iteration": iteration,
            "asim_compiled": asim_compiled,
            "run_id": run_id,
        }  # Save run_id
        with open(file_loc, mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls, file_loc):
        if not os.path.exists(file_loc):
            logger.info("Creating new stage info at {}".format(file_loc))
            return [None, None, 0, None, None, False, None]  # Return None for run_id
        with open(file_loc, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            data = data if data is not None else {}
            year = data.get("year", None)
            stage_str = data.get("stage", "null")
        stage = (
            None
            if stage_str is None or stage_str == "null"
            else WorkflowState.Stage[stage_str]
        )
        path = data.get("path", None)
        folder_name = data.get("folder_name", None)
        iteration = data.get("iteration", 0) or 0
        asim_compiled = data.get("asim_compiled", False)
        run_id = data.get("run_id", None)  # Read run_id
        return [year, stage, iteration, path, folder_name, asim_compiled, run_id]

    @classmethod
    def from_settings(cls, settings):
        start_year = settings["start_year"]
        end_year = settings["end_year"]
        travel_model_freq = settings.get("travel_model_freq", 1)
        land_use_enabled = settings["land_use_enabled"]
        vehicle_ownership_model_enabled = settings["vehicle_ownership_model_enabled"]
        activity_demand_enabled = settings["activity_demand_enabled"]
        traffic_assignment_enabled = settings["traffic_assignment_enabled"]
        replanning_enabled = settings["replanning_enabled"]
        file_loc = settings["state_file_loc"]
        output_directory = settings.get("output_directory", None)
        # copy_files = output_directory is not None # This logic was confusing, simplified below

        [year, stage, iteration, path, folder_name, asim_compiled, run_id] = (
            cls.read_current_stage(file_loc)
        )

        current_run_id = run_id  # Use loaded run_id if available

        year = year or start_year

        out = cls(
            start_year,
            end_year,
            travel_model_freq,
            land_use_enabled,
            vehicle_ownership_model_enabled,
            activity_demand_enabled,
            traffic_assignment_enabled,
            replanning_enabled,
            year,
            stage,
            iteration,
            None,
            path,
            folder_name,
            file_loc,
            asim_compiled,
            run_id=current_run_id,
        )

        out._settings["supply_demand_iters"] = settings.get("supply_demand_iters", 1)

        # Handle output directory logic and provenance tracker initialization
        if output_directory is not None:
            base_loc = os.path.expandvars(output_directory)
            if out.output_path is None or out.folder_name is None:
                # This is a new run or resuming a run that didn't properly save path/folder_name
                # Create new output directory structure and initialize tracker
                out._create_output_dir(settings)
                out.write_state()  # Save the newly created path, folder_name, run_id
            else:
                # Resuming an existing run with a valid output directory
                logger.info(
                    f"Resuming run {out.run_id} in existing output directory: {out.full_path}"
                )
                # Provenance tracker was already initialized in __init__ if path/folder_name/run_id were loaded
                # Ensure settings are initialized in the tracker if it's a resume
                if (
                    out.provenance_tracker
                    and out.provenance_tracker.run_info.get("settings_hash") is None
                ):
                    out.provenance_tracker.initialize_from_settings(settings)

        else:
            # output_directory is None - work in place with local inputs
            out.output_path = None
            out.folder_name = None
            # If no output directory, no file-based provenance tracking
            out.provenance_tracker = None
            if out.run_id is None:
                # Still generate a run_id even for in-place runs, just don't track to file
                out.run_id = str(uuid.uuid4())
                out.write_state()

        if year:
            # Calculate forecast_year based on current_year and frequency
            # Ensure forecast_year does not exceed end_year
            next_year_candidate = out.current_year + (
                out.initial_step if out.current_year == 2010 else out.travel_model_freq
            )
            out.forecast_year = (
                min(next_year_candidate, out.end_year)
                if land_use_enabled
                else out.start_year
            )

        # Initialize state if starting fresh (current_major_stage is None)
        if out.current_major_stage is None:
            out._initialize_first_stage()

        return out

    def _initialize_first_stage(self):
        """Set up the initial stage when starting a new workflow"""
        # Find the first enabled major stage
        for stage in self.major_stage_order:
            if stage in self.enabled_stages:
                self.current_major_stage = stage
                self.current_inner_iter = 0
                self.current_sub_stage = None

                # If starting with the supply-demand loop, set up the first substage
                if stage == self.Stage.supply_demand_loop and self.loop_substages:
                    self.current_sub_stage = self.loop_substages[0]

                logger.info(f"Starting workflow with stage: {stage.name}")
                self.write_state()
                break
        # If no enabled stages found, the workflow is effectively complete
        if self.current_major_stage is None:
            logger.info("No enabled stages found. Workflow is complete.")
            self._advance_to_next_year()  # This will set current_major_stage to None and handle state file removal

    def write_state(self):
        """Save the current state to file"""
        WorkflowState.write_stage(
            self.current_year,
            self.current_sub_stage or self.current_major_stage,
            self.file_loc,
            self.output_path,
            self.folder_name,
            self.current_inner_iter,
            self.__asim_compiled,
            self.run_id,  # Pass run_id to write
        )

    def is_enabled(self, stage: Stage) -> bool:
        """Checks if a stage is enabled in settings."""
        return stage in self.enabled_stages

    def should_run(
        self,
        target_major: Stage,
        target_inner_iter: int = 0,
        target_sub_stage: Optional[Stage] = None,
    ) -> bool:
        """
        Simplified logic: check if we should run the target stage/iteration/substage
        """
        # logger.debug(f"should_run check: Current=(Year={self.current_year}, Major={self.current_major_stage.name if self.current_major_stage else None}, Iter={self.current_inner_iter}, Sub={self.current_sub_stage.name if self.current_sub_stage else None}), Target=(Major={target_major.name}, Iter={target_inner_iter}, Sub={target_sub_stage.name if target_sub_stage else None})")

        # Check if the target stage is enabled
        check_stage = target_sub_stage if target_sub_stage is not None else target_major
        if not self.is_enabled(check_stage):
            # logger.debug(f"should_run: Target stage {check_stage.name} is disabled.")
            return False

        # If we haven't started yet, allow any enabled stage
        if self.current_major_stage is None:
            # logger.debug("should_run: No current stage, allowing run.")
            return True

        # For major stages (not substages)
        if target_sub_stage is None:
            # Check if the target major stage is the current one or comes after it
            try:
                current_idx = self.major_stage_order.index(self.current_major_stage)
                target_idx = self.major_stage_order.index(target_major)
                result = current_idx <= target_idx
                # logger.debug(f"should_run: Current major idx {current_idx} <= target idx {target_idx}? {result}")
                return result
            except ValueError:
                logger.warning(
                    f"Target major stage {target_major.name} not found in order, allowing run"
                )
                logger.debug(
                    f"Current major stage: {self.current_major_stage.name if self.current_major_stage else 'None'}"
                )
                logger.debug(
                    f"Major stage order: {[stage.name for stage in self.major_stage_order]}"
                )
                logger.debug(
                    f"Enabled stages: {[stage.name for stage in self.enabled_stages]}"
                )
                logger.debug(
                    f"Current major stage: {self.current_major_stage.name if self.current_major_stage else 'None'}"
                )
                logger.debug(
                    f"Major stage order: {[stage.name for stage in self.major_stage_order]}"
                )
                logger.debug(
                    f"Enabled stages: {[stage.name for stage in self.enabled_stages]}"
                )
                return True

        # For substages within the supply-demand loop
        if target_sub_stage is not None:
            # Must be in the supply-demand loop to run substages
            if self.current_major_stage != self.Stage.supply_demand_loop:
                # logger.debug(f"should_run: Not in loop, can't run substage {target_sub_stage.name}")
                return False

            # Check iteration first
            total_iters = self._settings.get("supply_demand_iters", 1)
            if self.current_inner_iter >= total_iters:
                # logger.debug(f"should_run: Past max iterations ({self.current_inner_iter} >= {total_iters})")
                return False

            if self.current_inner_iter != target_inner_iter:
                # logger.debug(f"should_run: Wrong iteration ({self.current_inner_iter} != {target_inner_iter})")
                return False

            # Check substage position
            if self.current_sub_stage is None:
                # At start of iteration, can run any substage
                # logger.debug("should_run: At start of iteration, allowing substage run")
                return True
            else:
                try:
                    current_sub_idx = self.loop_substages.index(self.current_sub_stage)
                    target_sub_idx = self.loop_substages.index(target_sub_stage)
                    result = current_sub_idx <= target_sub_idx
                    # logger.debug(f"should_run: Current sub idx {current_sub_idx} <= target sub idx {target_sub_idx}? {result}")
                    return result
                except ValueError:
                    logger.error(
                        f"Target substage {target_sub_stage.name} not found in sequence"
                    )
                    return False

        # logger.debug("should_run: No case matched, denying run")
        return False

    def is_start_year(self):
        return self.year == self.start_year

    def __iter__(self):
        """Iterator yields the current year as long as the workflow hasn't finished."""
        return self

    def __next__(self):
        """Returns the current year if the workflow is not finished, otherwise raises StopIteration."""
        # Check if the workflow is finished (current_year > end_year and no active stage)
        if (
            self.current_year is not None
            and self.current_year > self.end_year
            and self.current_major_stage is None
        ):
            # Clean up state file if finished
            if os.path.exists(self.file_loc):
                logger.info(f"Workflow finished. Removing state file: {self.file_loc}")
                os.remove(self.file_loc)
            raise StopIteration

        # If current_year is None, it means read_current_stage returned None for year,
        # which should only happen for a brand new run, where current_major_stage
        # would be initialized in from_settings. If current_major_stage is also None here,
        # it implies an issue or an empty workflow.
        if self.current_year is None and self.current_major_stage is None:
            logger.warning(
                "Workflow state indicates no current year and no active stage. Stopping."
            )
            raise StopIteration

        # If we are past the end year but still have an active stage, something is wrong
        if (
            self.current_year is not None
            and self.current_year > self.end_year
            and self.current_major_stage is not None
        ):
            logger.error(
                f"Workflow is past end year ({self.end_year}) but still in stage {self.current_major_stage.name}. Stopping."
            )
            raise StopIteration

        return self.current_year

    def complete_step(
        self,
        completed_major: Stage,
        completed_inner_iter: int = 0,
        completed_sub: Optional[Stage] = None,
    ):
        """
        Simplified state transition logic
        """
        logger.info(
            "Completed step: (Major: %s, Iter: %s, Sub: %s) for year %d",
            completed_major.name if completed_major else None,
            completed_inner_iter,
            completed_sub.name if completed_sub else None,
            self.current_year,
        )

        # Handle substage completion within supply-demand loop
        if completed_sub is not None:
            if self.current_major_stage != self.Stage.supply_demand_loop:
                logger.error(
                    f"Completed substage {completed_sub.name} but not in supply-demand loop. State inconsistency."
                )
                # Attempt to recover by advancing major stage
                self._advance_to_next_major_stage()
                self.write_state()
                return

            try:
                current_sub_idx = self.loop_substages.index(completed_sub)
                if current_sub_idx < len(self.loop_substages) - 1:
                    # Move to next substage in same iteration
                    self.current_sub_stage = self.loop_substages[current_sub_idx + 1]
                    logger.debug(
                        f"Moving to next substage: {self.current_sub_stage.name}"
                    )
                else:
                    # Last substage of iteration completed
                    total_iters = self._settings.get("supply_demand_iters", 1)
                    if completed_inner_iter < total_iters - 1:
                        # Start next iteration
                        self.current_inner_iter = completed_inner_iter + 1
                        self.current_sub_stage = (
                            self.loop_substages[0] if self.loop_substages else None
                        )
                        logger.debug(
                            f"Starting next iteration: {self.current_inner_iter}"
                        )
                    else:
                        # Loop completed, move to next major stage
                        self._advance_to_next_major_stage()
            except ValueError:
                logger.error(
                    f"Completed substage {completed_sub.name} not found in sequence. State inconsistency."
                )
                # Attempt to recover by advancing major stage
                self._advance_to_next_major_stage()

        # Handle major stage completion
        elif completed_major is not None:
            if self.current_major_stage != completed_major:
                logger.error(
                    f"Completed major stage {completed_major.name} but current major stage is {self.current_major_stage.name}. State inconsistency."
                )
                # Attempt to recover by advancing major stage from current state
                self._advance_to_next_major_stage()
            else:
                # Regular major stage completed
                self._advance_to_next_major_stage()

        self.write_state()

    def _advance_to_next_major_stage(self):
        """Move to the next enabled major stage or next year"""
        if self.current_major_stage is None:
            # Should not happen if workflow is running, but handle defensively
            self._advance_to_next_year()
            return

        try:
            current_idx = self.major_stage_order.index(self.current_major_stage)
            next_major = None

            # Find next enabled major stage
            for i in range(current_idx + 1, len(self.major_stage_order)):
                if self.major_stage_order[i] in self.enabled_stages:
                    next_major = self.major_stage_order[i]
                    break

            if next_major:
                self.current_major_stage = next_major
                self.current_inner_iter = 0
                self.current_sub_stage = None

                # If entering supply-demand loop, set up first substage
                if next_major == self.Stage.supply_demand_loop and self.loop_substages:
                    self.current_sub_stage = self.loop_substages[0]

                logger.debug(f"Advanced to next major stage: {next_major.name}")
            else:
                # No more stages for this year, advance to next year
                logger.debug("No more enabled major stages for this year.")
                self._advance_to_next_year()

        except ValueError:
            logger.error(
                f"Current major stage {self.current_major_stage.name} not found in order. State inconsistency."
            )
            # Attempt to recover by advancing to next year
            self._advance_to_next_year()

    def _advance_to_next_year(self):
        """Move to the next year and reset to first stage"""
        self.current_year += 1

        if self.current_year <= self.end_year:
            logger.info(f"Starting year {self.current_year}")
            # Reset to the first enabled major stage for the new year
            self._initialize_first_stage()
            # Recalculate forecast year for the new current year
            next_year_candidate = self.current_year + (
                self.initial_step
                if self.current_year == 2010
                else self.travel_model_freq
            )
            self.forecast_year = (
                min(next_year_candidate, self.end_year)
                if self._settings.get("land_use_enabled")
                else self.start_year
            )

        else:
            logger.info(f"Workflow complete at end year {self.end_year}")
            self.current_year = (
                self.end_year + 1
            )  # Set year just past end_year to signal completion
            self.current_major_stage = None
            self.current_inner_iter = 0
            self.current_sub_stage = None
            # The __next__ method will handle state file removal when it detects completion

    def get_sub_stages_from(self, start_sub_stage: Optional[Stage]):
        """Returns the sequence of sub-stages starting from the given stage (inclusive)."""
        if not self.loop_substages:
            return []
        if start_sub_stage is None:
            return self.loop_substages
        try:
            start_index = self.loop_substages.index(start_sub_stage)
            return self.loop_substages[start_index:]
        except ValueError:
            logger.error(
                "Start sub-stage %s not found in sequence. Restarting from beginning.",
                start_sub_stage.name if start_sub_stage else None,
            )
            return self.loop_substages
