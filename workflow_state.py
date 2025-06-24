from datetime import datetime
from enum import Enum
from typing import Optional
import os
import yaml
import json
import logging
from pilates.activitysim import preprocessor as asim_pre
from pilates.urbansim import preprocessor as usim_pre
from pilates.beam import preprocessor as beam_pre
from pilates.atlas import preprocessor as atlas_pre
import uuid

logger = logging.getLogger(__name__)

class WorkflowState:
    # Define all possible stages
    Stage = Enum('WorkflowStage', [
        'initialize_data',
        'land_use',
        'vehicle_ownership_model',
        'supply_demand_loop',
        'activity_demand',
        'activity_demand_directly_from_land_use',
        'traffic_assignment',
        'postprocessing',
    ])

    def __init__(self, start_year: int, end_year: int, travel_model_freq: int, land_use_enabled: bool,
                 vehicle_ownership_model_enabled: bool, activity_demand_enabled: bool,
                 traffic_assignment_enabled: bool, replanning_enabled: bool,
                 year: int | None, major_stage: Stage | None, inner_iter: int, sub_stage: Stage | None,
                 output_path: str | None, folder_name: str | None, file_loc: str, asim_compiled: bool,
                 run_id: str | None = None):

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
        self.output_path = output_path
        self.folder_name = folder_name
        self.file_loc = file_loc
        self.run_id = run_id

        self.__asim_compiled = asim_compiled
        self.initial_step = 7 if self.current_year == 2010 else None
        
        # Store settings for access by methods that need them
        self._settings = {
            'end_year': end_year,
            'supply_demand_iters': 1,  # Default, will be updated in from_settings
            'land_use_enabled': land_use_enabled,
            'vehicle_ownership_model_enabled': vehicle_ownership_model_enabled,
            'activity_demand_enabled': activity_demand_enabled,
            'traffic_assignment_enabled': traffic_assignment_enabled
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

        # If we have activity demand OR traffic assignment, we need the supply-demand loop
        if activity_demand_enabled or traffic_assignment_enabled:
            self.enabled_stages.add(self.Stage.supply_demand_loop)

        # Define the order of major stages
        self.major_stage_order = [
            self.Stage.land_use,
            self.Stage.vehicle_ownership_model,
            self.Stage.supply_demand_loop,
            self.Stage.postprocessing
        ]

        # Define what happens inside the supply-demand loop
        self.loop_substages = []
        if activity_demand_enabled:
            self.loop_substages.append(self.Stage.activity_demand)
        elif self.Stage.activity_demand_directly_from_land_use in self.enabled_stages:
            self.loop_substages.append(self.Stage.activity_demand_directly_from_land_use)
        if traffic_assignment_enabled:
            self.loop_substages.append(self.Stage.traffic_assignment)

        # For backward compatibility
        self.enabled_individual_stages = self.enabled_stages
        self.MAJOR_STAGE_SEQUENCE = self.major_stage_order
        self.SUB_STAGE_SEQUENCE = self.loop_substages

    @property
    def settings(self):
        return self._settings if hasattr(self, '_settings') else {'end_year': self.end_year}

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
            # Just the base output path
            return self.output_path

    @property
    def asim_compiled(self):
        if self.current_year is not None:
            if self.__asim_compiled:
                logger.info("ActivitySim already compiled in year %s", self.current_year)
            else:
                logger.info("ActivitySim not compiled in year %s, so running compilation (this will take longer)",
                            self.current_year)
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
        Only called when output_directory is specified in settings.
        """
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_loc = os.path.expandvars(settings['output_directory'])
        run_name = settings['output_run_name']
        folder_name = "{0}-{1}-{2}".format(settings['region'], run_name, dt)
        base_folder_path = os.path.join(base_loc, folder_name)

        self.run_id = str(uuid.uuid4())
        os.makedirs(base_folder_path, exist_ok=True)

        self.output_path = base_loc
        self.folder_name = folder_name

        logger.info(f"Created output directory structure for run {self.run_id}: {base_folder_path}")

        # Write run ID to JSON file in the archive directory
        self._write_run_id_file()

        # Create subdirectories for models and copy input data to mutable locations
        have_not_copied_usim_data = True
        for model in ['travel_model', 'activity_demand_model', 'vehicle_ownership_model', 'land_use_model']:
            if settings.get(model) is not None:
                model_name = settings[model]
                model_output_base = os.path.join(base_folder_path, model_name)
                os.makedirs(model_output_base, exist_ok=True)

                if (model_name == "urbansim") | ((model_name == "activitysim") & have_not_copied_usim_data):
                    output_dir = os.path.join(base_folder_path, settings['usim_local_mutable_data_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                    usim_pre.copy_data_to_mutable_location(settings, output_dir)
                    have_not_copied_usim_data = False
                if model_name == "beam":
                    input_dir = os.path.join(base_folder_path, settings['beam_local_mutable_data_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    beam_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(base_folder_path, settings['beam_local_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                if model_name == "atlas":
                    input_dir = os.path.join(base_folder_path, settings['atlas_host_mutable_input_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    atlas_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(base_folder_path, settings['atlas_host_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                if model_name == "activitysim":
                    asim_pre.copy_data_to_mutable_location(settings, base_folder_path)
                    output_dir = os.path.join(base_folder_path, settings['asim_local_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)

    def _write_run_id_file(self):
        """Write run ID and metadata to JSON file in the archive directory"""
        if self.output_path and self.folder_name and self.run_id:
            run_info = {
                "run_id": self.run_id,
                "created_at": datetime.now().isoformat(),
                "start_year": self.start_year,
                "end_year": self.end_year
            }
            
            run_id_file = os.path.join(self.output_path, self.folder_name, "run_info.json")
            with open(run_id_file, 'w') as f:
                json.dump(run_info, f, indent=2)
            logger.info(f"Created run info file: {run_id_file}")

    @classmethod
    def write_stage(cls, year: int, current_stage: Stage, file_loc, path, folder_name, iteration, asim_compiled, run_id: str):
        to_save = {"year": year, "stage": current_stage.name if current_stage else None, "path": path,
                   "folder_name": folder_name, "iteration": iteration, "asim_compiled": asim_compiled,
                   "run_id": run_id}
        with open(file_loc, mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls, file_loc):
        if not os.path.exists(file_loc):
            logger.info("Creating new stage info at {}".format(file_loc))
            return [None, None, 0, None, None, False, None]
        with open(file_loc, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            data = data if data is not None else {}
            year = data.get('year', None)
            stage_str = data.get('stage', 'null')
        stage = None if stage_str is None or stage_str == 'null' else WorkflowState.Stage[stage_str]
        path = data.get('path', None)
        folder_name = data.get('folder_name', None)
        iteration = data.get('iteration', 0) or 0
        asim_compiled = data.get('asim_compiled', False)
        run_id = data.get('run_id', None)
        return [year, stage, iteration, path, folder_name, asim_compiled, run_id]

    @classmethod
    def from_settings(cls, settings):
        start_year = settings['start_year']
        end_year = settings['end_year']
        travel_model_freq = settings.get('travel_model_freq', 1)
        land_use_enabled = settings['land_use_enabled']
        vehicle_ownership_model_enabled = settings['vehicle_ownership_model_enabled']
        activity_demand_enabled = settings['activity_demand_enabled']
        traffic_assignment_enabled = settings['traffic_assignment_enabled']
        replanning_enabled = settings['replanning_enabled']
        file_loc = settings['state_file_loc']
        output_directory = settings.get('output_directory', None)
        copy_files = output_directory is not None

        [year, stage, iteration, path, folder_name, asim_compiled, run_id] = cls.read_current_stage(file_loc)

        if year:
            logger.info("Found unfinished run: year=%s, stage=%s, filename=%s, run_id=%s)", year, stage, file_loc, run_id)
            current_run_id = run_id
        else:
            current_run_id = None

        year = year or start_year

        out = cls(start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                  activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage,
                  iteration, None, path, folder_name, file_loc, asim_compiled, run_id=current_run_id)

        out._settings['supply_demand_iters'] = settings.get('supply_demand_iters', 1)

        # Handle output directory logic
        if output_directory is not None and copy_files:
            # Create output directory and copy input data to mutable locations
            if ((path is None) or (folder_name is None)):
                out._create_output_dir(settings)
                out.write_state()
            else:
                # Use existing output directory structure
                out.output_path = path
                out.folder_name = folder_name
                # Update run info file if run_id was missing
                if out.run_id and not os.path.exists(os.path.join(path, folder_name, "run_info.json")):
                    out._write_run_id_file()
        elif output_directory is not None and not copy_files:
            # Set up minimal directory structure without copying files
            out.output_path = os.path.expandvars(output_directory)
            out.folder_name = "pilates"
            if out.run_id is None:
                out.run_id = str(uuid.uuid4())
                out.write_state()
                out._write_run_id_file()
        else:
            # output_directory is None - work in place with local inputs
            out.output_path = None
            out.folder_name = None
            if out.run_id is None:
                out.run_id = str(uuid.uuid4())
                out.write_state()

        if year:
            out.forecast_year = min(year + (out.initial_step or travel_model_freq),
                                    end_year) if land_use_enabled else start_year

        # Initialize state if starting fresh
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
            self.run_id
        )

    def is_enabled(self, stage: Stage) -> bool:
        """Checks if a stage is enabled in settings."""
        return stage in self.enabled_stages

    def should_run(self, target_major: Stage, target_inner_iter: int = 0,
                   target_sub_stage: Optional[Stage] = None) -> bool:
        """
        Simplified logic: check if we should run the target stage/iteration/substage
        """
        logger.debug(f"should_run check: Current=(Year={self.current_year}, Major={self.current_major_stage.name if self.current_major_stage else None}, Iter={self.current_inner_iter}, Sub={self.current_sub_stage.name if self.current_sub_stage else None}), Target=(Major={target_major.name}, Iter={target_inner_iter}, Sub={target_sub_stage.name if target_sub_stage else None})")

        # Check if the target stage is enabled
        check_stage = target_sub_stage if target_sub_stage is not None else target_major
        if not self.is_enabled(check_stage):
            logger.debug(f"should_run: Target stage {check_stage.name} is disabled.")
            return False

        # If we haven't started yet, allow any enabled stage
        if self.current_major_stage is None:
            logger.debug("should_run: No current stage, allowing run.")
            return True

        # For major stages (not substages)
        if target_sub_stage is None:
            if target_major == self.Stage.supply_demand_loop:
                # Special handling for the supply-demand loop
                if self.current_major_stage == self.Stage.supply_demand_loop:
                    # We're in the loop - check if we have more iterations
                    total_iters = self._settings.get('supply_demand_iters', 1)
                    result = self.current_inner_iter < total_iters
                    logger.debug(f"should_run: In loop, current iter {self.current_inner_iter} < total {total_iters}? {result}")
                    return result
                else:
                    # Check if we should enter the loop
                    try:
                        current_idx = self.major_stage_order.index(self.current_major_stage)
                        target_idx = self.major_stage_order.index(target_major)
                        result = current_idx <= target_idx
                        logger.debug(f"should_run: Current major idx {current_idx} <= target idx {target_idx}? {result}")
                        return result
                    except ValueError:
                        logger.warning(f"Stage not found in order, allowing run")
                        return True
            else:
                # Regular major stage
                try:
                    current_idx = self.major_stage_order.index(self.current_major_stage)
                    target_idx = self.major_stage_order.index(target_major)
                    result = current_idx <= target_idx
                    logger.debug(f"should_run: Current major idx {current_idx} <= target idx {target_idx}? {result}")
                    return result
                except ValueError:
                    logger.warning(f"Stage not found in order, allowing run")
                    return True

        # For substages within the supply-demand loop
        if target_sub_stage is not None:
            # Must be in the supply-demand loop to run substages
            if self.current_major_stage != self.Stage.supply_demand_loop:
                logger.debug(f"should_run: Not in loop, can't run substage {target_sub_stage.name}")
                return False

            # Check iteration first
            total_iters = self._settings.get('supply_demand_iters', 1)
            if self.current_inner_iter >= total_iters:
                logger.debug(f"should_run: Past max iterations ({self.current_inner_iter} >= {total_iters})")
                return False

            if self.current_inner_iter != target_inner_iter:
                logger.debug(f"should_run: Wrong iteration ({self.current_inner_iter} != {target_inner_iter})")
                return False

            # Check substage position
            if self.current_sub_stage is None:
                # At start of iteration, can run any substage
                logger.debug("should_run: At start of iteration, allowing substage run")
                return True
            else:
                try:
                    current_sub_idx = self.loop_substages.index(self.current_sub_stage)
                    target_sub_idx = self.loop_substages.index(target_sub_stage)
                    result = current_sub_idx <= target_sub_idx
                    logger.debug(f"should_run: Current sub idx {current_sub_idx} <= target sub idx {target_sub_idx}? {result}")
                    return result
                except ValueError:
                    logger.error(f"Substage not found in sequence")
                    return False

        logger.debug("should_run: No case matched, denying run")
        return False

    def is_start_year(self):
        return self.year == self.start_year

    def __iter__(self):
        """Iterator yields the current year as long as the workflow hasn't finished."""
        return self

    def __next__(self):
        """Returns the current year if the workflow is not finished, otherwise raises StopIteration."""
        if self.current_year is None or (self.current_year > self.end_year and self.current_major_stage is None):
            if os.path.exists(self.file_loc):
                if self.current_year is not None and self.current_year > self.settings['end_year'] and self.current_major_stage is None:
                     logger.info(f"Workflow finished. Removing state file: {self.file_loc}")
                     os.remove(self.file_loc)

            if self.current_year is not None and self.current_year <= self.settings['end_year']:
                 logger.warning(f"Workflow state is None, None but current_year ({self.current_year}) is <= end_year ({self.settings['end_year']}). Stopping.")

            raise StopIteration

        return self.current_year

    def complete_step(self, completed_major: Stage, completed_inner_iter: int = 0, completed_sub: Optional[Stage] = None):
        """
        Simplified state transition logic
        """
        logger.info("Completed step: (Major: %s, Iter: %s, Sub: %s) for year %d",
                    completed_major.name if completed_major else None,
                    completed_inner_iter,
                    completed_sub.name if completed_sub else None,
                    self.current_year)

        # Handle substage completion within supply-demand loop
        if completed_sub is not None:
            try:
                current_sub_idx = self.loop_substages.index(completed_sub)
                if current_sub_idx < len(self.loop_substages) - 1:
                    # Move to next substage in same iteration
                    self.current_sub_stage = self.loop_substages[current_sub_idx + 1]
                    logger.debug(f"Moving to next substage: {self.current_sub_stage.name}")
                else:
                    # Last substage of iteration completed
                    total_iters = self._settings.get('supply_demand_iters', 1)
                    if completed_inner_iter < total_iters - 1:
                        # Start next iteration
                        self.current_inner_iter = completed_inner_iter + 1
                        self.current_sub_stage = self.loop_substages[0] if self.loop_substages else None
                        logger.debug(f"Starting next iteration: {self.current_inner_iter}")
                    else:
                        # Loop completed, move to next major stage
                        self._advance_to_next_major_stage()
            except ValueError:
                logger.error(f"Completed substage {completed_sub.name} not found in sequence")
                return

        # Handle major stage completion
        elif completed_major is not None:
            if completed_major == self.Stage.supply_demand_loop:
                # Loop block completed
                self._advance_to_next_major_stage()
            else:
                # Regular major stage completed
                self._advance_to_next_major_stage()

        self.write_state()

    def _advance_to_next_major_stage(self):
        """Move to the next enabled major stage or next year"""
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
                self._advance_to_next_year()
                
        except ValueError:
            logger.error(f"Current major stage {self.current_major_stage.name} not found in order")
            self._advance_to_next_year()

    def _advance_to_next_year(self):
        """Move to the next year and reset to first stage"""
        self.current_year += 1
        
        if self.current_year <= self.end_year:
            logger.info(f"Starting year {self.current_year}")
            self._initialize_first_stage()
        else:
            logger.info(f"Workflow complete at end year {self.end_year}")
            self.current_major_stage = None
            self.current_inner_iter = 0
            self.current_sub_stage = None

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
            logger.error("Start sub-stage %s not found in sequence. Restarting from beginning.", start_sub_stage.name)
            return self.loop_substages
