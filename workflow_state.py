from datetime import datetime
from enum import Enum
from typing import Optional
import os
import yaml
import logging
from pilates.activitysim import preprocessor as asim_pre
from pilates.urbansim import preprocessor as usim_pre
from pilates.beam import preprocessor as beam_pre
from pilates.atlas import preprocessor as atlas_pre

logger = logging.getLogger(__name__)

class WorkflowState:
    # Define all possible stages, including major blocks and individual model/task steps
    Stage = Enum('WorkflowStage', [
        'initialize_data',
        'land_use',
        'supply_demand_loop', # Represents the entire inner feedback loop block
        'vehicle_ownership_model', # Atlas
        'activity_demand',
        'initialize_asim_for_replanning',
        'activity_demand_directly_from_land_use', # Alternative to activity_demand if no ABM
        'traffic_assignment',
        'postprocessing',
    ])

    # Define the sequence of major stages in the workflow
    MAJOR_STAGE_SEQUENCE = [Stage.initialize_data, Stage.land_use, Stage.vehicle_ownership_model,
                            Stage.supply_demand_loop, Stage.postprocessing]

    # Define the sequence of stages *within* the supply-demand inner loop
    SUB_STAGE_SEQUENCE = [] # This list will be populated in __init__ based on enabled models

    def __init__(self, start_year: int, end_year: int, travel_model_freq: int, land_use_enabled: bool,
                 vehicle_ownership_model_enabled: bool, activity_demand_enabled: bool,
                 traffic_assignment_enabled: bool, replanning_enabled: bool,
                 year: int | None, major_stage: Stage | None, inner_iter: int, sub_stage: Stage | None,
                 output_path: str | None, folder_name: str | None, file_loc: str, asim_compiled: bool):

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

        # Determine enabled individual stages based on settings
        self.enabled_individual_stages = set()

        if land_use_enabled:
            self.enabled_individual_stages.add(self.Stage.land_use)
        if vehicle_ownership_model_enabled:
            self.enabled_individual_stages.add(self.Stage.vehicle_ownership_model)
        if activity_demand_enabled:
            self.enabled_individual_stages.add(self.Stage.activity_demand)
            if replanning_enabled:
                self.enabled_individual_stages.add(self.Stage.initialize_asim_for_replanning)
        else:
            # If no ABM, direct activity from land use
            self.enabled_individual_stages.add(self.Stage.activity_demand_directly_from_land_use)

        if traffic_assignment_enabled:
            self.enabled_individual_stages.add(self.Stage.traffic_assignment)

        # Define SUB_STAGE_SEQUENCE based on enabled individual stages
        # Vehicle ownership runs outside the loop, so exclude it from sub-stages
        self.SUB_STAGE_SEQUENCE = [
            s for s in [self.Stage.activity_demand,
                        self.Stage.initialize_asim_for_replanning, self.Stage.activity_demand_directly_from_land_use,
                        self.Stage.traffic_assignment]
            if s in self.enabled_individual_stages
        ]

        # Add major stage 'supply_demand_loop' if there are any enabled sub-stages
        if self.SUB_STAGE_SEQUENCE:
            self.enabled_individual_stages.add(self.Stage.supply_demand_loop)

        # Always add initialize_data stage
        self.enabled_individual_stages.add(self.Stage.initialize_data)

    @property
    def settings(self):
        return self._settings if hasattr(self, '_settings') else {'end_year': self.end_year}

    @property
    def full_path(self):
        return os.path.join(self.output_path, self.folder_name)

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
        self.write_state() # Save state after compilation

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
        return stage in self.enabled_individual_stages

    def _create_output_dir(self, settings: dict):
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_loc = os.path.expandvars(settings['output_directory'])
        run_name = settings['output_run_name']
        folder_name = "{0}-{1}-{2}".format(settings['region'], run_name, dt)
        folder_path = os.path.join(base_loc, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        have_not_copied_usim_data = True

        for model in ['travel_model', 'activity_demand_model', 'vehicle_ownership_model', 'land_use_model']:
            if settings.get(model) is not None:
                model_name = settings[model]
                os.makedirs(os.path.join(folder_path, model_name), exist_ok=True)
                if (model_name == "urbansim") | ((model_name == "activitysim") & have_not_copied_usim_data):
                    output_dir = os.path.join(folder_path, settings['usim_local_mutable_data_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                    usim_pre.copy_data_to_mutable_location(settings, output_dir)
                    have_not_copied_usim_data = False
                if model_name == "beam":
                    input_dir = os.path.join(folder_path, settings['beam_local_mutable_data_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    beam_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(folder_path, settings['beam_local_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                if model_name == "atlas":
                    input_dir = os.path.join(folder_path, settings['atlas_host_mutable_input_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    atlas_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(folder_path, settings['atlas_host_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                if model_name == "activitysim":
                    asim_pre.copy_data_to_mutable_location(settings, folder_path)
                    output_dir = os.path.join(folder_path, settings['asim_local_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)

        self.output_path = base_loc
        self.folder_name = folder_name

        print('STOP')

    @classmethod
    def write_stage(cls, year: int, current_stage: Stage, file_loc, path, folder_name, iteration, sub_iteration, asim_compiled):
        to_save = {"year": year, "stage": current_stage.name if current_stage else None, "path": path,
                   "folder_name": folder_name, "iteration": iteration, "sub_iteration": sub_iteration, "asim_compiled": asim_compiled} # Added sub_iteration to save
        with open(file_loc, mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls, file_loc):
        if not os.path.exists(file_loc):
            logger.info("Creating new stage info at {}".format(file_loc))
            return [None, None, 0, None, None, 0, False] # Added default sub_iteration
        with open(file_loc, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            data = data if data is not None else {}
            year = data.get('year', None)
            stage_str = data.get('stage', 'null')
        stage = None if stage_str == 'null' else WorkflowState.Stage[stage_str]
        path = data.get('path', None)
        folder_name = data.get('folder_name', None)
        iteration = data.get('iteration', 0) or 0
        sub_iteration = data.get('sub_iteration', 0) or 0 # Added reading sub_iteration
        asim_compiled = data.get('asim_compiled', False)
        return [year, stage, iteration, path, folder_name, sub_iteration, asim_compiled] # Added sub_iteration to return

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
        copy_files = settings.get("copy_files", True)

        [year, stage, iteration, path, folder_name, sub_iteration, asim_compiled] = cls.read_current_stage(file_loc)

        if year:
            logger.info("Found unfinished run: year=%s, stage=%s, filename=%s)", year, stage, file_loc)

        year = year or start_year

        out = cls(start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                  activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage,
                  iteration, sub_iteration, path, folder_name, file_loc, asim_compiled)
        out._settings['supply_demand_iters'] = settings.get('supply_demand_iters', 1)

        if ((path is None) or (folder_name is None)) and copy_files:
            out._create_output_dir(settings)
        if not copy_files:
            out.output_path = ""
            out.folder_name = "pilates"

        if year:
            out.forecast_year = min(year + (out.initial_step or travel_model_freq),
                                    end_year) if land_use_enabled else start_year

        return out

    def write_state(self):
        """Save the current state to file"""
        WorkflowState.write_stage(
            self.current_year,
            self.current_sub_stage or self.current_major_stage,
            self.file_loc,
            self.output_path,
            self.folder_name,
            self.current_inner_iter,
            0,  # sub_iteration placeholder
            self.__asim_compiled
        )

    def is_enabled(self, stage: Stage) -> bool:
        """Checks if an individual stage (model/task) is enabled in settings."""
        return stage in self.enabled_individual_stages

    def should_run(self, target_major: Stage, target_inner_iter: int = 0,
                   target_sub_stage: Optional[Stage] = None) -> bool:
        """Checks if the target step is at or after the current state and if the target major stage is enabled."""
        # Check if stage is enabled
        check_stage = target_sub_stage if target_sub_stage is not None else target_major
        if not self.is_enabled(check_stage):
            return False

        # For major stages outside the supply-demand loop (land_use, vehicle_ownership_model)
        if target_sub_stage is None and target_major != self.Stage.supply_demand_loop:
            # These run once per year, just check if we're at or before this stage
            if self.current_major_stage is None:
                return True  # Haven't started any stages yet

            # Find positions in major sequence
            try:
                current_major_idx = self.MAJOR_STAGE_SEQUENCE.index(
                    self.current_major_stage) if self.current_major_stage in self.MAJOR_STAGE_SEQUENCE else -1
                if target_major == self.Stage.vehicle_ownership_model:
                    # Vehicle ownership runs after land_use but before supply_demand_loop
                    if self.current_major_stage is None:
                        return True  # Haven't started yet
                    elif self.current_major_stage == self.Stage.land_use:
                        return True  # Just finished land use
                    elif self.current_major_stage == self.Stage.vehicle_ownership_model:
                        return False  # Already completed
                    elif self.current_major_stage == self.Stage.supply_demand_loop:
                        return False  # Already past this stage
                    else:
                        # Check if land use is enabled - if not, vehicle ownership can run first
                        if not self._settings.get('land_use_enabled', False):
                            return True  # Land use disabled, can run vehicle ownership
                        else:
                            return True  # Before this stage
                else:
                    # For other major stages, use normal sequence logic
                    target_major_idx = self.MAJOR_STAGE_SEQUENCE.index(
                        target_major) if target_major in self.MAJOR_STAGE_SEQUENCE else -1
                    return current_major_idx <= target_major_idx
            except ValueError:
                return True  # If stage not in sequence, assume we should run it

        # For the supply-demand loop block
        if target_major == self.Stage.supply_demand_loop and target_sub_stage is None:
            # Check if we should start the entire supply-demand loop
            if self.current_major_stage is None or self.current_major_stage == self.Stage.land_use or self.current_major_stage == self.Stage.vehicle_ownership_model:
                return True
            return self.current_major_stage == self.Stage.supply_demand_loop

        # For specific sub-stages within the supply-demand loop
        if target_sub_stage is not None:
            # Must be in the supply-demand loop to run sub-stages
            if self.current_major_stage != self.Stage.supply_demand_loop:
                return True  # Haven't started loop yet

            # Compare iteration first
            if self.current_inner_iter < target_inner_iter:
                return False  # Not at target iteration yet
            elif self.current_inner_iter > target_inner_iter:
                return False  # Past target iteration

            # Same iteration - compare sub-stage position
            if self.current_sub_stage is None:
                return True  # At start of iteration

            try:
                current_sub_idx = self.SUB_STAGE_SEQUENCE.index(self.current_sub_stage)
                target_sub_idx = self.SUB_STAGE_SEQUENCE.index(target_sub_stage)
                return current_sub_idx <= target_sub_idx
            except ValueError:
                return True  # Stage not found, assume we should run

        return True

    def is_start_year(self):
        return self.year == self.start_year

    def should_do(self, stage: Stage) -> bool:
        return stage in self.enabled_individual_stages and self.on_or_after_current_stage(stage)

    def on_or_after_current_stage(self, stage):
        if not self.stage:
            return True
        return self.stage.value <= stage.value

    def complete(self, stage):
        logger.info("Completed %s of %d", stage, self.year)
        self.stage = None
        [year, next_stage] = self.next_stage(self.year, stage)
        if year:
            # When moving to the next stage/year, reset iteration and sub_iteration
            WorkflowState.write_stage(year, next_stage, self.file_loc, self.output_path, self.folder_name, 0, 0,
                                      self.__asim_compiled) # Reset iteration and sub_iteration
        else:
            os.remove(self.file_loc)

    def complete_iteration(self, iteration):
        logger.info("Completed iteration %d of stage %s of %d", iteration, self.stage, self.year)
        self.iteration += 1
        # When completing an iteration, reset sub_iteration
        WorkflowState.write_stage(self.year, self.stage, self.file_loc, self.output_path, self.folder_name,
                                  self.iteration, 0, self.__asim_compiled) # Reset sub_iteration

    def next_stage(self, year: int, stage: Stage):
        next_enabled_stage = next(filter(self.enabled, list(WorkflowState.Stage)[stage.value:]), None)
        if not next_enabled_stage:
            year = year + 1
        if year >= self.end_year:
            return [None, None]
        else:
            return [year, next_enabled_stage]

    def __iter__(self):
        """Iterator yields the current year as long as the workflow hasn't finished."""
        return self

    def __next__(self):
        """Returns the current year if the workflow is not finished, otherwise raises StopIteration."""
        # Check if the workflow has finished (current_year is beyond end_year or state indicates completion)
        if self.current_year is None or self.current_year > self.end_year:
            if os.path.exists(self.file_loc):
                os.remove(self.file_loc)
            raise StopIteration

        return self.current_year  # Yield the current year


    def complete_step(self, completed_major: Stage, completed_inner_iter: int = 0, completed_sub: Optional[Stage] = None):
        """
        Marks the completed step and updates the state to the next logical step.
        Handles transitions between sub-stages, inner iterations, major stages, and years.
        """
        logger.info("Completed step: (Major: %s, Iter: %s, Sub: %s) for year %d",
                    completed_major.name if completed_major else None,
                    completed_inner_iter,
                    completed_sub.name if completed_sub else None,
                    self.current_year)

        next_major = completed_major
        next_iter = completed_inner_iter
        next_sub = completed_sub
        year_finished = False

        if completed_sub is not None: # Completed a specific sub-stage within the supply-demand loop
            # Find the index of the completed sub-stage
            try:
                completed_sub_idx = self.SUB_STAGE_SEQUENCE.index(completed_sub)
                # Find the next sub-stage
                if completed_sub_idx < len(self.SUB_STAGE_SEQUENCE) - 1:
                    next_sub = self.SUB_STAGE_SEQUENCE[completed_sub_idx + 1]
                    logger.debug("Next sub-stage: %s", next_sub.name)
                    # Major stage and iter remain the same
                else: # Last sub-stage of this inner iteration completed
                    # Need to check if this iteration is the last one
                    total_inner_iters = self.settings.get('supply_demand_iters', 1)
                    if completed_inner_iter < total_inner_iters - 1:
                         next_sub = self.SUB_STAGE_SEQUENCE[0] # Reset to first sub-stage for the next iteration
                         next_iter = completed_inner_iter + 1
                         logger.debug("Inner iteration %d complete. Starting next iter %d, first sub-stage %s.",
                                      completed_inner_iter, next_iter, next_sub.name)
                    else: # Last inner iteration completed
                         next_iter = 0 # Reset iteration count
                         next_sub = None # Exit sub-stage sequence
                         # Find the next major stage after supply_demand_loop
                         try:
                             current_major_idx = self.MAJOR_STAGE_SEQUENCE.index(self.Stage.supply_demand_loop)
                             if current_major_idx < len(self.MAJOR_STAGE_SEQUENCE) - 1:
                                 next_major = self.MAJOR_STAGE_SEQUENCE[current_major_idx + 1]
                                 logger.debug("Supply-demand loop complete for year %d. Moving to major stage: %s", self.current_year, next_major.name)
                             else: # Last major stage (supply_demand_loop) completed
                                 next_major = None # No more major stages in sequence for this year
                                 year_finished = True
                                 logger.debug("Supply-demand loop was the last major stage for year %d.", self.current_year)
                         except ValueError:
                             logger.error("MAJOR_STAGE_SEQUENCE is missing WorkflowState.Stage.supply_demand_loop")
                             next_major = None # Assume year finished if major sequence is invalid
                             year_finished = True

            except ValueError:
                logger.error("Completed sub-stage %s not found in SUB_STAGE_SEQUENCE. State NOT updated.", completed_sub.name)
                # State remains unchanged, this indicates an error
                return # Do not update state if the completed sub-stage is not recognized

        elif completed_major is not None: # Completed a major stage (outside the inner loop or the inner loop block itself)
            try:
                # Find the index of the completed major stage
                completed_major_idx = self.MAJOR_STAGE_SEQUENCE.index(completed_major)
                # Find the next major stage
                if completed_major_idx < len(self.MAJOR_STAGE_SEQUENCE) - 1:
                    next_major = self.MAJOR_STAGE_SEQUENCE[completed_major_idx + 1]
                    logger.debug("Major stage %s complete. Moving to major stage: %s", completed_major.name, next_major.name)
                    if next_major == self.Stage.supply_demand_loop and self.SUB_STAGE_SEQUENCE:
                        # Transitioning into the inner loop block
                        next_iter = 0
                        next_sub = self.SUB_STAGE_SEQUENCE[0] # Start at first sub-stage
                        logger.debug("Starting supply-demand loop for year %d, iter %d, sub-stage %s", self.current_year, next_iter, next_sub.name)
                    else:
                        next_iter = 0 # Reset iteration counter for non-inner loop stages
                        next_sub = None # Not in sub-stage sequence
                else: # Last major stage completed
                    next_major = None # No more major stages in sequence
                    year_finished = True
                    logger.debug("Major stage %s was the last stage for year %d.", completed_major.name, self.current_year)

            except ValueError:
                logger.error("Completed major stage %s not found in MAJOR_STAGE_SEQUENCE. State NOT updated.", completed_major.name)
                # State remains unchanged, this indicates an error
                return # Do not update state if the completed major stage is not recognized

        else: # Called complete_step with no completed stage?
            logger.warning("complete_step called with no completed stage specified. State NOT updated.")
            return # Do not update state

        # Update the state variables to the *next* step
        self.current_major_stage = next_major
        self.current_inner_iter = next_iter
        self.current_sub_stage = next_sub

        # If the year is finished, advance to the next year and reset state for the start of the new year
        if year_finished:
             self.current_year += 1
             if self.current_year <= self.settings['end_year']:
                 logger.info("Workflow for year %d complete. Starting year %d.", self.current_year - 1, self.current_year)
                 # Determine the first enabled major stage for the next year
                 first_major = next(filter(self.enabled_individual_stages.__contains__, self.MAJOR_STAGE_SEQUENCE), None)
                 if first_major is None:
                      logger.info("No enabled stages for year %d. Workflow finished.", self.current_year)
                      self.current_year = self.settings['end_year'] + 1 # Ensure __next__ stops
                 else:
                     self.current_major_stage = first_major # Start with the first enabled major stage of the new year
                     self.current_inner_iter = 0
                     self.current_sub_stage = None
             else:
                 logger.info("End year %d reached. Workflow complete.", self.settings['end_year'])
                 # Keep state pointing past the end year, so __next__ stops
                 self.current_major_stage = None # Indicate workflow finished
                 self.current_inner_iter = 0
                 self.current_sub_stage = None

        self.write_state() # Save the updated state

    # Helper to get sub-stages starting from a specific one (for resuming loop)
    def get_sub_stages_from(self, start_sub_stage: Optional[Stage]):
        if not self.SUB_STAGE_SEQUENCE:
            return [] # No sub-stages configured
        if start_sub_stage is None:
            return self.SUB_STAGE_SEQUENCE # Start from the beginning
        try:
            start_index = self.SUB_STAGE_SEQUENCE.index(start_sub_stage)
            return self.SUB_STAGE_SEQUENCE[start_index:]
        except ValueError:
            logger.error("Start sub-stage %s not found in SUB_STAGE_SEQUENCE. Restarting sub-stages from beginning.", start_sub_stage.name)
            return self.SUB_STAGE_SEQUENCE # Fallback to start from beginning if state is inconsistent

