from datetime import datetime
from enum import Enum
import os
import yaml
import logging
from pilates.activitysim import preprocessor as asim_pre
from pilates.urbansim import preprocessor as usim_pre
from pilates.beam import preprocessor as beam_pre
from pilates.atlas import preprocessor as atlas_pre  ##

logger = logging.getLogger(__name__)


class WorkflowState:
    Stage = Enum('WorkflowStage',
                 ['land_use', 'vehicle_ownership_model', 'activity_demand', 'initialize_asim_for_replanning',
                  'activity_demand_directly_from_land_use', 'traffic_assignment', 'traffic_assignment_replan'])

    def __init__(self, start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                 activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage, iteration, sub_iteration,
                 output_path, folder_name, file_loc, asim_compiled=False):
        self.iteration_started = False
        self.start_year = start_year
        self.end_year = end_year
        self.travel_model_freq = travel_model_freq
        self.year = year
        self.stage = stage
        self.iteration = iteration
        self.sub_iteration = sub_iteration # Added sub_iteration tracking
        self.forecast_year = None
        self.enabled_stages = set([])
        self.folder_name = folder_name
        self.output_path = output_path
        self.file_loc = file_loc
        self.__asim_compiled = asim_compiled
        if year == 2010:
            self.initial_step = 7
        else:
            self.initial_step = None
        if land_use_enabled:
            self.enabled_stages.add(WorkflowState.Stage.land_use)
        if vehicle_ownership_model_enabled:
            self.enabled_stages.add(WorkflowState.Stage.vehicle_ownership_model)
        if activity_demand_enabled:
            self.enabled_stages.add(WorkflowState.Stage.activity_demand)
            if replanning_enabled:
                self.enabled_stages.add(WorkflowState.Stage.initialize_asim_for_replanning)
        else:
            self.enabled_stages.add(WorkflowState.Stage.activity_demand_directly_from_land_use)
        if traffic_assignment_enabled:
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment)
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment_replan)

    @property
    def full_path(self):
        return os.path.join(self.output_path, self.folder_name)

    @property
    def asim_compiled(self):
        if self.__asim_compiled:
            logger.info("ActivitySim already compiled in year %s", self.year)
        else:
            logger.info("ActivitySim not compiled in year %s, so running compilation (this will take longer)",
                        self.year)
        return self.__asim_compiled

    def compile_asim(self):
        self.__asim_compiled = True
        logger.info("Completed compiling activitysim in year %s", self.year)
        WorkflowState.write_stage(self.year, self.stage, self.file_loc, self.output_path, self.folder_name,
                                  self.iteration, self.__asim_compiled)

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
        vehicle_ownership_model_enabled = settings['vehicle_ownership_model_enabled']  # Atlas
        activity_demand_enabled = settings['activity_demand_enabled']
        traffic_assignment_enabled = settings['traffic_assignment_enabled']
        replanning_enabled = settings['replanning_enabled']
        file_loc = settings['state_file_loc']
        copy_files = settings.get("copy_files", True)
        [year, stage, iteration, path, folder_name, sub_iteration, asim_compiled] = cls.read_current_stage(file_loc) # Added sub_iteration
        if year:
            logger.info("Found unfinished run: year=%s, stage=%s, filename=%s)", year, stage, file_loc)
        year = year or start_year
        out = WorkflowState(start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                            activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage,
                            iteration, sub_iteration, path, folder_name, file_loc, asim_compiled) # Added sub_iteration
        if ((path is None) | (folder_name is None)) & copy_files:
            out._create_output_dir(settings)
        if not copy_files:
            out.output_path = ""
            out.folder_name = "pilates"
        if year:
            out.forecast_year = min(year + (out.initial_step or travel_model_freq),
                                    end_year) if land_use_enabled else start_year
        return out

    def enabled(self, stage) -> bool:
        return stage in self.enabled_stages

    def should_continue(self) -> bool:
        if self.initial_step is not None:
            step = self.initial_step
            self.initial_step = None
        else:
            step = None
        next_year = self.year + (step or self.travel_model_freq) if self.iteration_started else self.year
        if next_year >= self.end_year:
            return False
        self.year = next_year
        logger.info("Started year %d", self.year)
        self.iteration_started = True
        self.forecast_year = \
            min(self.year + (step or self.travel_model_freq), self.end_year) if self.enabled(
                WorkflowState.Stage.land_use) else self.start_year
        return self.year < self.end_year

    def is_start_year(self):
        return self.year == self.start_year

    def should_do(self, stage: Stage) -> bool:
        return stage in self.enabled_stages and self.on_or_after_current_stage(stage)

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
        return self

    def __next__(self):
        if self.should_continue():
            return self.year
        else:
            raise StopIteration
