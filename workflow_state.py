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
                 ['land_use_pre', 'land_use', 'land_use_post',
                  'vehicle_ownership_model_pre', 'vehicle_ownership_model', 'vehicle_ownership_model_post',
                  'activity_demand_pre', 'activity_demand', 'activity_demand_post',
                  'initialize_asim_for_replanning_pre', 'initialize_asim_for_replanning', 'initialize_asim_for_replanning_post',
                  'activity_demand_directly_from_land_use_pre', 'activity_demand_directly_from_land_use', 'activity_demand_directly_from_land_use_post',
                  'traffic_assignment_pre', 'traffic_assignment', 'traffic_assignment_post',
                  'traffic_assignment_replan_pre', 'traffic_assignment_replan', 'traffic_assignment_replan_post'])

    def __init__(self, start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                 activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage, iteration,
                 output_path, folder_name, file_loc, asim_compiled=False):
        self.iteration_started = False
        self.start_year = start_year
        self.end_year = end_year
        self.travel_model_freq = travel_model_freq
        self.year = year
        self.stage = stage
        self.iteration = iteration
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
            self.enabled_stages.add(WorkflowState.Stage.land_use_pre)
            self.enabled_stages.add(WorkflowState.Stage.land_use)
            self.enabled_stages.add(WorkflowState.Stage.land_use_post)
        if vehicle_ownership_model_enabled:
            self.enabled_stages.add(WorkflowState.Stage.vehicle_ownership_model_pre)
            self.enabled_stages.add(WorkflowState.Stage.vehicle_ownership_model)
            self.enabled_stages.add(WorkflowState.Stage.vehicle_ownership_model_post)
        if activity_demand_enabled:
            self.enabled_stages.add(WorkflowState.Stage.activity_demand_pre)
            self.enabled_stages.add(WorkflowState.Stage.activity_demand)
            self.enabled_stages.add(WorkflowState.Stage.activity_demand_post)
            if replanning_enabled:
                self.enabled_stages.add(WorkflowState.Stage.initialize_asim_for_replanning_pre)
                self.enabled_stages.add(WorkflowState.Stage.initialize_asim_for_replanning)
                self.enabled_stages.add(WorkflowState.Stage.initialize_asim_for_replanning_post)
        else:
            self.enabled_stages.add(WorkflowState.Stage.activity_demand_directly_from_land_use_pre)
            self.enabled_stages.add(WorkflowState.Stage.activity_demand_directly_from_land_use)
            self.enabled_stages.add(WorkflowState.Stage.activity_demand_directly_from_land_use_post)
        if traffic_assignment_enabled:
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment_pre)
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment)
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment_post)
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment_replan_pre)
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment_replan)
            self.enabled_stages.add(WorkflowState.Stage.traffic_assignment_replan_post)

    @property
    def full_path(self):
        return os.path.join(self.output_path, self.folder_name)

    @property
    def asim_compiled(self):
        if self.__asim_compiled:
            logger.info("ActivitySim already compiled in year %s", self.year)
        else:
            logger.info("ActivitySim not compiled in year %s, so running compilation (this will take longer)", self.year)
        return self.__asim_compiled

    def compile_asim(self):
        self.__asim_compiled = True
        logger.info("Completed compiling activitysim in year %s", self.year)
        WorkflowState.write_stage(self.year, self.stage, self.file_loc, self.output_path, self.folder_name, self.iteration)

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
    def write_stage(cls, year: int, current_stage: Stage, file_loc, path, folder_name, iteration):
        to_save = {"year": year, "stage": current_stage.name if current_stage else None, "path": path,
                   "folder_name": folder_name, "iteration": iteration, "asim_compiled": False}
        with open(file_loc, mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls, file_loc):
        if not os.path.exists(file_loc):
            logger.info("Creating new stage info at {}".format(file_loc))
            return [None, None, None, None, None, False]
        with open(file_loc, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            data = data if data is not None else {}
            year = data.get('year', None)
            stage_str = data.get('stage', 'null')
            stage = None if stage_str == 'null' else WorkflowState.Stage[stage_str]
            path = data.get('path', None)
            folder_name = data.get('folder_name', None)
            iteration = data.get('iteration', 0)
            asim_compiled = data.get('asim_compiled', False)
            return [year, stage, iteration, path, folder_name, asim_compiled]

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
        [year, stage, iteration, path, folder_name, asim_compiled] = cls.read_current_stage(file_loc)
        if year:
            logger.info("Found unfinished run: year=%s, stage=%s, filename=%s)", year, stage, file_loc)
        year = year or start_year
        out = WorkflowState(start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                            activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage,
                            iteration, path, folder_name, file_loc, asim_compiled)
        if (path is None) | (folder_name is None):
            out._create_output_dir(settings)
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

    def get_pre_stage(self, stage: Stage) -> Stage:
        """Get the pre-processing stage for a given stage"""
        stage_name = stage.name
        if stage_name.endswith('_post'):
            return None
        elif stage_name.endswith('_pre'):
            return stage
        else:
            return WorkflowState.Stage[stage_name + '_pre']

    def get_post_stage(self, stage: Stage) -> Stage:
        """Get the post-processing stage for a given stage"""
        stage_name = stage.name
        if stage_name.endswith('_pre'):
            return WorkflowState.Stage[stage_name[:-4] + '_post']
        elif stage_name.endswith('_post'):
            return None
        else:
            return WorkflowState.Stage[stage_name + '_post']

    def get_main_stage(self, stage: Stage) -> Stage:
        """Get the main stage (without pre/post) for a given stage"""
        stage_name = stage.name
        if stage_name.endswith('_pre') or stage_name.endswith('_post'):
            return WorkflowState.Stage[stage_name[:-4]]
        return stage

    def complete(self, stage):
        """Complete a stage and move to the next appropriate stage"""
        logger.info("Completed %s of %d", stage, self.year)
        
        # If this is a pre stage, move to main stage
        if stage.name.endswith('_pre'):
            self.stage = self.get_main_stage(stage)
            WorkflowState.write_stage(self.year, self.stage, self.file_loc, self.output_path, self.folder_name, self.iteration)
            return
            
        # If this is a main stage, move to post stage if it exists
        if not stage.name.endswith('_post'):
            post_stage = self.get_post_stage(stage)
            if post_stage in self.enabled_stages:
                self.stage = post_stage
                WorkflowState.write_stage(self.year, self.stage, self.file_loc, self.output_path, self.folder_name, self.iteration)
                return
        
        # If this is a post stage or there is no post stage, move to next main stage
        self.stage = None
        [year, next_stage] = self.next_stage(self.year, stage)
        if year:
            # If there's a next stage, start with its pre stage if it exists
            if next_stage:
                pre_stage = self.get_pre_stage(next_stage)
                if pre_stage in self.enabled_stages:
                    next_stage = pre_stage
            WorkflowState.write_stage(year, next_stage, self.file_loc, self.output_path, self.folder_name, 0)
        else:
            os.remove(self.file_loc)

    def complete_iteration(self, iteration):
        logger.info("Completed iteration %d of stage %s of %d", iteration, self.stage, self.year)
        self.iteration += 1
        WorkflowState.write_stage(self.year, self.stage, self.file_loc, self.output_path, self.folder_name,
                                  self.iteration)

    def next_stage(self, year: int, stage: Stage):
        """Get the next stage to run, skipping pre/post stages if they're not enabled"""
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
