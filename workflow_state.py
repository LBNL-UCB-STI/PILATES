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
                 activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage, output_path,
                 folder_name, file_loc):
        self.iteration_started = False
        self.start_year = start_year
        self.end_year = end_year
        self.travel_model_freq = travel_model_freq
        self.year = year
        self.stage = stage
        self.forecast_year = None
        self.enabled_stages = set([])
        self.folder_name = folder_name
        self.output_path = output_path
        self.file_loc = file_loc
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

    def _create_output_dir(self, settings: dict):
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_loc = os.path.expandvars(settings['output_directory'])
        run_name = settings['output_run_name']
        folder_name = "{0}-{1}-{2}".format(settings['region'], run_name, dt)
        folder_path = os.path.join(base_loc, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for model in ['land_use_model', 'travel_model', 'activity_demand_model', 'vehicle_ownership_model']:
            if settings.get(model) is not None:
                model_name = settings[model]
                os.makedirs(os.path.join(folder_path, model_name))
                if model_name == "urbansim":
                    output_dir = os.path.join(folder_path, settings['usim_local_mutable_data_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                    usim_pre.copy_data_to_mutable_location(settings, output_dir)
                elif model_name == "beam":
                    input_dir = os.path.join(folder_path, settings['beam_local_mutable_data_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    beam_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(folder_path, settings['beam_local_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                elif model_name == "atlas":
                    input_dir = os.path.join(folder_path, settings['atlas_host_mutable_data_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    atlas_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(folder_path, settings['atlas_host_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)
                elif model_name == "activitysim":
                    input_dir = os.path.join(folder_path, settings['asim_local_mutable_data_folder'])
                    os.makedirs(input_dir, exist_ok=True)
                    asim_pre.copy_data_to_mutable_location(settings, input_dir)
                    output_dir = os.path.join(folder_path, settings['asim_local_output_folder'])
                    os.makedirs(output_dir, exist_ok=True)

        self.output_path = base_loc
        self.folder_name = folder_name

        print('STOP')

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
        [year, stage, path, folder_name] = cls.read_current_stage(file_loc)
        if year:
            logger.info("Found unfinished run: year=%s, stage=%s, filename=%s)", year, stage, file_loc)
        year = year or start_year
        out = WorkflowState(start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                            activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage, path,
                            folder_name, file_loc)
        if (path is None) | (folder_name is None):
            out._create_output_dir(settings)
        return out

    @classmethod
    def write_stage(cls, year: int, current_stage: Stage, file_loc, path, folder_name):
        to_save = {"year": year, "stage": current_stage.name if current_stage else None, "path": path,
                   "folder_name": folder_name}
        with open(file_loc, mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls, file_loc):
        if not os.path.exists(file_loc):
            logger.info("Creating new stage info at {}".format(file_loc))
            return [None, None, None, None]
        with open(file_loc, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            data = data if data is not None else {}
            year = data.get('year', None)
            stage_str = data.get('stage', 'null')
            stage = None if stage_str == 'null' else WorkflowState.Stage[stage_str]
            path = data.get('path', None)
            folder_name = data.get('folder_name', None)
            return [year, stage, path, folder_name]

    def enabled(self, stage) -> bool:
        return stage in self.enabled_stages

    def should_continue(self) -> bool:
        next_year = self.year + self.travel_model_freq if self.iteration_started else self.year
        if next_year >= self.end_year:
            return False
        self.year = next_year
        logger.info("Started year %d", self.year)
        self.iteration_started = True
        self.forecast_year = \
            min(self.year + self.travel_model_freq, self.end_year) if self.enabled(WorkflowState.Stage.land_use) \
                else self.start_year
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
            WorkflowState.write_stage(year, next_stage, self.file_loc, self.output_path, self.folder_name)
        else:
            os.remove(self.file_loc)

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
