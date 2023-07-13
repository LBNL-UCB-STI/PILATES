from enum import Enum
import os
import yaml
import logging

logger = logging.getLogger(__name__)

class WorkflowState:
    Stage = Enum('WorkflowStage', ['land_use', 'vehicle_ownership_model', 'activity_demand', 'initialize_asim_for_replanning', 'activity_demand_directly_from_land_use', 'traffic_assignment', 'traffic_assignment_replan'])

    def __init__(self, start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                 activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage):
        self.iteration_started = False
        self.start_year = start_year
        self.end_year = end_year
        self.travel_model_freq = travel_model_freq
        self.year = year
        self.stage = stage
        self.forecast_year = None
        self.enabled_stages = set([])
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
        [year, stage] = cls.read_current_stage()
        year = year or start_year
        return WorkflowState(start_year, end_year, travel_model_freq, land_use_enabled, vehicle_ownership_model_enabled,
                             activity_demand_enabled, traffic_assignment_enabled, replanning_enabled, year, stage)

    @classmethod
    def write_stage(cls, year: int, current_stage: Stage):
        to_save = {"year": year, "stage": current_stage.name if current_stage else None}
        with open('current_stage.yaml', mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls):
        if not os.path.exists('current_stage.yaml'):
            return [None, None]
        with open('current_stage.yaml', encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            data = data if data is not None else {}
            year = data.get('year', None)
            stage_str = data.get('stage', 'null')
            stage = None if stage_str == 'null' else WorkflowState.Stage[stage_str]
            return [year, stage]

    def enabled(self, stage) -> bool:
        return stage in self.enabled_stages

    def should_continue(self) -> bool:
        self.year = self.year + self.travel_model_freq if self.iteration_started else self.year
        logger.info("Started year %d", self.year)
        self.iteration_started = True
        self.forecast_year = \
            min(self.year + self.travel_model_freq, self.end_year) if self.enabled(WorkflowState.Stage.land_use)\
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
            WorkflowState.write_stage(year, next_stage)
        else:
            os.remove('current_stage.yaml')

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