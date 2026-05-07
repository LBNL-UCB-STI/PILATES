from bisect import bisect_left
from enum import Enum
import logging
import os
from typing import Dict, Optional, Tuple

import yaml

from pilates.config import PilatesConfig
from pilates.workflows._profile import ensure_runtime_flags_initialized

logger = logging.getLogger(__name__)


class WorkflowState:
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
        impacts_enabled: bool,
        year: int | None,
        major_stage: Stage | None,
        inner_iter: int,
        sub_stage: Stage | None,
        file_loc: Optional[str],
        asim_compiled: bool,
        full_settings: PilatesConfig,
        sub_stage_progress: Optional[str] = None,
        run_info_path: Optional[str] = None,
        data_initialized: bool = False,
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.travel_model_freq = travel_model_freq

        self.current_year = year or start_year
        self.current_major_stage = major_stage
        self.current_inner_iter = inner_iter
        self.current_sub_stage = sub_stage
        self.sub_stage_progress = sub_stage_progress
        self.run_info_path = run_info_path
        self.is_restart_run = bool(run_info_path)
        self.data_initialized = data_initialized

        self.forecast_year: int | None = None
        self.file_loc = file_loc
        self.mirror_file_loc: Optional[str] = None
        self.__asim_compiled = asim_compiled

        self._settings = {
            "supply_demand_iters": 1,
            "land_use_enabled": land_use_enabled,
            "vehicle_ownership_model_enabled": vehicle_ownership_model_enabled,
            "activity_demand_enabled": activity_demand_enabled,
            "traffic_assignment_enabled": traffic_assignment_enabled,
            "impacts_enabled": impacts_enabled,
        }
        self.full_settings = full_settings
        self._year_schedule: Tuple[int, ...] = ()
        self._year_schedule_positions: Dict[int, int] = {}
        self._schedule_index: Optional[int] = None

        self.enabled_stages = set()
        if land_use_enabled:
            self.enabled_stages.add(self.Stage.land_use)
        if vehicle_ownership_model_enabled:
            self.enabled_stages.add(self.Stage.vehicle_ownership_model)
        if activity_demand_enabled:
            self.enabled_stages.add(self.Stage.activity_demand)
        elif land_use_enabled:
            self.enabled_stages.add(self.Stage.activity_demand_directly_from_land_use)
        if traffic_assignment_enabled:
            self.enabled_stages.add(self.Stage.traffic_assignment)
        if impacts_enabled or full_settings.postprocessing is not None:
            self.enabled_stages.add(self.Stage.postprocessing)

        self.major_stage_order = [
            self.Stage.land_use,
            self.Stage.vehicle_ownership_model,
            self.Stage.supply_demand_loop,
            self.Stage.postprocessing,
        ]

        if activity_demand_enabled or traffic_assignment_enabled:
            self.enabled_stages.add(self.Stage.supply_demand_loop)

        self.loop_substages = []
        if activity_demand_enabled:
            self.loop_substages.append(self.Stage.activity_demand)
        elif self.Stage.activity_demand_directly_from_land_use in self.enabled_stages:
            self.loop_substages.append(
                self.Stage.activity_demand_directly_from_land_use
            )
        if traffic_assignment_enabled:
            self.loop_substages.append(self.Stage.traffic_assignment)

        self._set_year_schedule(self._compute_year_schedule())

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

    def set_run_info_path(self, run_info_path: Optional[str]):
        self.run_info_path = run_info_path
        self.write_state()

    def set_sub_stage_progress(self, sub_stage_progress: Optional[str]):
        self.sub_stage_progress = sub_stage_progress
        self.write_state()

    @classmethod
    def write_stage(
        cls,
        year: int,
        current_stage: Stage,
        file_loc,
        iteration,
        asim_compiled,
        sub_stage_progress: Optional[str] = None,
        run_info_path: Optional[str] = None,
        data_initialized: bool = False,
    ):
        to_save = {
            "year": year,
            "stage": current_stage.name if current_stage else None,
            "iteration": iteration,
            "asim_compiled": asim_compiled,
            "sub_stage_progress": sub_stage_progress,
            "run_info_path": run_info_path,
            "data_initialized": data_initialized,
        }
        directory = os.path.dirname(file_loc)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_loc, mode="w", encoding="utf-8") as f:
            yaml.dump(to_save, f)

    @classmethod
    def read_current_stage(cls, file_loc):
        if not os.path.exists(file_loc):
            logger.info("Creating new stage info at {}".format(file_loc))
            return [None, None, 0, False, None, None, False]
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
        iteration = data.get("iteration", 0) or 0
        asim_compiled = data.get("asim_compiled", False)
        sub_stage_progress = data.get("sub_stage_progress", None)
        run_info_path = data.get("run_info_path", None)
        data_initialized = data.get("data_initialized", False)
        return [
            year,
            stage,
            iteration,
            asim_compiled,
            sub_stage_progress,
            run_info_path,
            data_initialized,
        ]

    @classmethod
    def from_settings(cls, settings: PilatesConfig):
        ensure_runtime_flags_initialized(settings)
        start_year = settings.run.start_year
        end_year = settings.run.end_year
        travel_model_freq = settings.run.travel_model_freq
        runtime_flags = settings.runtime.flags
        runtime_options = settings.runtime.options

        land_use_enabled = settings.land_use_enabled
        vehicle_ownership_model_enabled = settings.vehicle_ownership_model_enabled
        activity_demand_enabled = settings.activity_demand_enabled
        traffic_assignment_enabled = settings.traffic_assignment_enabled
        impacts_enabled = settings.impacts_enabled
        file_loc = settings.state_file_loc

        if file_loc:
            [
                year,
                stage,
                iteration,
                asim_compiled,
                sub_stage_progress,
                run_info_path,
                data_initialized,
            ] = cls.read_current_stage(file_loc)
        else:
            year = None
            stage = None
            iteration = 0
            asim_compiled = False
            sub_stage_progress = None
            run_info_path = None
            data_initialized = False

        resume_major_stage = stage
        resume_sub_stage = None
        loop_stage_aliases = {
            cls.Stage.activity_demand,
            cls.Stage.activity_demand_directly_from_land_use,
            cls.Stage.traffic_assignment,
        }
        if stage in loop_stage_aliases:
            resume_major_stage = cls.Stage.supply_demand_loop
            resume_sub_stage = stage

        year = year or start_year

        out = cls(
            start_year,
            end_year,
            travel_model_freq,
            land_use_enabled,
            vehicle_ownership_model_enabled,
            activity_demand_enabled,
            traffic_assignment_enabled,
            impacts_enabled,
            year,
            resume_major_stage,
            iteration,
            resume_sub_stage,
            file_loc,
            asim_compiled,
            settings,
            sub_stage_progress=sub_stage_progress,
            run_info_path=run_info_path,
            data_initialized=data_initialized,
        )

        out._settings["supply_demand_iters"] = settings.run.supply_demand_iters

        if (
            out.current_major_stage is not None
            and out.current_major_stage not in out.enabled_stages
        ):
            logger.info(
                "Persisted major stage %s is disabled in current settings; resetting to first enabled stage.",
                out.current_major_stage.name,
            )
            out.current_major_stage = None
            out.current_sub_stage = None
            out.current_inner_iter = 0

        if out.current_major_stage == out.Stage.supply_demand_loop:
            if not out.loop_substages:
                logger.info(
                    "Persisted supply-demand loop state has no enabled substages; resetting to first enabled major stage."
                )
                out.current_major_stage = None
                out.current_sub_stage = None
                out.current_inner_iter = 0
            elif (
                out.current_sub_stage is not None
                and out.current_sub_stage not in out.loop_substages
            ):
                logger.info(
                    "Persisted supply-demand substage %s is disabled in current settings; resetting to first enabled substage %s.",
                    out.current_sub_stage.name,
                    out.loop_substages[0].name,
                )
                out.current_sub_stage = out.loop_substages[0]
                out.sub_stage_progress = None

        if year:
            out.forecast_year = out._compute_forecast_year()

        if out.current_major_stage is None and not (
            out.current_year is not None and out.current_year > out.end_year
        ):
            out._initialize_first_stage()
        elif out.current_major_stage is None:
            logger.info(
                "Loaded terminal workflow state (year=%s > end_year=%s); not reinitializing stages.",
                out.current_year,
                out.end_year,
            )

        return out

    def _initialize_first_stage(self):
        for stage in self.major_stage_order:
            if stage in self.enabled_stages:
                self.current_major_stage = stage
                self.current_inner_iter = 0
                self.current_sub_stage = None
                if stage == self.Stage.supply_demand_loop and self.loop_substages:
                    self.current_sub_stage = self.loop_substages[0]
                logger.info(f"Starting workflow with stage: {stage.name}")
                self.write_state()
                break
        if self.current_major_stage is None:
            logger.info("No enabled stages found. Workflow is complete.")
            self._advance_to_next_year()

    def _compute_year_schedule(self) -> Tuple[int, ...]:
        years = [self.start_year]
        if not self._settings.get("land_use_enabled"):
            return tuple(years)

        current_year = self.start_year
        while current_year < self.end_year:
            step = 7 if current_year == 2010 else self.travel_model_freq
            if step <= 0:
                break
            next_year = min(current_year + step, self.end_year)
            if next_year <= current_year:
                break
            years.append(next_year)
            current_year = next_year
        return tuple(years)

    def _set_year_schedule(self, schedule: Tuple[int, ...]) -> None:
        normalized = list(schedule) if schedule else [self.start_year]
        if (
            self.current_year is not None
            and self.current_year <= self.end_year
            and self.current_year not in normalized
        ):
            normalized.append(self.current_year)
            normalized = sorted(set(normalized))
        self._year_schedule = tuple(normalized)
        self._year_schedule_positions = {
            year: index for index, year in enumerate(self._year_schedule)
        }
        if self.current_year is None:
            self._schedule_index = 0
        elif self.current_year > self.end_year:
            self._schedule_index = len(self._year_schedule)
        else:
            self._schedule_index = self._year_schedule_positions.get(self.current_year)

    def _advance_schedule_to_current_year(self) -> None:
        if self.current_year is None:
            self._schedule_index = 0
            return
        if self.current_year > self.end_year:
            self._schedule_index = len(self._year_schedule)
            return
        if self.current_year in self._year_schedule_positions:
            self._schedule_index = self._year_schedule_positions[self.current_year]
            return
        self._set_year_schedule(self._year_schedule)

    def _forecast_year_for_current_schedule_position(self) -> int:
        if not self._settings.get("land_use_enabled"):
            return self.current_year
        self._advance_schedule_to_current_year()
        if self._schedule_index is None:
            return self.current_year
        next_index = self._schedule_index + 1
        if next_index >= len(self._year_schedule):
            return self.end_year
        return self._year_schedule[next_index]

    def _compute_forecast_year(self) -> int:
        return self._forecast_year_for_current_schedule_position()

    def write_state(self):
        targets = []
        if self.file_loc:
            targets.append(self.file_loc)
        mirror = self.mirror_file_loc
        if mirror and mirror not in targets:
            targets.append(mirror)
        for target in targets:
            WorkflowState.write_stage(
                self.current_year,
                self.current_sub_stage or self.current_major_stage,
                target,
                self.current_inner_iter,
                self.__asim_compiled,
                self.sub_stage_progress,
                self.run_info_path,
                self.data_initialized,
            )

    def is_enabled(self, stage: Stage) -> bool:
        return stage in self.enabled_stages

    def should_run(
        self,
        target_major: Stage,
        target_inner_iter: int = 0,
        target_sub_stage: Optional[Stage] = None,
    ) -> bool:
        check_stage = target_sub_stage if target_sub_stage is not None else target_major
        if not self.is_enabled(check_stage):
            return False

        if self.current_major_stage is None:
            return True

        if target_sub_stage is None:
            try:
                current_idx = self.major_stage_order.index(self.current_major_stage)
                target_idx = self.major_stage_order.index(target_major)
                return current_idx == target_idx
            except ValueError:
                logger.error(
                    "State inconsistency: current major stage %s or target stage %s not found in major_stage_order; refusing run.",
                    self.current_major_stage.name if self.current_major_stage else None,
                    target_major.name if target_major else None,
                )
                return False

        if self.current_major_stage != self.Stage.supply_demand_loop:
            return False

        total_iters = self._settings.get("supply_demand_iters", 1)
        if self.current_inner_iter >= total_iters or self.current_inner_iter != target_inner_iter:
            return False

        if self.current_sub_stage is None:
            return True

        try:
            current_sub_idx = self.loop_substages.index(self.current_sub_stage)
            target_sub_idx = self.loop_substages.index(target_sub_stage)
            return current_sub_idx <= target_sub_idx
        except ValueError:
            logger.error(
                f"Target substage {target_sub_stage.name} not found in sequence"
            )
            return False

    def is_start_year(self):
        return self.year == self.start_year

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.current_year is not None
            and self.current_year > self.end_year
            and self.current_major_stage is None
        ):
            cleanup_paths = []
            if self.file_loc:
                cleanup_paths.append(self.file_loc)
            mirror = self.mirror_file_loc
            if mirror and mirror not in cleanup_paths:
                cleanup_paths.append(mirror)
            for path in cleanup_paths:
                if os.path.exists(path):
                    logger.info(f"Workflow finished. Removing state file: {path}")
                    os.remove(path)
            raise StopIteration

        if self.current_year is None and self.current_major_stage is None:
            logger.warning(
                "Workflow state indicates no current year and no active stage. Stopping."
            )
            raise StopIteration

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
        logger.info(
            "Completed step: (Major: %s, Iter: %s, Sub: %s) for year %d",
            completed_major.name if completed_major else None,
            completed_inner_iter,
            completed_sub.name if completed_sub else None,
            self.current_year,
        )

        if completed_sub is not None:
            if self.current_major_stage != self.Stage.supply_demand_loop:
                logger.error(
                    f"Completed substage {completed_sub.name} but not in supply-demand loop. State inconsistency."
                )
                self._advance_to_next_major_stage()
                self.write_state()
                return

            try:
                current_sub_idx = self.loop_substages.index(completed_sub)
                if current_sub_idx < len(self.loop_substages) - 1:
                    self.current_sub_stage = self.loop_substages[current_sub_idx + 1]
                    logger.debug(
                        f"Moving to next substage: {self.current_sub_stage.name}"
                    )
                else:
                    total_iters = self._settings.get("supply_demand_iters", 1)
                    if completed_inner_iter < total_iters - 1:
                        self.current_inner_iter = completed_inner_iter + 1
                        self.current_sub_stage = (
                            self.loop_substages[0] if self.loop_substages else None
                        )
                        logger.debug(
                            f"Starting next iteration: {self.current_inner_iter}"
                        )
                    else:
                        self._advance_to_next_major_stage()
            except ValueError:
                logger.error(
                    f"Completed substage {completed_sub.name} not found in sequence. State inconsistency."
                )
                self._advance_to_next_major_stage()
        elif completed_major is not None:
            if self.current_major_stage != completed_major:
                current_major_name = (
                    self.current_major_stage.name
                    if self.current_major_stage is not None
                    else "None"
                )
                logger.error(
                    f"Completed major stage {completed_major.name} but current major stage is {current_major_name}. State inconsistency."
                )
                self._advance_to_next_major_stage()
            else:
                self._advance_to_next_major_stage()

        self.write_state()

    def _advance_to_next_major_stage(self):
        if self.current_major_stage is None:
            self._advance_to_next_year()
            return

        try:
            current_idx = self.major_stage_order.index(self.current_major_stage)
            next_major = None
            for i in range(current_idx + 1, len(self.major_stage_order)):
                if self.major_stage_order[i] in self.enabled_stages:
                    next_major = self.major_stage_order[i]
                    break

            if next_major:
                self.current_major_stage = next_major
                self.current_inner_iter = 0
                self.current_sub_stage = None
                self.sub_stage_progress = None
                if next_major == self.Stage.supply_demand_loop and self.loop_substages:
                    self.current_sub_stage = self.loop_substages[0]
                logger.debug(f"Advanced to next major stage: {next_major.name}")
            else:
                logger.debug("No more enabled major stages for this year.")
                self._advance_to_next_year()
        except ValueError:
            logger.error(
                f"Current major stage {self.current_major_stage.name} not found in order. State inconsistency."
            )
            self._advance_to_next_year()

    def _advance_to_next_year(self):
        self._advance_schedule_to_current_year()
        if self._schedule_index is None:
            current_year = (
                self.current_year if self.current_year is not None else self.start_year
            )
            self._schedule_index = bisect_left(self._year_schedule, current_year)
        next_index = self._schedule_index + 1
        if next_index < len(self._year_schedule):
            self.current_year = self._year_schedule[next_index]
            self._schedule_index = next_index
        else:
            self.current_year = self.end_year + 1
            self._schedule_index = len(self._year_schedule)

        if self.current_year <= self.end_year:
            logger.info(f"Starting year {self.current_year}")
            self._initialize_first_stage()
            self.forecast_year = self._compute_forecast_year()
        else:
            logger.info(f"Workflow complete at end year {self.end_year}")
            self.current_year = self.end_year + 1
            self.current_major_stage = None
            self.current_inner_iter = 0
            self.current_sub_stage = None
            self.sub_stage_progress = None

    def set_data_initialized(self, value: bool):
        self.data_initialized = value
        self.write_state()
