from enum import Enum
from typing import Optional
import os
import yaml
import logging

from pilates.config import PilatesConfig
from pilates.workflows._profile import ensure_runtime_flags_initialized

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
        year: int | None,
        major_stage: Stage | None,
        inner_iter: int,
        sub_stage: Stage | None,
        file_loc: Optional[str],
        asim_compiled: bool,
        full_settings: PilatesConfig,
        sub_stage_progress: Optional[str] = None,
        run_info_path: Optional[str] = None,  # new
        data_initialized: bool = False,
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
        self.sub_stage_progress = sub_stage_progress
        self.run_info_path = run_info_path  # new
        # Preserve whether the run *started* as a restart before launcher later
        # populates ``run_info_path`` for fresh runs too.
        self.is_restart_run = bool(run_info_path)
        self.data_initialized = data_initialized

        self.forecast_year: int | None = None
        self.file_loc = file_loc
        self.mirror_file_loc: Optional[str] = None

        self.__asim_compiled = asim_compiled

        # Store settings for access by methods that need them
        self._settings = {
            "supply_demand_iters": 1,  # Default, will be updated in from_settings
            "land_use_enabled": land_use_enabled,
            "vehicle_ownership_model_enabled": vehicle_ownership_model_enabled,
            "activity_demand_enabled": activity_demand_enabled,
            "traffic_assignment_enabled": traffic_assignment_enabled,
        }
        self.full_settings = full_settings or {}

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
    def from_settings(
        cls,
        settings: PilatesConfig,
    ):
        ensure_runtime_flags_initialized(settings)
        # Get settings from nested or legacy locations
        start_year = settings.run.start_year
        end_year = settings.run.end_year
        travel_model_freq = settings.run.travel_model_freq
        runtime_flags = settings.runtime.flags
        runtime_options = settings.runtime.options

        # These are always added at top-level by parse_args_and_settings()
        land_use_enabled = runtime_flags.land_use_enabled
        vehicle_ownership_model_enabled = runtime_flags.vehicle_ownership_model_enabled
        activity_demand_enabled = runtime_flags.activity_demand_enabled
        traffic_assignment_enabled = runtime_flags.traffic_assignment_enabled
        file_loc = runtime_options.state_file_loc

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

        # If restarting from a saved state whose major stage is now disabled
        # (common during config/migration changes), reset to the first enabled stage.
        if (
            out.current_major_stage is not None
            and out.current_major_stage not in out.enabled_stages
        ):
            logger.info(
                "Persisted major stage %s is disabled in current settings; "
                "resetting to first enabled stage.",
                out.current_major_stage.name,
            )
            out.current_major_stage = None
            out.current_sub_stage = None
            out.current_inner_iter = 0

        if out.current_major_stage == out.Stage.supply_demand_loop:
            if not out.loop_substages:
                logger.info(
                    "Persisted supply-demand loop state has no enabled substages; "
                    "resetting to first enabled major stage."
                )
                out.current_major_stage = None
                out.current_sub_stage = None
                out.current_inner_iter = 0
            elif (
                out.current_sub_stage is not None
                and out.current_sub_stage not in out.loop_substages
            ):
                logger.info(
                    "Persisted supply-demand substage %s is disabled in current settings; "
                    "resetting to first enabled substage %s.",
                    out.current_sub_stage.name,
                    out.loop_substages[0].name,
                )
                out.current_sub_stage = out.loop_substages[0]
                out.sub_stage_progress = None

        if year:
            out.forecast_year = out._compute_forecast_year()

        # Initialize state only when we're not already in a terminal completed state.
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

    def _interval_step_for_year(self, year: int) -> int:
        # Keep 2010 as a one-time bridge into regular interval boundaries.
        return 7 if year == 2010 else self.travel_model_freq

    def _compute_forecast_year(self) -> int:
        if not self._settings.get("land_use_enabled"):
            return self.current_year
        next_year_candidate = self.current_year + self._interval_step_for_year(
            self.current_year
        )
        return min(next_year_candidate, self.end_year)

    def _next_current_year(self) -> int:
        if not self._settings.get("land_use_enabled"):
            return self.end_year + 1

        next_year = (
            self.forecast_year
            if self.forecast_year is not None
            else self._compute_forecast_year()
        )
        # Guard against stale or capped forecast years that would cause
        # backward moves or no-op loops.
        if next_year <= self.current_year:
            return self.end_year + 1
        return next_year

    def write_state(self):
        """Save the current state to file"""
        targets = []
        if self.file_loc:
            targets.append(self.file_loc)
        mirror = getattr(self, "mirror_file_loc", None)
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
        # Check if the target stage is enabled
        check_stage = target_sub_stage if target_sub_stage is not None else target_major
        if not self.is_enabled(check_stage):
            return False

        # If we haven't started yet, allow any enabled stage
        if self.current_major_stage is None:
            return True

        # For major stages (not substages)
        if target_sub_stage is None:
            # Only run the current major stage.
            try:
                current_idx = self.major_stage_order.index(self.current_major_stage)
                target_idx = self.major_stage_order.index(target_major)
                return current_idx == target_idx
            except ValueError:
                logger.error(
                    "State inconsistency: current major stage %s or target stage %s "
                    "not found in major_stage_order; refusing run.",
                    self.current_major_stage.name if self.current_major_stage else None,
                    target_major.name if target_major else None,
                )
                return False

        # For substages within the supply-demand loop
        if target_sub_stage is not None:
            # Must be in the supply-demand loop to run substages
            if self.current_major_stage != self.Stage.supply_demand_loop:
                return False

            # Check iteration first
            total_iters = self._settings.get("supply_demand_iters", 1)
            if self.current_inner_iter >= total_iters:
                return False

            if self.current_inner_iter != target_inner_iter:
                return False

            # Check substage position
            if self.current_sub_stage is None:
                # At start of iteration, can run any substage
                return True
            else:
                try:
                    current_sub_idx = self.loop_substages.index(self.current_sub_stage)
                    target_sub_idx = self.loop_substages.index(target_sub_stage)
                    result = current_sub_idx <= target_sub_idx
                    return result
                except ValueError:
                    logger.error(
                        f"Target substage {target_sub_stage.name} not found in sequence"
                    )
                    return False

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
            cleanup_paths = []
            if self.file_loc:
                cleanup_paths.append(self.file_loc)
            mirror = getattr(self, "mirror_file_loc", None)
            if mirror and mirror not in cleanup_paths:
                cleanup_paths.append(mirror)
            for path in cleanup_paths:
                if os.path.exists(path):
                    logger.info(f"Workflow finished. Removing state file: {path}")
                    os.remove(path)
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
                current_major_name = (
                    self.current_major_stage.name
                    if self.current_major_stage is not None
                    else "None"
                )
                logger.error(
                    f"Completed major stage {completed_major.name} but current major stage is {current_major_name}. State inconsistency."
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
                self.sub_stage_progress = None  # new

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
        """Move to the next interval boundary year and reset to first stage"""
        self.current_year = self._next_current_year()

        if self.current_year <= self.end_year:
            logger.info(f"Starting year {self.current_year}")
            # Reset to the first enabled major stage for the new year
            self._initialize_first_stage()
            self.forecast_year = self._compute_forecast_year()

        else:
            logger.info(f"Workflow complete at end year {self.end_year}")
            self.current_year = (
                self.end_year + 1
            )  # Set year just past end_year to signal completion
            self.current_major_stage = None
            self.current_inner_iter = 0
            self.current_sub_stage = None
            self.sub_stage_progress = None  # new
            # The __next__ method will handle state file removal when it detects completion

    def set_data_initialized(self, value: bool):
        self.data_initialized = value
        self.write_state()
