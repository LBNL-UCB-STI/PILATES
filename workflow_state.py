from enum import Enum
from typing import Optional
import os
import yaml
import logging

from pilates.config import PilatesConfig

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
        file_loc: str,
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
        self.data_initialized = data_initialized

        self.forecast_year = None
        self.file_loc = file_loc

        self.__asim_compiled = asim_compiled
        self.initial_step = 7 if self.current_year == 2010 else None

        # Store settings for access by methods that need them
        self._settings = {
            "end_year": end_year,
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

    @property
    def settings(self):
        return (
            self._settings
            if hasattr(self, "_settings")
            else {"end_year": self.end_year}
        )

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

    def enabled(self, stage):
        return stage in self.enabled_stages

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
        # Get settings from nested or legacy locations
        start_year = settings.run.start_year
        end_year = settings.run.end_year
        travel_model_freq = settings.run.travel_model_freq

        # These are always added at top-level by parse_args_and_settings()
        land_use_enabled = getattr(settings, "land_use_enabled", False)
        vehicle_ownership_model_enabled = getattr(
            settings, "vehicle_ownership_model_enabled", False
        )
        activity_demand_enabled = getattr(settings, "activity_demand_enabled", False)
        traffic_assignment_enabled = getattr(
            settings, "traffic_assignment_enabled", False
        )
        replanning_enabled = getattr(settings, "replanning_enabled", False)
        file_loc = getattr(settings, "state_file_loc", "current_stage.yaml")

        [
            year,
            stage,
            iteration,
            asim_compiled,
            sub_stage_progress,
            run_info_path,
            data_initialized,
        ] = cls.read_current_stage(file_loc)

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
                logger.warning(
                    f"Target major stage {target_major.name} not found in order, allowing run"
                )
                return True

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
            self.sub_stage_progress = None  # new
            # The __next__ method will handle state file removal when it detects completion

    def set_data_initialized(self, value: bool):
        self.data_initialized = value
        self.write_state()

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
