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

    # ... (rest of WorkflowState class unchanged, see user message for full code)
