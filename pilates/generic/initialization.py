from typing import Optional

from pilates.config import PilatesConfig
from pilates.generic.model import Model
from pilates.generic.records import RecordStore
from pilates.workspace import Workspace
from pilates.utils.provenance import FileProvenanceTracker
from pilates.generic.model_factory import ModelFactory

import os
from pilates.utils.settings_helper import get as get_setting
import logging

logger = logging.getLogger(__name__)


class Initialization(Model):
    """
    A dedicated class to handle the initialization of mutable data and
    provenance recording. This class consolidates input data copying for
    all models into one single initialization job. It does not conform to the
    GenericPreprocessor interface.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
        provenance_tracker: Optional[FileProvenanceTracker],
    ):
        super().__init__(model_name, state, provenance_tracker)

    def run(self, settings: PilatesConfig, workspace: Workspace) -> None:
        """
        Execute the initialization process:
          - Copy all necessary input data from production directories to mutable locations.
          - Record the input and output file locations as provenance for the initialization job.
        """
        initialization_records_in = RecordStore()
        initialization_records_out = RecordStore()
        have_not_copied_usim_data = True
        model_factory = ModelFactory()

        # BEAM model initialization
        if settings.run.models.travel == "beam":
            beam_preprocessor = model_factory.get_preprocessor(
                "beam", self.state, self.provenance_tracker
            )

            beam_input_dir = workspace.get_beam_mutable_data_dir()
            result = beam_preprocessor.copy_data_to_mutable_location(
                settings, beam_input_dir
            )
            if result:
                rec_in, rec_out = result
                initialization_records_in += rec_in
                initialization_records_out += rec_out
                # Make available to later preprocess()
                workspace.input_data["beam"] = rec_in
                workspace.output_data["beam"] = rec_out

        # Other models
        model_map = {
            "activity_demand": settings.run.models.activity_demand,
            "vehicle_ownership": settings.run.models.vehicle_ownership,
            "land_use": settings.run.models.land_use,
        }

        for model_key, model_name in model_map.items():
            if not model_name:
                continue

            # UrbanSim data copy (once)
            if model_name == "urbansim" or (
                model_name == "activitysim" and have_not_copied_usim_data
            ):

                output_dir = workspace.get_usim_mutable_data_dir()
                os.makedirs(output_dir, exist_ok=True)
                usim_preprocessor = model_factory.get_preprocessor(
                    "urbansim", self.state, self.provenance_tracker
                )
                result = usim_preprocessor.copy_data_to_mutable_location(
                    settings, output_dir
                )
                if result:
                    rec_in, rec_out = result
                    if model_name in workspace.input_data:
                        workspace.input_data[model_name] += rec_in
                    else:
                        workspace.input_data[model_name] = rec_in
                    if model_name in workspace.output_data:
                        workspace.output_data[model_name] += rec_out
                    else:
                        workspace.output_data[model_name] = rec_out
                have_not_copied_usim_data = False
                if result:
                    rec_in, rec_out = result
                    initialization_records_in += rec_in
                    initialization_records_out += rec_out

            # Atlas data copy
            if model_name == "atlas":
                input_dir = workspace.get_atlas_mutable_input_dir()
                os.makedirs(input_dir, exist_ok=True)
                atlas_preprocessor = model_factory.get_preprocessor(
                    "atlas", self.state, self.provenance_tracker
                )
                rec_in, rec_out = atlas_preprocessor.copy_data_to_mutable_location(
                    settings, input_dir
                )
                initialization_records_in += rec_in
                initialization_records_out += rec_out
                os.makedirs(workspace.get_atlas_output_dir(), exist_ok=True)

            # ActivitySim config copy
            if model_name == "activitysim":

                activitysim_preprocessor = model_factory.get_preprocessor(
                    model_name, self.state, self.provenance_tracker
                )
                asim_input_dir = workspace.get_asim_mutable_data_dir()
                rec_in, rec_out = (
                    activitysim_preprocessor.copy_data_to_mutable_location(
                        settings, asim_input_dir
                    )
                )
                initialization_records_in += rec_in
                initialization_records_out += rec_out
                if model_name in workspace.input_data:
                    workspace.input_data[model_name] += rec_in
                else:
                    workspace.input_data[model_name] = rec_in
                if model_name in workspace.output_data:
                    workspace.output_data[model_name] += rec_out
                else:
                    workspace.output_data[model_name] = rec_out

        # You can add further model-specific blocks (e.g., for urbansim, atlas) as needed

        # Record the combined initialization provenance as a single job.
        if self.provenance_tracker is not None:
            init_run_hash = self.provenance_tracker.start_model_run(
                "initialization",
                year=get_setting(settings, "run.start_year"),
                iteration=0,
                description="Initialization: copying all mutable data",
                inputs=initialization_records_in,
            )
            self.provenance_tracker.complete_model_run(
                run_hash=init_run_hash,
                output_records=initialization_records_out.all_records(),
            )
        # If no provenance tracker is supplied, we simply skip provenance logging.
