from pilates.generic.records import RecordStore
from pilates.workspace import Workspace
from pilates.utils.provenance import FileProvenanceTracker

import os

class Initialization:
    """
    A dedicated class to handle the initialization of mutable data and
    provenance recording. This class consolidates input data copying for
    all models into one single initialization job. It does not conform to the
    GenericPreprocessor interface.
    """

    @staticmethod
    def run(settings: dict, workspace: Workspace, provenance_tracker: FileProvenanceTracker) -> None:
        """
        Execute the initialization process:
          - Copy all necessary input data from production directories to mutable locations.
          - Record the input and output file locations as provenance for the initialization job.
        """
        initialization_records_in = RecordStore()
        initialization_records_out = RecordStore()
        have_not_copied_usim_data = True

        # BEAM model initialization
        if settings.get("travel_model") == "beam":
            from pilates.beam import preprocessor as beam_pre

            beam_input_dir = workspace.get_beam_mutable_data_dir()
            rec_in, rec_out = beam_pre.copy_data_to_mutable_location(
                settings, beam_input_dir, provenance_tracker
            )
            initialization_records_in += rec_in
            initialization_records_out += rec_out
            # Make available to later preprocess()
            workspace.input_data["beam"] = rec_in
            workspace.output_data["beam"] = rec_out

        # Other models
        for model_key in [
            "activity_demand_model",
            "vehicle_ownership_model",
            "land_use_model",
        ]:
            model_name = settings.get(model_key)
            if not model_name:
                continue

            # UrbanSim data copy (once)
            if model_name == "urbansim" or (
                model_name == "activitysim" and have_not_copied_usim_data
            ):
                from pilates.urbansim import preprocessor as usim_pre

                output_dir = workspace.get_usim_mutable_data_dir()
                os.makedirs(output_dir, exist_ok=True)
                rec_in, rec_out = usim_pre.copy_data_to_mutable_location(
                    settings, output_dir, provenance_tracker
                )
                if model_name in workspace.input_data:
                    workspace.input_data[model_name] += rec_in
                else:
                    workspace.input_data[model_name] = rec_in
                if model_name in workspace.output_data:
                    workspace.output_data[model_name] += rec_out
                else:
                    workspace.output_data[model_name] = rec_out
                have_not_copied_usim_data = False

            # Atlas data copy
            if model_name == "atlas":
                input_dir = workspace.get_atlas_mutable_input_dir()
                os.makedirs(input_dir, exist_ok=True)
                from pilates.atlas import preprocessor as atlas_pre

                atlas_pre.copy_data_to_mutable_location(settings, input_dir)
                os.makedirs(workspace.get_atlas_output_dir(), exist_ok=True)
                # No input/output records for atlas currently

            # ActivitySim config copy
            if model_name == "activitysim":
                from pilates.activitysim import preprocessor as asim_pre

                asim_input_dir = workspace.get_asim_mutable_data_dir()
                rec_in, rec_out = asim_pre.copy_data_to_mutable_location(
                    settings, asim_input_dir, provenance_tracker
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
        init_run_hash = provenance_tracker.start_model_run(
            "initialization",
            year=settings.get("start_year"),
            iteration=0,
            description="Initialization: copying all mutable data",
            inputs=initialization_records_in,
        )
        provenance_tracker.complete_model_run(
            run_hash=init_run_hash, output_records=initialization_records_out
        )
