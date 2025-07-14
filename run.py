from pilates.generic.records import ModelRunInfo
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.provenance import FileProvenanceTracker, OpenLineageTracker
from pilates.atlas.runner import AtlasRunner
from pilates.urbansim.runner import UrbansimRunner
from pilates.activitysim.runner import ActivitysimRunner
from pilates.beam.runner import BeamRunner

def run_atlas(
    settings,
    state: WorkflowState,
    client,
    workspace: Workspace,
    provenance_tracker: OpenLineageTracker,
    warm_start_atlas,
    forecast=False,
    atlas_run_count=1,
):
    # ...

    from pilates.atlas import preprocessor as atlas_pre

    atlas_pre_hash = provenance_tracker.start_model_run(
        f"atlas_preprocessor", state.current_year, description="Atlas preprocessing"
    )

    # ...

    yrs = range(state.start_year, state.end_year + 1)

    for yr_it in yrs:
        if warm_start_atlas and yr_it == state.start_year:
            # Use warm-start input files
            pass
        else:
            # Use generated input files
            atlas_pre.prepare_atlas_inputs(
                settings,
                yr_it,
                workspace,
                provenance_tracker,
                model_run_hash=atlas_pre_hash,
                warm_start=False,
            )

    # ...
