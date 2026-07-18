"""
Workflow step public surface.

The public surface exposes only committed native ``StepDefinition`` objects.
"""

from __future__ import annotations

from typing import Any

from pilates.workflows.step_definition import StepDefinition

from .shared import (
    validate_workflow_step_contracts as validate_workflow_step_contracts,
    _activitysim_output_facet_meta as _activitysim_output_facet_meta,
    _atlas_artifact_facet_meta as _atlas_artifact_facet_meta,
    _beam_artifact_facets as _beam_artifact_facets,
    _beam_log_facet_meta as _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta as _beam_postprocess_split_facet_meta,
    _urbansim_output_facet_meta as _urbansim_output_facet_meta,
)
from .activitysim import (
    activitysim_postprocess as activitysim_postprocess,
    activitysim_preprocess as activitysim_preprocess,
    activitysim_run as activitysim_run,
)
from .beam import (
    beam_full_skim as beam_full_skim,
    beam_postprocess as beam_postprocess,
    beam_preprocess as beam_preprocess,
    beam_run as beam_run,
)
from .postprocessing import (
    postprocessing as postprocessing_definition,
)
from .urbansim_atlas import (
    atlas_postprocess as atlas_postprocess,
    atlas_preprocess as atlas_preprocess,
    atlas_run as atlas_run,
    urbansim_postprocess as urbansim_postprocess,
    urbansim_preprocess as urbansim_preprocess,
    urbansim_run as urbansim_run,
)

# Re-export modules for callers/tests that monkeypatch module-level symbols.
from . import (
    activitysim as activitysim,
    beam as beam,
    postprocessing as postprocessing,
    shared as shared,
    urbansim_atlas as urbansim_atlas,
)


STEP_DEFINITIONS: dict[str, StepDefinition[Any]] = {
    definition.name: definition
    for definition in (
        urbansim_preprocess,
        urbansim_run,
        urbansim_postprocess,
        atlas_preprocess,
        atlas_run,
        atlas_postprocess,
        activitysim_preprocess,
        activitysim_run,
        activitysim_postprocess,
        beam_preprocess,
        beam_run,
        beam_postprocess,
        beam_full_skim,
        postprocessing_definition,
    )
}
