"""
Workflow step public surface.

Shared step infrastructure lives in ``shared.py`` and model-specific factory
implementations live in sibling modules.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from pilates.generic.model_factory import ModelFactory as ModelFactory

from .shared import (
    STEP_OUTPUTS_CLASSES as STEP_OUTPUTS_CLASSES,
    StepOutputsHolder as StepOutputsHolder,
    validate_step_ready as validate_step_ready,
    validate_workflow_step_contracts as validate_workflow_step_contracts,
    _activitysim_output_facet_meta as _activitysim_output_facet_meta,
    _atlas_artifact_facet_meta as _atlas_artifact_facet_meta,
    _beam_artifact_facets as _beam_artifact_facets,
    _beam_log_facet_meta as _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta as _beam_postprocess_split_facet_meta,
    _urbansim_output_facet_meta as _urbansim_output_facet_meta,
)
from .activitysim import (
    activitysim_compile_output_paths as activitysim_compile_output_paths,
    make_activitysim_compile_step as make_activitysim_compile_step,
    make_activitysim_postprocess_step as make_activitysim_postprocess_step,
    make_activitysim_preprocess_step as make_activitysim_preprocess_step,
    make_activitysim_run_step as make_activitysim_run_step,
)
from .beam import (
    make_beam_full_skim_step as make_beam_full_skim_step,
    make_beam_postprocess_step as make_beam_postprocess_step,
    make_beam_preprocess_step as make_beam_preprocess_step,
    make_beam_run_step as make_beam_run_step,
)
from .postprocessing import (
    make_postprocessing_step as make_postprocessing_step,
)
from .urbansim_atlas import (
    make_atlas_postprocess_step as make_atlas_postprocess_step,
    make_atlas_preprocess_step as make_atlas_preprocess_step,
    make_atlas_run_step as make_atlas_run_step,
    make_urbansim_postprocess_step as make_urbansim_postprocess_step,
    make_urbansim_preprocess_step as make_urbansim_preprocess_step,
    make_urbansim_run_step as make_urbansim_run_step,
    urbansim_run_output_paths as urbansim_run_output_paths,
)

# Re-export modules for callers/tests that monkeypatch module-level symbols.
from . import (
    activitysim as activitysim,
    beam as beam,
    postprocessing as postprocessing,
    shared as shared,
    urbansim_atlas as urbansim_atlas,
)


SCHEMA_STEP_BUILDERS: Dict[str, Callable[..., Any]] = {
    "urbansim_preprocess": make_urbansim_preprocess_step,
    "urbansim_run": make_urbansim_run_step,
    "urbansim_postprocess": make_urbansim_postprocess_step,
    "atlas_preprocess": make_atlas_preprocess_step,
    "atlas_run": make_atlas_run_step,
    "atlas_postprocess": make_atlas_postprocess_step,
    "activitysim_preprocess": make_activitysim_preprocess_step,
    "activitysim_compile": make_activitysim_compile_step,
    "activitysim_run": make_activitysim_run_step,
    "activitysim_postprocess": make_activitysim_postprocess_step,
    "beam_preprocess": make_beam_preprocess_step,
    "beam_run": make_beam_run_step,
    "beam_postprocess": make_beam_postprocess_step,
    "beam_full_skim": make_beam_full_skim_step,
}


def schema_step_builder_registry() -> Dict[str, Callable[..., Any]]:
    """
    Return the canonical schema-step builder registry keyed by step name.

    The returned mapping is a shallow copy so callers can inspect or filter it
    without mutating the shared registration surface.

    This registry intentionally stays separate from the workflow catalog: the
    catalog is static contract metadata, while this module owns the executable
    builder functions that close over model-local runtime behavior.
    """
    return dict(SCHEMA_STEP_BUILDERS)
