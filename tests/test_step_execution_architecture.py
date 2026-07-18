"""Architecture guards for the native Consist step execution surface."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_PRODUCTION_FILES = (
    Path("pilates/workflows/stages/supply_demand_beam.py"),
    Path("pilates/workflows/steps/activitysim.py"),
    Path("pilates/workflows/steps/beam.py"),
    Path("pilates/workflows/steps/urbansim_atlas.py"),
    Path("pilates/workflows/steps/postprocessing.py"),
    Path("pilates/workflows/binding.py"),
    Path("pilates/workflows/boundary_audit.py"),
    Path("pilates/workflows/steps/shared.py"),
)
_LEGACY_FACTORY_NAMES = {
    "make_activitysim_preprocess_step",
    "make_activitysim_run_step",
    "make_activitysim_postprocess_step",
    "make_beam_preprocess_step",
    "make_beam_run_step",
    "make_beam_postprocess_step",
    "make_beam_full_skim_step",
    "make_urbansim_preprocess_step",
    "make_urbansim_run_step",
    "make_urbansim_postprocess_step",
    "make_atlas_preprocess_step",
    "make_atlas_run_step",
    "make_atlas_postprocess_step",
    "_try_restore_completed_beam_run_for_restart",
    "build_binding_plan",
    "build_key_only_binding_plan",
}
_LEGACY_IMPORT_NAMES = {
    "StageRunner",
    "BindingPlan",
    "StepBindingSpec",
    "StepOutputsHolder",
    "StandardStepSpec",
    "build_standard_step",
}


@pytest.mark.parametrize("path", _PRODUCTION_FILES)
def test_native_step_production_modules_exclude_retired_execution_symbols(
    path: Path,
) -> None:
    """Keep retired holder/factory/replay code out of the production AST."""
    tree = ast.parse(path.read_text(), filename=str(path))
    defined_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }

    assert not defined_names & _LEGACY_FACTORY_NAMES
    assert not imported_names & _LEGACY_IMPORT_NAMES
    assert not referenced_names & _LEGACY_IMPORT_NAMES
