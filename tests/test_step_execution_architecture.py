"""Architecture guards for the native Consist step execution surface."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RETIRED_EXECUTION_SYMBOLS = frozenset(
    {
        "BindingPlan",
        "StepBindingSpec",
        "StepRef",
        "StandardStepSpec",
        "StepOutputsHolder",
        "run_with_cache_recovery",
        "run_manifested_steps",
    }
)
_RETIRED_FACTORY_NAMES = frozenset(
    {
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
)
_RETIRED_SYMBOLS = _RETIRED_EXECUTION_SYMBOLS | _RETIRED_FACTORY_NAMES


def _tracked_production_python_files() -> tuple[Path, ...]:
    """Return present tracked production modules without inspecting untracked worktrees."""
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "pilates"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    files = tuple(
        _REPOSITORY_ROOT / Path(raw_path.decode())
        for raw_path in completed.stdout.split(b"\0")
        if raw_path.endswith(b".py")
        and (_REPOSITORY_ROOT / Path(raw_path.decode())).is_file()
    )
    assert files, "Expected tracked production Python modules under pilates/."
    return files


def _retired_symbols_in(tree: ast.AST) -> set[str]:
    """Find retired execution API use through Python syntax, not text matching."""
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    imports = {
        candidate
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        for candidate in (alias.name, alias.asname)
        if candidate is not None
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    return _RETIRED_SYMBOLS & (defined | imports | names | attributes)


def test_native_step_production_modules_exclude_retired_execution_symbols() -> None:
    """Every tracked production module must stay on the single native path."""
    violations: dict[Path, set[str]] = {}
    for path in _tracked_production_python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        retired = _retired_symbols_in(tree)
        if retired:
            violations[path.relative_to(_REPOSITORY_ROOT)] = retired

    assert not violations, "Retired execution symbols found: " + repr(violations)


def test_ast_guard_detects_retired_execution_imports_and_references() -> None:
    """Keep the all-production scan sensitive to the prohibited API forms."""
    imports = "\n".join(
        f"from retired_execution import {symbol}"
        for symbol in sorted(_RETIRED_EXECUTION_SYMBOLS)
    )
    attributes = "\n".join(
        f"retired_execution.{symbol}" for symbol in sorted(_RETIRED_EXECUTION_SYMBOLS)
    )

    assert _retired_symbols_in(ast.parse(imports)) >= _RETIRED_EXECUTION_SYMBOLS
    assert _retired_symbols_in(ast.parse(attributes)) >= _RETIRED_EXECUTION_SYMBOLS
