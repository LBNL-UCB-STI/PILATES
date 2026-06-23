from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PILATES_ROOT = REPO_ROOT / "pilates"
WORKFLOW_STATE_PATH = REPO_ROOT / "workflow_state.py"
WORKFLOW_STAGES_ROOT = PILATES_ROOT / "workflows" / "stages"

ALLOWED_PROFILE_IMPORT_FILES: set[Path] = set()
ALLOWED_RUNTIME_FLAG_CALL_FILES = {
    Path("workflow_state.py"),
    Path("pilates/generic/initialization.py"),
    Path("pilates/runtime/launcher.py"),
    Path("pilates/workflows/surface.py"),
}
DELETED_RESTART_AND_AUDIT_SYMBOLS = {
    "RestartExactRewindContract",
    "_copy_historical_artifact_to_current",
    "_materialize_run_output_paths",
    "_remap_outputs_workspace_paths",
    "_remap_workspace_local_path",
    "_resolve_historical_workspace_artifact_path",
    "emit_artifact_lifecycle_audit_event",
    "emit_consist_audit_event",
    "hydrate_missing_restart_artifacts",
    "hydrate_rewind_runner_inputs",
    "restart_exact_rewind_contract",
}
ALLOWED_DIRECT_MANIFEST_CONFIG_STAGE_IMPORTS = {
    Path("pilates/workflows/stages/land_use.py"),
    Path("pilates/workflows/stages/postprocessing.py"),
    Path("pilates/workflows/stages/supply_demand.py"),
    Path("pilates/workflows/stages/supply_demand_activity.py"),
    Path("pilates/workflows/stages/vehicle_ownership.py"),
}


def _production_python_files() -> Iterable[Path]:
    for path in sorted(PILATES_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path
    yield WORKFLOW_STATE_PATH


def _stage_python_files() -> Iterable[Path]:
    for path in sorted(WORKFLOW_STAGES_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _relative(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_arg_names(path: Path, function_name: str) -> list[str]:
    tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [arg.arg for arg in node.args.posonlyargs]
            args.extend(arg.arg for arg in node.args.args)
            args.extend(arg.arg for arg in node.args.kwonlyargs)
            return args
    raise AssertionError(f"Could not find function {function_name!r} in {path}")


def test_production_code_only_imports_profile_from_the_compat_shim() -> None:
    violations: list[str] = []

    for path in _production_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "pilates.workflows.profile"
            ):
                if rel not in ALLOWED_PROFILE_IMPORT_FILES:
                    violations.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name == "pilates.workflows.profile"
                        and rel not in ALLOWED_PROFILE_IMPORT_FILES
                    ):
                        violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "Profile shim imports are only allowed in the compatibility shim itself. "
        f"Violations: {violations}"
    )


def test_production_code_does_not_call_build_workflow_profile() -> None:
    violations: list[str] = []

    for path in _production_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "build_workflow_profile"
            ):
                violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        f"build_workflow_profile() should be gone from production code: {violations}"
    )


def test_production_code_does_not_build_a_surface_only_to_read_profile() -> None:
    violations: list[str] = []

    for path in _production_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "profile":
                continue
            if not isinstance(node.value, ast.Call):
                continue
            call_name = _call_name(node.value)
            if call_name == "build_enabled_workflow_surface":
                violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "Do not call build_enabled_workflow_surface(...).profile as a shortcut for booleans. "
        f"Violations: {violations}"
    )


def test_runtime_flag_initialization_only_happens_in_approved_modules() -> None:
    call_sites: set[Path] = set()

    for path in _production_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "ensure_runtime_flags_initialized"
            ):
                call_sites.add(rel)

    assert call_sites == ALLOWED_RUNTIME_FLAG_CALL_FILES


def test_binding_plan_call_sites_pass_surface_explicitly() -> None:
    violations: list[str] = []

    for path in _production_python_files():
        rel = _relative(path)
        if rel == Path("pilates/workflows/binding.py"):
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in {
                "build_binding_plan",
                "beam_preprocess_binding_plan",
            }:
                continue
            if not any(keyword.arg == "surface" for keyword in node.keywords):
                violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "Binding-plan calls outside binding.py should stay surface-driven. "
        f"Violations: {violations}"
    )


def test_surface_driven_entry_points_do_not_accept_profile_any_more() -> None:
    binding_path = REPO_ROOT / "pilates/workflows/binding.py"
    planning_path = REPO_ROOT / "pilates/workflows/planning.py"
    runtime_path = REPO_ROOT / "pilates/runtime/scenario_runtime.py"

    for path, function_name in (
        (binding_path, "build_binding_plan"),
        (binding_path, "beam_preprocess_binding_plan"),
        (planning_path, "build_static_execution_plan"),
        (runtime_path, "filter_schema_steps_for_enabled_models"),
    ):
        arg_names = _function_arg_names(path, function_name)
        assert "surface" in arg_names
        assert "profile" not in arg_names


def test_legacy_archive_doctor_stays_deleted() -> None:
    legacy_doctor_path = REPO_ROOT / "pilates/runtime/legacy_archive_doctor.py"

    assert not legacy_doctor_path.exists()


def test_deleted_restart_and_audit_symbols_stay_out_of_production_code() -> None:
    violations: list[str] = []

    for path in _production_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in DELETED_RESTART_AND_AUDIT_SYMBOLS:
                    violations.append(f"{rel}:{node.lineno}:defines:{node.name}")
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in DELETED_RESTART_AND_AUDIT_SYMBOLS:
                        violations.append(f"{rel}:{node.lineno}:imports:{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_name = alias.name.rsplit(".", maxsplit=1)[-1]
                    if imported_name in DELETED_RESTART_AND_AUDIT_SYMBOLS:
                        violations.append(
                            f"{rel}:{node.lineno}:imports:{imported_name}"
                        )

    assert not violations, (
        "Deleted restart hydration and audit-emitter APIs must not be defined "
        f"or imported by production code. Violations: {violations}"
    )


def test_stage_manifest_config_imports_stay_on_migration_allowlist() -> None:
    direct_imports: set[Path] = set()

    for path in _stage_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "pilates.workflows.orchestration"
                and any(alias.name == "ManifestConfig" for alias in node.names)
            ):
                direct_imports.add(rel)

    assert direct_imports == ALLOWED_DIRECT_MANIFEST_CONFIG_STAGE_IMPORTS


def test_archive_materialization_flag_stays_out_of_production_code() -> None:
    violations: list[str] = []

    for path in _production_python_files():
        rel = _relative(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_names = [arg.arg for arg in node.args.args]
                arg_names.extend(arg.arg for arg in node.args.kwonlyargs)
                if "materialize_from_archive" in arg_names:
                    violations.append(
                        f"{rel}:{node.lineno}:defines-arg:materialize_from_archive"
                    )
            elif isinstance(node, ast.Call):
                if any(
                    keyword.arg == "materialize_from_archive"
                    for keyword in node.keywords
                ):
                    violations.append(
                        f"{rel}:{node.lineno}:passes-kwarg:materialize_from_archive"
                    )

    assert not violations, (
        "Archive materialization should go through Consist artifact materializers, "
        f"not resolve_existing_path(..., materialize_from_archive=True). "
        f"Violations: {violations}"
    )
