from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PILATES_ROOT = REPO_ROOT / "pilates"
WORKFLOW_STATE_PATH = REPO_ROOT / "workflow_state.py"

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
    "activitysim_datastore_selection_rules",
    "build_activitysim_inputs",
    "build_atlas_inputs",
    "build_urbansim_inputs",
    "canonicalize_artifact_mapping",
    "emit_artifact_lifecycle_audit_event",
    "emit_consist_audit_event",
    "hydrate_missing_restart_artifacts",
    "hydrate_rewind_runner_inputs",
    "log_coupler_value",
    "resolve_artifact_from_value",
    "resolve_artifact_key",
    "restart_exact_rewind_contract",
    "urbansim_datastore_selection_rules",
}
ZERO_GETATTR_MODULES = (Path("pilates/workflows/binding.py"),)


def _production_python_files() -> Iterable[Path]:
    for path in sorted(PILATES_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path
    yield WORKFLOW_STATE_PATH


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


def test_native_step_entry_points_are_surface_and_resolved_input_driven() -> None:
    planning_path = REPO_ROOT / "pilates/workflows/planning.py"
    runtime_path = REPO_ROOT / "pilates/runtime/scenario_runtime.py"
    execution_path = REPO_ROOT / "pilates/workflows/step_execution.py"

    for path, function_name in (
        (planning_path, "build_static_execution_plan"),
        (runtime_path, "filter_schema_steps_for_enabled_models"),
    ):
        arg_names = _function_arg_names(path, function_name)
        assert "surface" in arg_names
        assert "profile" not in arg_names

    execution_arg_names = _function_arg_names(execution_path, "execute_step")
    assert {"definition", "resolved_inputs"} <= set(execution_arg_names)
    assert "profile" not in execution_arg_names


def test_legacy_archive_doctor_stays_deleted() -> None:
    legacy_doctor_path = REPO_ROOT / "pilates/runtime/legacy_archive_doctor.py"

    assert not legacy_doctor_path.exists()


def test_artifact_key_migrations_stays_deleted() -> None:
    migrations_path = REPO_ROOT / "pilates/workflows/artifact_key_migrations.py"

    assert not migrations_path.exists()


def test_legacy_model_inputs_modules_stay_deleted() -> None:
    for rel_path in (
        "pilates/activitysim/inputs.py",
        "pilates/urbansim/inputs.py",
        "pilates/atlas/inputs.py",
    ):
        assert not (REPO_ROOT / rel_path).exists()


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


def _is_type_checking_guard(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _collect_runtime_imports(tree: ast.AST) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            for orelse_node in node.orelse:
                _visit(orelse_node)
            return
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
            return
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return imports


def test_config_and_utils_do_not_import_runtime_or_workflows() -> None:
    forbidden_prefixes = ("pilates.runtime", "pilates.workflows")
    violations: list[str] = []

    for subdir in ("config", "utils"):
        for path in sorted((PILATES_ROOT / subdir).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = _relative(path)
            tree = _parse(path)
            for lineno, module in _collect_runtime_imports(tree):
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{rel}:{lineno}:{module}")

    assert not violations, (
        "pilates/config/ and pilates/utils/ must not import from pilates.runtime "
        f"or pilates.workflows at runtime. Violations: {violations}"
    )


def test_zero_getattr_modules_stay_free_of_getattr_and_setattr() -> None:
    violations: list[str] = []

    for rel in ZERO_GETATTR_MODULES:
        path = REPO_ROOT / rel
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in {
                "getattr",
                "setattr",
            }:
                violations.append(f"{rel}:{node.lineno}:{_call_name(node)}")

    assert not violations, (
        "Typed workflow modules must use narrow protocols instead of getattr/setattr. "
        f"Violations: {violations}"
    )
