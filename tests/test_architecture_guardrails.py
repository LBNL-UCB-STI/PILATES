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
            if isinstance(node, ast.ImportFrom) and node.module == "pilates.workflows.profile":
                if rel not in ALLOWED_PROFILE_IMPORT_FILES:
                    violations.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pilates.workflows.profile" and rel not in ALLOWED_PROFILE_IMPORT_FILES:
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
            if isinstance(node, ast.Call) and _call_name(node) == "build_workflow_profile":
                violations.append(f"{rel}:{node.lineno}")

    assert not violations, f"build_workflow_profile() should be gone from production code: {violations}"


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
            if isinstance(node, ast.Call) and _call_name(node) == "ensure_runtime_flags_initialized":
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
            if _call_name(node) not in {"build_binding_plan", "beam_preprocess_binding_plan"}:
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
