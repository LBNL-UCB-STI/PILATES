"""
Architecture contracts for coupler ownership boundaries.

These tests document and enforce the ownership split used by the workflow:

1. Stage/step assembly should not call coupler methods directly.
2. Coupler reads for input resolution should go through
   ``pilates/workflows/input_resolution.py``.
3. Coupler writes should go through ``pilates/utils/coupler_helpers.py``.
4. ``pilates/workflows/orchestration.py`` may inspect coupler keys for
   diagnostics, but should not mutate coupler state directly.

The goal is to prevent drift back to ad hoc coupler access patterns that make
input precedence and cross-step state harder to reason about.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PILATES_ROOT = REPO_ROOT / "pilates"


COUPLER_METHODS = {
    "get",
    "set",
    "set_from_artifact",
    "require",
    "pop",
    "update",
    "keys",
    "view",
}


ALLOWED_DIRECT_CALL_FILES = {
    Path("pilates/runtime/bootstrap.py"),
    Path("pilates/workflows/input_resolution.py"),
    Path("pilates/utils/coupler_helpers.py"),
    Path("pilates/workflows/orchestration.py"),
}


ALLOWED_DIRECT_CALLS_BY_FILE = {
    Path("pilates/runtime/bootstrap.py"): {"set"},
    Path("pilates/workflows/input_resolution.py"): {"get", "view"},
    Path("pilates/utils/coupler_helpers.py"): {"set", "set_from_artifact", "view"},
    Path("pilates/workflows/orchestration.py"): {"keys"},
}


@dataclass(frozen=True)
class CouplerCall:
    path: Path
    method: str
    lineno: int


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        # Ignore generated caches if present.
        if "__pycache__" in path.parts:
            continue
        yield path


def _is_coupler_base(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "coupler"
    if isinstance(node, ast.Attribute):
        # Allow detection of patterns like scenario.coupler.get(...)
        return node.attr == "coupler"
    return False


def _find_direct_coupler_calls(path: Path) -> List[CouplerCall]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: List[CouplerCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            continue
        if fn.attr not in COUPLER_METHODS:
            continue
        if not _is_coupler_base(fn.value):
            continue
        calls.append(
            CouplerCall(
                path=path.relative_to(REPO_ROOT),
                method=fn.attr,
                lineno=node.lineno,
            )
        )
    return calls


def _format_calls(calls: Iterable[CouplerCall]) -> str:
    return "\n".join(
        f"- {call.path}:{call.lineno} uses coupler.{call.method}(...)" for call in calls
    )


def test_direct_coupler_calls_are_limited_to_gateway_modules():
    """
    Only gateway modules may call coupler methods directly.

    This keeps ownership explicit:
    - input resolution reads
    - coupler helper writes
    - orchestration diagnostics
    """
    violations: List[CouplerCall] = []
    for path in _iter_python_files(PILATES_ROOT):
        rel = path.relative_to(REPO_ROOT)
        calls = _find_direct_coupler_calls(path)
        if not calls:
            continue
        if rel not in ALLOWED_DIRECT_CALL_FILES:
            violations.extend(calls)
            continue
        allowed_methods = ALLOWED_DIRECT_CALLS_BY_FILE[rel]
        for call in calls:
            if call.method not in allowed_methods:
                violations.append(call)

    assert not violations, (
        "Direct coupler method calls are restricted to ownership gateway modules.\n"
        + _format_calls(violations)
    )


def test_stage_modules_do_not_call_coupler_methods_directly():
    """
    Stage modules must resolve coupler values via workflow helpers only.

    Stage assembly should stay declarative (`StepRef` + input resolution) and
    avoid imperative coupler manipulation.
    """
    violations: List[CouplerCall] = []
    stages_dir = PILATES_ROOT / "workflows" / "stages"
    for path in _iter_python_files(stages_dir):
        violations.extend(_find_direct_coupler_calls(path))

    assert not violations, (
        "Stage modules must not call coupler methods directly.\n"
        + _format_calls(violations)
    )


def test_step_modules_do_not_call_coupler_methods_directly():
    """
    Step modules should publish/read via shared helpers, not direct calls.

    This avoids duplicated semantics around key migrations, canonicalization,
    and provenance behavior.
    """
    violations: List[CouplerCall] = []
    steps_dir = PILATES_ROOT / "workflows" / "steps"
    for path in _iter_python_files(steps_dir):
        violations.extend(_find_direct_coupler_calls(path))

    assert not violations, (
        "Step modules must not call coupler methods directly.\n"
        + _format_calls(violations)
    )
