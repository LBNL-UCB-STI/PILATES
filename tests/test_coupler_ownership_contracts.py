"""
Architecture contracts for coupler ownership boundaries.

These tests document and enforce the ownership split used by the workflow:

1. Native step resolvers select inputs through their committed resolver paths,
   not direct coupler calls in stage assembly.
2. ``pilates/workflows/coupler_namespace.py`` owns direct namespace-aware
   coupler reads.
3. ``pilates/utils/coupler_helpers.py`` owns direct coupler publication and
   materialization helpers.

The goal is to prevent drift back to ad hoc coupler access patterns that make
input precedence and cross-step state harder to reason about.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pilates.workflows.coupler_namespace import (
    coupler_storage_keys,
    coupler_storage_value,
)


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
    Path("pilates/workflows/coupler_namespace.py"),
    Path("pilates/utils/coupler_helpers.py"),
}


ALLOWED_DIRECT_CALLS_BY_FILE = {
    Path("pilates/runtime/bootstrap.py"): {"set"},
    Path("pilates/workflows/coupler_namespace.py"): {"get", "keys", "view"},
    Path("pilates/utils/coupler_helpers.py"): {"set", "set_from_artifact", "view"},
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
    - native resolvers select their inputs through committed resolver paths
    - the namespace helper owns direct reads
    - the coupler helper owns publication and materialization writes
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

    Stage assembly should stay declarative and use native resolver results rather
    than imperative coupler manipulation.
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


def test_coupler_storage_keys_preserves_raw_storage_order() -> None:
    class Coupler:
        def keys(self) -> tuple[str, ...]:
            return (
                "beam/events_parquet_2030_1",
                "events_parquet_2030_1",
                "raw_od_skims_2030_1",
            )

    assert coupler_storage_keys(Coupler()) == (
        "beam/events_parquet_2030_1",
        "events_parquet_2030_1",
        "raw_od_skims_2030_1",
    )


def test_coupler_storage_value_uses_only_the_global_storage_key() -> None:
    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {"beam_plans_in": "global"}.get(key, default)

        def view(self, namespace: str) -> object:
            assert namespace == "beam"
            return type(
                "BeamView",
                (),
                {"get": lambda _self, key, default=None: "view"},
            )()

    assert coupler_storage_value(Coupler(), "beam_plans_in") == "global"
