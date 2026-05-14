from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable


_REQUIRED_TRACKER_METHODS = (
    "find_matching_run",
    "register_run_output_recovery_copies",
)
_REQUIRED_H5_POLICY_KWARGS = (
    "container_recovery_unit",
    "child_recovery_policy",
    "representation_policy",
)


@dataclass(frozen=True)
class ConsistCapabilityReport:
    ok: bool
    missing: tuple[str, ...]

    def format_error(self) -> str:
        details = ", ".join(self.missing) if self.missing else "unknown"
        return (
            "PILATES requires a Consist build with semantic run matching and "
            "H5 recovery-policy support. Missing capability/capabilities: "
            f"{details}. Use the Consist feature release that includes "
            "Tracker.find_matching_run(...), "
            "Tracker.register_run_output_recovery_copies(...), and H5 "
            "container recovery-policy kwargs, or install the local editable "
            "Consist checkout that provides them."
        )


def _callable_attr(obj: Any, name: str) -> Callable[..., Any] | None:
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _signature_accepts_kwargs(
    func: Callable[..., Any],
    names: Iterable[str],
) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    accepts_arbitrary_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if accepts_arbitrary_kwargs:
        return True
    return all(name in parameters for name in names)


def check_consist_runtime_capabilities(tracker: Any) -> ConsistCapabilityReport:
    missing: list[str] = []
    for method_name in _REQUIRED_TRACKER_METHODS:
        if _callable_attr(tracker, method_name) is None:
            missing.append(f"tracker.{method_name}")

    log_h5_container = _callable_attr(tracker, "log_h5_container")
    if log_h5_container is None:
        missing.append("tracker.log_h5_container")
    elif not _signature_accepts_kwargs(log_h5_container, _REQUIRED_H5_POLICY_KWARGS):
        missing.append(
            "tracker.log_h5_container(" + ", ".join(_REQUIRED_H5_POLICY_KWARGS) + ")"
        )

    return ConsistCapabilityReport(ok=not missing, missing=tuple(missing))


def require_consist_runtime_capabilities(tracker: Any) -> None:
    report = check_consist_runtime_capabilities(tracker)
    if not report.ok:
        raise RuntimeError(report.format_error())
