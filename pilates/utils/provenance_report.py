from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Optional


def _iter_artifacts(artifacts: Any) -> list[Any]:
    if artifacts is None:
        return []
    if isinstance(artifacts, Mapping):
        return list(artifacts.values())
    return list(artifacts)


def _artifact_key(artifact: Any) -> str:
    for key_attr in ("key", "name", "short_name", "artifact_key"):
        value = getattr(artifact, key_attr, None)
        if value:
            return str(value)
    if isinstance(artifact, Mapping):
        for key_attr in ("key", "name", "short_name", "artifact_key"):
            value = artifact.get(key_attr)
            if value:
                return str(value)
    return "unknown"


def _resolve_via_tracker(tracker: Any, value: str) -> Optional[str]:
    if tracker is None:
        return None
    resolve_uri = getattr(tracker, "resolve_uri", None)
    if not callable(resolve_uri):
        return None
    try:
        return str(resolve_uri(value))
    except Exception:
        return None


def _artifact_path(artifact: Any, tracker: Any = None) -> str:
    container_uri = getattr(artifact, "container_uri", None)
    if not container_uri and isinstance(artifact, Mapping):
        container_uri = artifact.get("container_uri")
    if isinstance(container_uri, str) and "://" in container_uri:
        resolved = _resolve_via_tracker(tracker, container_uri)
        if resolved:
            return resolved

    for path_attr in ("path", "uri", "file_path", "container_uri"):
        value = getattr(artifact, path_attr, None)
        if value:
            value_str = str(value)
            if "://" in value_str:
                resolved = _resolve_via_tracker(tracker, value_str)
                if resolved:
                    return resolved
            return value_str
    if isinstance(artifact, Mapping):
        for path_attr in ("path", "uri", "file_path", "container_uri"):
            value = artifact.get(path_attr)
            if value:
                value_str = str(value)
                if "://" in value_str:
                    resolved = _resolve_via_tracker(tracker, value_str)
                    if resolved:
                        return resolved
                return value_str
    return ""


def _node_id(prefix: str, value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    return f"{prefix}_{token[:96]}"


def _step_name(step: Any) -> str:
    if not isinstance(step, Mapping):
        return "unknown_step"
    for key in ("name", "run_name", "model"):
        value = step.get(key)
        if value:
            return str(value)
    return "unknown_step"


def _summarize_artifacts(artifacts: list[Any], tracker: Any = None) -> list[str]:
    rows = []
    for artifact in artifacts:
        key = _artifact_key(artifact)
        path = _artifact_path(artifact, tracker=tracker)
        if path:
            rows.append(f"`{key}` -> `{path}`")
        else:
            rows.append(f"`{key}`")
    return rows


def build_provenance_report(tracker: Any, run_id: str) -> str:
    """
    Build a Markdown provenance report for a run, including a Mermaid DAG.
    """
    run = tracker.get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    steps = (run.meta or {}).get("steps", []) if getattr(run, "meta", None) else []
    lines = [
        "# Provenance Report",
        "",
        "## Scenario",
        f"- `run_id`: `{getattr(run, 'id', run_id)}`",
        f"- `status`: `{getattr(run, 'status', 'unknown')}`",
        f"- `model`: `{getattr(run, 'model_name', 'unknown')}`",
        f"- `step_count`: `{len(steps)}`",
        "",
        "## Step Chain",
    ]

    nodes: dict[str, str] = {}
    edges: set[tuple[str, str]] = set()
    step_nodes: list[str] = []

    for idx, step in enumerate(steps, start=1):
        step_id = step.get("id") if isinstance(step, Mapping) else None
        step_name = _step_name(step)
        step_node = _node_id("step", f"{idx}_{step_name}")
        step_nodes.append(step_node)
        nodes[step_node] = f"{idx}. {step_name}"
        lines.append(f"{idx}. `{step_name}` (`{step_id}`)")

        if not step_id:
            lines.append("- inputs: 0")
            lines.append("- outputs: 0")
            continue

        step_artifacts = tracker.get_artifacts_for_run(step_id)
        inputs = _iter_artifacts(getattr(step_artifacts, "inputs", None))
        outputs = _iter_artifacts(getattr(step_artifacts, "outputs", None))

        lines.append(f"- inputs: {len(inputs)}")
        for row in _summarize_artifacts(inputs, tracker=tracker):
            lines.append(f"  - {row}")
        lines.append(f"- outputs: {len(outputs)}")
        for row in _summarize_artifacts(outputs, tracker=tracker):
            lines.append(f"  - {row}")

        for artifact in inputs:
            artifact_key = _artifact_key(artifact)
            artifact_path = _artifact_path(artifact, tracker=tracker)
            artifact_name = (
                f"{artifact_key}\\n{Path(artifact_path).name}"
                if artifact_path
                else artifact_key
            )
            artifact_node = _node_id("artifact", f"in_{artifact_key}_{artifact_path}")
            nodes[artifact_node] = artifact_name
            edges.add((artifact_node, step_node))

        for artifact in outputs:
            artifact_key = _artifact_key(artifact)
            artifact_path = _artifact_path(artifact, tracker=tracker)
            artifact_name = (
                f"{artifact_key}\\n{Path(artifact_path).name}"
                if artifact_path
                else artifact_key
            )
            artifact_node = _node_id("artifact", f"out_{artifact_key}_{artifact_path}")
            nodes[artifact_node] = artifact_name
            edges.add((step_node, artifact_node))

    for idx in range(1, len(step_nodes)):
        edges.add((step_nodes[idx - 1], step_nodes[idx]))

    lines.extend(
        [
            "",
            "## Input/Output DAG",
            "```mermaid",
            "graph TD",
        ]
    )
    for node_id, label in nodes.items():
        lines.append(f'    {node_id}["{label}"]')
    for src, dst in sorted(edges):
        lines.append(f"    {src} --> {dst}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_provenance_report(
    tracker: Any,
    run_id: str,
    output_path: Path,
) -> str:
    """
    Render and write a provenance report to disk.
    """
    report = build_provenance_report(tracker, run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report
