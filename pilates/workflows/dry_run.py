"""Dry-run tracing of ``run_pilates()`` for zero-I/O execution planning and lineage rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from pilates.config.models import PilatesConfig
from pilates.workflows.artifact_keys import (
    BEAM_FULL_SKIMS,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.state import WorkflowState
from pilates.workspace import Workspace


@dataclass(frozen=True, slots=True)
class SymbolicArtifact:
    """Placeholder Artifact/Path returned by steps during dry-run planning."""

    key: str
    producer_step: str | None = None
    path: Path = Path("<symbolic>")

    def __fspath__(self) -> str:
        return str(self.path)


@dataclass(frozen=True, slots=True)
class PlannedStepRun:
    """One recorded step invocation in a dry-run execution plan."""

    step_id: str
    step_name: str
    year: int
    iteration: int | None
    inputs: dict[str, str]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannedEdge:
    """One dataflow edge between a producer step (or bootstrap) and a consumer step."""

    source_step: str
    target_step: str
    artifact_key: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Complete execution plan produced by tracing ``run_pilates()`` in dry-run mode."""

    steps: tuple[PlannedStepRun, ...]
    edges: tuple[PlannedEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SymbolicOutputs(dict[str, SymbolicArtifact]):
    """Symbolic outputs mapping supporting both ``dict.get()`` and typed attribute access."""

    @property
    def processed_outputs(self) -> SymbolicOutputs:
        return self

    @property
    def usim_datastore_h5(self) -> SymbolicArtifact | None:
        return self.get(USIM_DATASTORE_H5)

    @property
    def zarr_skims(self) -> SymbolicArtifact | None:
        return self.get(ZARR_SKIMS)

    @property
    def final_skims_omx(self) -> SymbolicArtifact | None:
        return self.get(FINAL_SKIMS_OMX)

    @property
    def full_skims(self) -> SymbolicArtifact | None:
        return self.get(BEAM_FULL_SKIMS)


@dataclass(frozen=True, slots=True)
class SymbolicRun:
    """Minimal run metadata stub for dry-run ``RunResult.run``."""

    id: str


@dataclass(frozen=True, slots=True)
class SymbolicRunResult:
    """Minimal ``RunResult`` stub returned by ``execute_step()`` in dry-run mode."""

    run: SymbolicRun
    outputs: dict[str, SymbolicArtifact]
    cache_hit: bool = False


@dataclass(slots=True)
class SymbolicScenario:
    """Minimal scenario stub for dry-run ``run_pilates()``."""

    coupler: None = None


@dataclass(slots=True)
class DryRunRecorder:
    """Collects ``PlannedStepRun`` and ``PlannedEdge`` records during dry-run tracing."""

    steps: list[PlannedStepRun] = field(default_factory=list)
    edges: list[PlannedEdge] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)

    def _allocate_step_id(
        self,
        *,
        step_name: str,
        year: int,
        iteration: int | None,
    ) -> str:
        base = (
            f"{step_name}_y{year}_i{iteration}"
            if iteration is not None
            else f"{step_name}_y{year}"
        )
        candidate = base
        suffix = 2
        while candidate in self._seen_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self._seen_ids.add(candidate)
        return candidate

    def record_step(
        self,
        *,
        step_name: str,
        year: int,
        iteration: int | None,
        inputs: Mapping[str, object],
        output_keys: Iterable[str],
    ) -> str:
        step_id = self._allocate_step_id(
            step_name=step_name,
            year=year,
            iteration=iteration,
        )
        mapped_inputs: dict[str, str] = {}
        for role, value in inputs.items():
            if isinstance(value, SymbolicArtifact):
                producer = value.producer_step or "bootstrap"
            else:
                producer = "bootstrap"
            mapped_inputs[role] = producer
            self.edges.append(
                PlannedEdge(
                    source_step=producer,
                    target_step=step_id,
                    artifact_key=role,
                )
            )
        self.steps.append(
            PlannedStepRun(
                step_id=step_id,
                step_name=step_name,
                year=year,
                iteration=iteration,
                inputs=mapped_inputs,
                outputs=tuple(output_keys),
            )
        )
        return step_id

    def to_plan(self) -> ExecutionPlan:
        return ExecutionPlan(
            steps=tuple(self.steps),
            edges=tuple(self.edges),
        )


def plan_pilates(
    settings: PilatesConfig,
    *,
    include_postprocessing: bool = True,
) -> ExecutionPlan:
    """Trace ``run_pilates()`` in zero-I/O dry-run mode and return its ``ExecutionPlan``."""
    from pilates.workflows.orchestrator import run_pilates

    active_settings = (
        settings.model_copy(deep=True)
        if include_postprocessing or settings.postprocessing is None
        else settings.model_copy(update={"postprocessing": None}, deep=True)
    )
    state = WorkflowState.from_settings(active_settings)
    workspace = Workspace(
        settings=active_settings,
        output_path=active_settings.run.output_directory or ".",
        folder_name="dry_run",
        dry_run=True,
    )
    recorder = DryRunRecorder()
    run_pilates(
        scenario=SymbolicScenario(),  # type: ignore[arg-type]
        settings=active_settings,
        state=state,
        workspace=workspace,
        plan_recorder=recorder,
    )
    return recorder.to_plan()


def render_plan_json(plan: ExecutionPlan) -> str:
    """Render ``ExecutionPlan`` as formatted JSON."""
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True)


def _mermaid_node_id(raw_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", raw_id)


def render_plan_mermaid(
    plan: ExecutionPlan,
    *,
    hide_terminal_artifacts: bool = False,
) -> str:
    """Render ``ExecutionPlan`` as a Mermaid flowchart."""
    lines: list[str] = [
        "flowchart TD",
        "classDef step fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.5px,color:#0f172a;",
        "classDef bootstrap fill:#fef3c7,stroke:#d97706,stroke-dasharray: 4 2,color:#78350f;",
        "classDef artifact fill:#ecfccb,stroke:#65a30d,stroke-width:1.2px,color:#1f2937;",
    ]

    has_bootstrap = any(edge.source_step == "bootstrap" for edge in plan.edges)
    if has_bootstrap:
        lines.append('bootstrap(["bootstrap"])')

    step_node_ids = {
        step.step_id: _mermaid_node_id(step.step_id) for step in plan.steps
    }

    for step in plan.steps:
        node_id = step_node_ids[step.step_id]
        scope = (
            f"y{step.year} / i{step.iteration}"
            if step.iteration is not None
            else f"y{step.year}"
        )
        label = json.dumps(f"{step.step_name}<br/>{scope}")
        lines.append(f"{node_id}[{label}]")

    for edge in plan.edges:
        source_id = (
            "bootstrap"
            if edge.source_step == "bootstrap"
            else step_node_ids.get(edge.source_step, _mermaid_node_id(edge.source_step))
        )
        target_id = step_node_ids.get(
            edge.target_step, _mermaid_node_id(edge.target_step)
        )
        edge_label = json.dumps(edge.artifact_key)
        lines.append(f"{source_id} -->|{edge_label}| {target_id}")

    terminal_nodes: list[str] = []
    if not hide_terminal_artifacts:
        consumed_pairs = {(edge.source_step, edge.artifact_key) for edge in plan.edges}
        term_index = 0
        for step in plan.steps:
            node_id = step_node_ids[step.step_id]
            for key in step.outputs:
                if (step.step_id, key) not in consumed_pairs:
                    term_index += 1
                    term_id = f"term_{term_index}"
                    terminal_nodes.append(term_id)
                    lines.append(f"{term_id}([{json.dumps(key)}])")
                    lines.append(f"{node_id} --> {term_id}")

    if plan.steps:
        lines.append(
            f"class {','.join(step_node_ids[step.step_id] for step in plan.steps)} step;"
        )
    if has_bootstrap:
        lines.append("class bootstrap bootstrap;")
    if terminal_nodes:
        lines.append(f"class {','.join(terminal_nodes)} artifact;")

    return "\n".join(lines)


def render_plan_html(
    plan: ExecutionPlan,
    *,
    hide_terminal_artifacts: bool = False,
) -> str:
    """Render ``ExecutionPlan`` as a self-contained HTML page with Mermaid diagram."""
    mermaid_src = html.escape(
        render_plan_mermaid(plan, hide_terminal_artifacts=hide_terminal_artifacts)
    ).replace("</", "<\\/")
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8" />',
            "  <title>PILATES Execution Plan</title>",
            '  <script type="module">',
            "    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';",
            "    mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });",
            "  </script>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; margin: 24px; color: #0f172a; }",
            "    .summary { margin-bottom: 16px; font-size: 14px; color: #334155; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>PILATES Execution Plan</h1>",
            f'  <div class="summary">Steps: {len(plan.steps)} | Edges: {len(plan.edges)}</div>',
            f'  <pre class="mermaid">{mermaid_src}</pre>',
            "</body>",
            "</html>",
        ]
    )
