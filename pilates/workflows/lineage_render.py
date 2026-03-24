from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, List

from pilates.workflows.planning import StaticExecutionPlan


def render_plan_json(plan: StaticExecutionPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True)


def _mermaid_label(label: str) -> str:
    return json.dumps(label.replace("\n", "<br/>"))


def _filtered_plan_for_render(
    plan: StaticExecutionPlan,
    *,
    hide_terminal_artifacts: bool = False,
) -> StaticExecutionPlan:
    if not hide_terminal_artifacts:
        return plan

    consumed_artifact_ids = {
        edge.source
        for edge in plan.edges
        if edge.kind == "consumes"
    }
    kept_artifacts = [
        artifact for artifact in plan.artifacts if artifact.id in consumed_artifact_ids
    ]
    kept_node_ids = {step.id for step in plan.step_runs} | {
        artifact.id for artifact in kept_artifacts
    }
    kept_edges = [
        edge
        for edge in plan.edges
        if edge.source in kept_node_ids and edge.target in kept_node_ids
    ]
    return replace(
        plan,
        artifacts=kept_artifacts,
        edges=kept_edges,
    )


def render_plan_mermaid(
    plan: StaticExecutionPlan,
    *,
    hide_terminal_artifacts: bool = False,
) -> str:
    plan = _filtered_plan_for_render(
        plan,
        hide_terminal_artifacts=hide_terminal_artifacts,
    )
    lines: List[str] = [
        "flowchart TD",
        "classDef step fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.5px,color:#0f172a;",
        "classDef artifact fill:#ecfccb,stroke:#65a30d,stroke-width:1.2px,color:#1f2937;",
        "classDef external fill:#fef3c7,stroke:#d97706,stroke-dasharray: 4 2,color:#78350f;",
        "classDef optional fill:#f3e8ff,stroke:#7e22ce,stroke-dasharray: 3 2,color:#4c1d95;",
        "classDef gap fill:#fee2e2,stroke:#dc2626,stroke-dasharray: 3 2,color:#7f1d1d;",
    ]

    for step in plan.step_runs:
        lines.append("%s[%s]" % (step.id, _mermaid_label(step.label)))

    for artifact in plan.artifacts:
        lines.append("%s([%s])" % (artifact.id, _mermaid_label(artifact.label)))

    gap_node_ids: Dict[str, str] = {}
    for gap in plan.contract_gaps:
        gap_node_id = "gapnode_%s" % gap.id
        gap_node_ids[gap.id] = gap_node_id
        lines.append("%s[%s]" % (gap_node_id, _mermaid_label(gap.message)))

    for edge in plan.edges:
        if edge.kind == "depends_on":
            lines.append("%s -.-> %s" % (edge.source, edge.target))
            continue
        if edge.optional:
            lines.append("%s -.-> %s" % (edge.source, edge.target))
            continue
        lines.append("%s --> %s" % (edge.source, edge.target))

    for gap in plan.contract_gaps:
        gap_node_id = gap_node_ids[gap.id]
        lines.append("%s -.-> %s" % (gap_node_id, gap.step_run_id))

    if plan.step_runs:
        lines.append(
            "class %s step;"
            % ",".join(step.id for step in plan.step_runs)
        )
    artifact_classes: List[str] = []
    external_classes: List[str] = []
    optional_classes: List[str] = []
    for artifact in plan.artifacts:
        if artifact.external:
            external_classes.append(artifact.id)
        elif artifact.optional:
            optional_classes.append(artifact.id)
        else:
            artifact_classes.append(artifact.id)
    if artifact_classes:
        lines.append("class %s artifact;" % ",".join(artifact_classes))
    if external_classes:
        lines.append("class %s external;" % ",".join(external_classes))
    if optional_classes:
        lines.append("class %s optional;" % ",".join(optional_classes))
    if gap_node_ids:
        lines.append("class %s gap;" % ",".join(gap_node_ids.values()))

    return "\n".join(lines)


def render_plan_html(
    plan: StaticExecutionPlan,
    *,
    hide_terminal_artifacts: bool = False,
) -> str:
    original_artifact_count = len(plan.artifacts)
    plan = _filtered_plan_for_render(
        plan,
        hide_terminal_artifacts=hide_terminal_artifacts,
    )
    summary_rows = [
        ("Config", plan.config_path or "(in-memory settings)"),
        ("Years", str(len(plan.metadata.get("years", [])))),
        ("Step Runs", str(len(plan.step_runs))),
        ("Artifacts", str(len(plan.artifacts))),
        ("Edges", str(len(plan.edges))),
        ("Contract Gaps", str(len(plan.contract_gaps))),
        ("Full Skim Schedule", str(plan.metadata.get("full_skim_schedule", "unknown"))),
        (
            "Render Filter",
            "hide terminal artifacts"
            if hide_terminal_artifacts
            else "show all artifacts",
        ),
    ]
    summary_html = "\n".join(
        "<tr><th>%s</th><td>%s</td></tr>" % (_html_escape(label), _html_escape(value))
        for label, value in summary_rows
    )
    assumptions_html = "\n".join(
        "<li>%s</li>" % _html_escape(item) for item in plan.assumptions
    )
    gaps_html = "\n".join(
        (
            '<li><strong>%s</strong> <code>%s</code>: %s</li>'
            % (
                _html_escape(gap.kind),
                _html_escape(_step_name_for_gap(plan, gap.step_run_id)),
                _html_escape(gap.message),
            )
        )
        for gap in plan.contract_gaps[:50]
    )
    if not gaps_html:
        gaps_html = "<li>No contract gaps reported.</li>"

    step_by_id = {step.id: step for step in plan.step_runs}
    consumed_by_artifact_id: Dict[str, List[str]] = {}
    for edge in plan.edges:
        if edge.kind != "consumes":
            continue
        consumed_by_artifact_id.setdefault(edge.source, []).append(edge.target)

    def _stage_lane(stage_name: str) -> int:
        lane_by_stage = {
            "land_use": 0,
            "vehicle_ownership_model": 1,
            "activity_demand": 2,
            "traffic_assignment": 3,
            "postprocessing": 4,
        }
        return lane_by_stage.get(stage_name, 5)

    def _step_column(step: Any) -> tuple:
        year = step.year if step.year is not None else -1
        iteration = step.iteration if step.iteration is not None else -1
        atlas_year = step.atlas_year if step.atlas_year is not None else -1
        lane = _stage_lane(step.stage_name)
        phase_rank = {
            "preprocess": 0,
            "compile": 1,
            "run": 2,
            "postprocess": 3,
        }.get(step.phase, 4)
        return (year, iteration, lane, atlas_year, phase_rank, step.sequence)

    ordered_steps = sorted(plan.step_runs, key=_step_column)
    column_by_step_id = {
        step.id: column_index for column_index, step in enumerate(ordered_steps)
    }
    lane_spacing = 270
    step_positions: Dict[str, Dict[str, float]] = {}
    lane_offsets: Dict[int, int] = {}
    for step in plan.step_runs:
        lane = _stage_lane(step.stage_name)
        lane_offsets.setdefault(lane, 0)
        step_positions[step.id] = {
            "x": 180 + column_by_step_id[step.id] * 190,
            "y": 140 + lane * lane_spacing + lane_offsets[lane] * 6,
        }
        lane_offsets[lane] += 1

    produced_artifacts_by_step_id: Dict[str, List[Any]] = {}
    for artifact in plan.artifacts:
        if artifact.producer_step_run_id is not None:
            produced_artifacts_by_step_id.setdefault(
                artifact.producer_step_run_id, []
            ).append(artifact)

    artifact_positions: Dict[str, Dict[str, float]] = {}
    for step_id, artifacts in produced_artifacts_by_step_id.items():
        base = step_positions.get(step_id)
        if base is None:
            continue
        for index, artifact in enumerate(artifacts):
            band = index % 6
            stack = index // 6
            direction = -1 if band < 3 else 1
            offset_band = band % 3
            artifact_positions[artifact.id] = {
                "x": base["x"] + 120 + stack * 72,
                "y": base["y"] + direction * (80 + offset_band * 46),
            }

    external_index = 0
    for artifact in plan.artifacts:
        if artifact.id in artifact_positions:
            continue
        consumer_ids = consumed_by_artifact_id.get(artifact.id, [])
        if consumer_ids:
            consumer_positions = [
                step_positions[consumer_id]
                for consumer_id in consumer_ids
                if consumer_id in step_positions
            ]
            if consumer_positions:
                anchor_x = min(item["x"] for item in consumer_positions)
                anchor_y = sum(item["y"] for item in consumer_positions) / len(
                    consumer_positions
                )
            else:
                anchor_x = 160
                anchor_y = 180
        else:
            anchor_x = 160
            anchor_y = 180
        artifact_positions[artifact.id] = {
            "x": anchor_x - 140 - (external_index // 8) * 48,
            "y": anchor_y + ((external_index % 8) - 4) * 42,
        }
        external_index += 1

    positions_json = json.dumps(
        {
            **step_positions,
            **artifact_positions,
        },
        indent=2,
    )

    lane_guides = [
        ("Land Use", 140 + _stage_lane("land_use") * lane_spacing),
        ("Vehicle Ownership", 140 + _stage_lane("vehicle_ownership_model") * lane_spacing),
        ("Activity Demand", 140 + _stage_lane("activity_demand") * lane_spacing),
        ("Traffic Assignment", 140 + _stage_lane("traffic_assignment") * lane_spacing),
        ("Postprocessing", 140 + _stage_lane("postprocessing") * lane_spacing),
    ]
    lane_guides_html = "\n".join(
        '<div class="lane-guide" style="top:%spx;">%s</div>'
        % (int(y - 18), _html_escape(label))
        for label, y in lane_guides
    )

    elements = []
    for step in plan.step_runs:
        elements.append(
            {
                "data": {
                    "id": step.id,
                    "label": step.label,
                    "detail": step.label.replace("\n", " | "),
                    "kind": "step",
                    "step_name": step.step_name,
                    "stage_name": step.stage_name,
                    "phase": step.phase,
                    "optional": bool(step.optional),
                    "locked": True,
                }
            }
        )

    for artifact in plan.artifacts:
        artifact_kind = "external_artifact" if artifact.external else "artifact"
        elements.append(
            {
                "data": {
                    "id": artifact.id,
                    "label": artifact.label,
                    "detail": artifact.label.replace("\n", " | "),
                    "kind": artifact_kind,
                    "instance_key": artifact.instance_key,
                    "artifact_key": artifact.artifact_key,
                    "canonical_key": artifact.canonical_key,
                    "family": artifact.family,
                    "optional": bool(artifact.optional),
                    "dynamic": bool(artifact.dynamic),
                    "locked": False,
                }
            }
        )

    for edge in plan.edges:
        label = edge.kind
        if edge.artifact_key:
            label = "%s: %s" % (edge.kind, edge.artifact_key)
        elements.append(
            {
                "data": {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "label": label,
                    "kind": edge.kind,
                    "artifact_key": edge.artifact_key,
                    "optional": bool(edge.optional),
                    "dynamic": bool(edge.dynamic),
                }
            }
        )

    selection_index = {
        step.id: {
            "title": step.step_name,
            "body": {
                "kind": "step",
                "stage_name": step.stage_name,
                "phase": step.phase,
                "year": step.year,
                "forecast_year": step.forecast_year,
                "iteration": step.iteration,
                "atlas_year": step.atlas_year,
                "optional": step.optional,
                "depends_on": step.depends_on,
                "upstream_step_inputs": step.upstream_step_inputs,
            },
        }
        for step in plan.step_runs
    }
    selection_index.update(
        {
            artifact.id: {
                "title": artifact.artifact_key,
                "body": {
                    "kind": "external_artifact" if artifact.external else "artifact",
                    "instance_key": artifact.instance_key,
                    "canonical_key": artifact.canonical_key,
                    "year": artifact.year,
                    "forecast_year": artifact.forecast_year,
                    "iteration": artifact.iteration,
                    "atlas_year": artifact.atlas_year,
                    "optional": artifact.optional,
                    "dynamic": artifact.dynamic,
                    "family": artifact.family,
                    "producer_step_run_id": artifact.producer_step_run_id,
                    "path_role": artifact.path_role,
                    "resolved_path_hint": artifact.resolved_path_hint,
                    "path_notes": artifact.path_notes,
                },
            }
            for artifact in plan.artifacts
        }
    )
    selection_index.update(
        {
            edge.id: {
                "title": edge.kind,
                "body": {
                    "kind": "edge",
                    "source": edge.source,
                    "target": edge.target,
                    "artifact_key": edge.artifact_key,
                    "optional": edge.optional,
                    "dynamic": edge.dynamic,
                    "family": edge.family,
                },
            }
            for edge in plan.edges
        }
    )

    elements_json = json.dumps(elements, indent=2)
    selection_json = json.dumps(selection_index, indent=2)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PILATES Workflow Lineage</title>
  <style>
    :root {{
      --bg: #f7f4ea;
      --ink: #1c1917;
      --panel: #fffdf8;
      --line: #d6d3d1;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --warn: #b91c1c;
      --warn-soft: #fee2e2;
      --muted: #57534e;
      --shadow: 0 10px 30px rgba(28, 25, 23, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.08), transparent 25%),
        var(--bg);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .hero {{
      margin-bottom: 24px;
      padding: 28px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(250,250,249,0.88));
      box-shadow: var(--shadow);
    }}
    #app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 20px;
      min-height: 78vh;
    }}
    .panel {{
      padding: 18px 20px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    #cy {{
      width: 100%;
      min-height: 78vh;
      border: 1px solid var(--line);
      background:
        radial-gradient(circle at 20% 20%, rgba(15, 118, 110, 0.04), transparent 22%),
        radial-gradient(circle at 80% 30%, rgba(217, 119, 6, 0.04), transparent 18%),
        #fff;
      box-shadow: var(--shadow);
    }}
    #graphShell {{
      position: relative;
    }}
    #laneGuides {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: 0;
    }}
    .lane-guide {{
      position: absolute;
      left: 12px;
      right: 12px;
      height: 0;
      border-top: 1px dashed rgba(120, 113, 108, 0.28);
      color: rgba(87, 83, 78, 0.72);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .lane-guide::before {{
      content: attr(style-label);
    }}
    #sidebar {{
      display: flex;
      flex-direction: column;
      gap: 20px;
      min-width: 0;
    }}
    #toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    button {{
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 7px 12px;
      background: #fff;
      cursor: pointer;
      font-size: 12px;
    }}
    button:hover {{
      background: #f5f5f4;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      width: 42%;
      color: var(--muted);
      font-weight: 600;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li + li {{
      margin-top: 8px;
    }}
    code, pre {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      padding: 14px;
      border: 1px solid var(--line);
      background: #fafaf9;
      overflow: auto;
      margin: 0;
    }}
    .gaps {{
      background: linear-gradient(180deg, rgba(254, 242, 242, 0.95), rgba(255, 255, 255, 0.9));
      border-color: #fecaca;
    }}
    .gaps h2 {{
      color: var(--warn);
    }}
    .note {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 0 0 12px;
    }}
    @media (max-width: 980px) {{
      #app {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>PILATES Workflow Lineage</h1>
      <p>Static execution plan generated from workflow contracts. Artifact edges come from catalog metadata; underdeclared areas are surfaced as contract gaps instead of guessed.</p>
    </section>
    <section id="app">
      <section class="panel">
        <div id="toolbar">
          <button id="fitBtn" type="button">Fit</button>
          <button id="layoutBtn" type="button">Re-layout</button>
          <button id="orderedBtn" type="button">Ordered Layout</button>
          <button id="relaxArtifactsBtn" type="button">Relax Artifacts</button>
          <button id="resetBtn" type="button">Reset Selection</button>
        </div>
        <p class="hint">Step boxes start in fixed year/iteration/stage columns. Artifact boxes can be relaxed around them while step positions stay anchored. Scroll to zoom, drag the background to pan, and drag nodes to reposition boxes.</p>
        <div id="graphShell">
          <div id="laneGuides">
            {lane_guides_html}
          </div>
          <div id="cy"></div>
        </div>
      </section>
      <aside id="sidebar">
        <section class="panel">
          <h2>Summary</h2>
          <table>
            {summary_html}
          </table>
          <div class="note">Rendered artifacts: {len(plan.artifacts)} of {original_artifact_count}.</div>
        </section>
        <section class="panel">
          <h2>Selection</h2>
          <pre id="selectionInfo">Nothing selected.</pre>
        </section>
        <section class="panel" style="margin-top:20px;">
          <h2>Assumptions</h2>
          <ul>
            {assumptions_html}
          </ul>
        </section>
        <section class="panel gaps" style="margin-top:20px;">
          <h2>Contract Gaps</h2>
          <ul>
            {gaps_html}
          </ul>
          <div class="note">Showing up to the first 50 gaps.</div>
        </section>
      </aside>
    </section>
  </main>
  <script type="module">
    import cytoscape from "https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/+esm";

    const elements = {elements_json};
    const selectionIndex = {selection_json};
    const presetPositions = {positions_json};

    function presetPosition(node) {{
      return presetPositions[node.id()] || {{ x: 100, y: 100 }};
    }}

    function lockStepNodes() {{
      cy.nodes('[kind = "step"]').lock();
    }}

    function unlockAllNodes() {{
      cy.nodes().unlock();
    }}

    function unlockStepNodes() {{
      cy.nodes('[kind = "step"]').unlock();
    }}

    const cy = cytoscape({{
      container: document.getElementById("cy"),
      elements,
      wheelSensitivity: 0.18,
      style: [
        {{
          selector: 'node[kind = "step"]',
          style: {{
            'shape': 'round-rectangle',
            'background-color': '#dbeafe',
            'border-color': '#1d4ed8',
            'border-width': 1.5,
            'color': '#0f172a',
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '170px',
            'font-size': 12,
            'font-weight': 600,
            'padding': '12px',
            'width': 'label',
            'height': 'label'
          }}
        }},
        {{
          selector: 'node[kind = "artifact"]',
          style: {{
            'shape': 'round-rectangle',
            'background-color': '#ecfccb',
            'border-color': '#65a30d',
            'border-width': 1.2,
            'color': '#1f2937',
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '160px',
            'font-size': 10,
            'padding': '9px',
            'width': 'label',
            'height': 'label'
          }}
        }},
        {{
          selector: 'node[kind = "external_artifact"]',
          style: {{
            'shape': 'round-rectangle',
            'background-color': '#fef3c7',
            'border-color': '#d97706',
            'border-width': 1.2,
            'border-style': 'dashed',
            'color': '#78350f',
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '160px',
            'font-size': 10,
            'padding': '9px',
            'width': 'label',
            'height': 'label'
          }}
        }},
        {{
          selector: 'node[optional = 1]',
          style: {{
            'border-style': 'dashed'
          }}
        }},
        {{
          selector: 'edge',
          style: {{
            'curve-style': 'bezier',
            'width': 1.2,
            'line-color': '#78716c',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#78716c',
            'arrow-scale': 0.8,
            'label': 'data(label)',
            'font-size': 8,
            'text-background-color': '#ffffff',
            'text-background-opacity': 0.9,
            'text-background-padding': 1,
            'text-rotation': 'autorotate',
            'color': '#57534e'
          }}
        }},
        {{
          selector: 'edge[kind = "depends_on"]',
          style: {{
            'line-style': 'dashed',
            'target-arrow-shape': 'chevron',
            'line-color': '#0f766e',
            'target-arrow-color': '#0f766e',
            'color': '#0f766e'
          }}
        }},
        {{
          selector: 'edge[optional = 1]',
          style: {{
            'line-style': 'dashed'
          }}
        }},
        {{
          selector: ':selected',
          style: {{
            'overlay-opacity': 0.1,
            'overlay-color': '#f59e0b',
            'border-color': '#f59e0b',
            'line-color': '#f59e0b',
            'target-arrow-color': '#f59e0b'
          }}
        }}
      ],
      layout: {{
        name: 'preset',
        positions: presetPosition,
        fit: true,
        padding: 60
      }}
    }});
    unlockAllNodes();

    function updateSelectionInfo() {{
      const selected = cy.$(':selected');
      const el = document.getElementById('selectionInfo');
      if (selected.length === 0) {{
        el.textContent = 'Nothing selected.';
        return;
      }}
      const data = selected[0].data();
      const payload = selectionIndex[data.id];
      el.textContent = JSON.stringify(payload || data, null, 2);
    }}

    cy.on('select unselect tap', updateSelectionInfo);
    updateSelectionInfo();

    document.getElementById('fitBtn').addEventListener('click', function () {{
      cy.fit(undefined, 40);
    }});
    document.getElementById('layoutBtn').addEventListener('click', function () {{
      unlockAllNodes();
      cy.layout({{
        name: 'cose',
        animate: 'end',
        fit: true,
        padding: 40,
        nodeRepulsion: 12000,
        idealEdgeLength: 135
      }}).run();
    }});
    document.getElementById('orderedBtn').addEventListener('click', function () {{
      unlockAllNodes();
      cy.layout({{
        name: 'preset',
        positions: presetPosition,
        fit: true,
        padding: 60,
        animate: true
      }}).run();
      unlockAllNodes();
    }});
    document.getElementById('relaxArtifactsBtn').addEventListener('click', function () {{
      cy.nodes('[kind = "step"]').positions(function (node) {{
        return presetPosition(node);
      }});
      lockStepNodes();
      cy.nodes('[kind != "step"]').unlock();
      const layout = cy.layout({{
        name: 'cose',
        animate: 'end',
        fit: true,
        padding: 40,
        nodeRepulsion: 36000,
        idealEdgeLength: 240,
        edgeElasticity: 40,
        nestingFactor: 0.8,
        gravity: 0.12,
        componentSpacing: 160
      }});
      layout.on('layoutstop', function () {{
        unlockStepNodes();
      }});
      layout.run();
    }});
    document.getElementById('resetBtn').addEventListener('click', function () {{
      cy.$(':selected').unselect();
      updateSelectionInfo();
    }});
  </script>
</body>
</html>
"""


def _step_name_for_gap(plan: StaticExecutionPlan, step_run_id: str) -> str:
    for step in plan.step_runs:
        if step.id == step_run_id:
            return step.label.replace("\n", " | ")
    return step_run_id


def _html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
