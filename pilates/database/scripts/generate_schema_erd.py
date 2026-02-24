#!/usr/bin/env python3
"""
Generate a Mermaid ER diagram from curated PILATES SQLModel schema classes.

This script reads schema classes from ``pilates.database.schema.registry`` and
extracts columns plus foreign-key relationships from SQLModel ``sa_column``
metadata. It works with the abstract schema stubs checked into PILATES.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Support direct script execution from repo checkout without requiring package install.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pilates.database.schema.registry import get_consist_schemas


@dataclass(frozen=True)
class ColumnInfo:
    table: str
    name: str
    type_name: str


@dataclass(frozen=True)
class RelationshipInfo:
    source_table: str
    source_column: str
    target_table: str
    target_column: str


def _sanitize_identifier(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]", "_", value or "")
    token = token.strip("_") or "col"
    if token[0].isdigit():
        token = f"col_{token}"
    return token


def _sanitize_type(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]", "_", value or "")
    token = token.strip("_") or "TEXT"
    return token.upper()


def _escape_dot_html(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _iter_schema_fields(schema_cls: type) -> Iterable[Tuple[str, object]]:
    fields = getattr(schema_cls, "model_fields", None)
    if isinstance(fields, dict):
        for py_name, field in fields.items():
            yield py_name, field


def _extract_columns_and_relationships(
    schema_classes: Sequence[type],
) -> Tuple[Dict[str, List[ColumnInfo]], List[RelationshipInfo]]:
    columns_by_table: Dict[str, List[ColumnInfo]] = {}
    relationships: List[RelationshipInfo] = []

    for schema_cls in schema_classes:
        table_name = schema_cls.__name__
        table_columns: List[ColumnInfo] = []

        for py_name, field in _iter_schema_fields(schema_cls):
            sa_column = getattr(field, "sa_column", None)
            if sa_column is None:
                continue

            column_name = getattr(sa_column, "name", None) or py_name
            column_type = str(getattr(sa_column, "type", "TEXT"))
            table_columns.append(
                ColumnInfo(
                    table=table_name,
                    name=str(column_name),
                    type_name=column_type,
                )
            )

            for fk in getattr(sa_column, "foreign_keys", set()):
                target_fullname = getattr(fk, "target_fullname", "") or ""
                if "." not in target_fullname:
                    continue
                target_table, target_column = target_fullname.split(".", 1)
                relationships.append(
                    RelationshipInfo(
                        source_table=table_name,
                        source_column=str(column_name),
                        target_table=str(target_table),
                        target_column=str(target_column),
                    )
                )

        # Preserve definition order from model_fields, and dedupe by name.
        seen: Set[str] = set()
        deduped_columns: List[ColumnInfo] = []
        for col in table_columns:
            if col.name in seen:
                continue
            deduped_columns.append(col)
            seen.add(col.name)
        columns_by_table[table_name] = deduped_columns

    return columns_by_table, relationships


def build_mermaid_erd(*, include_columns: bool = True) -> str:
    schema_classes = get_consist_schemas()
    columns_by_table, relationships = _extract_columns_and_relationships(schema_classes)

    entity_names: Set[str] = set(columns_by_table.keys())
    for rel in relationships:
        entity_names.add(rel.source_table)
        entity_names.add(rel.target_table)

    lines: List[str] = ["erDiagram"]

    if include_columns:
        for table_name in sorted(entity_names):
            table_columns = columns_by_table.get(table_name, [])
            lines.append(f"    {table_name} {{")
            if not table_columns:
                lines.append("        TEXT placeholder")
            else:
                for col in table_columns:
                    type_name = _sanitize_type(col.type_name)
                    col_name = _sanitize_identifier(col.name)
                    lines.append(f"        {type_name} {col_name}")
            lines.append("    }")

    unique_relationships = {
        (
            rel.source_table,
            rel.source_column,
            rel.target_table,
            rel.target_column,
        )
        for rel in relationships
    }
    for source_table, source_column, target_table, target_column in sorted(
        unique_relationships
    ):
        label = f"{source_column} -> {target_column}".replace('"', "'")
        lines.append(
            f'    {source_table} }}o--|| {target_table} : "{label}"'
        )

    return "\n".join(lines) + "\n"


def build_graphviz_dot(*, include_columns: bool = True) -> str:
    schema_classes = get_consist_schemas()
    columns_by_table, relationships = _extract_columns_and_relationships(schema_classes)

    entity_names: Set[str] = set(columns_by_table.keys())
    for rel in relationships:
        entity_names.add(rel.source_table)
        entity_names.add(rel.target_table)

    lines: List[str] = [
        "digraph erd {",
        "    graph [rankdir=LR, splines=true, overlap=false];",
        '    node [fontname="Helvetica"];',
        '    edge [fontname="Helvetica"];',
    ]

    for table_name in sorted(entity_names):
        if include_columns:
            table_columns = columns_by_table.get(table_name, [])
            table_title = _escape_dot_html(table_name)
            row_lines = [
                "        <TR><TD BGCOLOR=\"lightgray\"><B>"
                f"{table_title}</B></TD></TR>"
            ]
            if table_columns:
                for col in table_columns:
                    col_name = _escape_dot_html(col.name)
                    col_type = _escape_dot_html(_sanitize_type(col.type_name))
                    row_lines.append(
                        "        <TR><TD ALIGN=\"LEFT\">"
                        f"{col_name} : {col_type}</TD></TR>"
                    )
            else:
                row_lines.append(
                    "        <TR><TD ALIGN=\"LEFT\">placeholder : TEXT</TD></TR>"
                )
            lines.append(f'    "{table_name}" [shape=plain, label=<')
            lines.append(
                "      <TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\">"
            )
            lines.extend(row_lines)
            lines.append("      </TABLE>")
            lines.append("    >];")
        else:
            lines.append(f'    "{table_name}" [shape=box, label="{table_name}"];')

    unique_relationships = {
        (
            rel.source_table,
            rel.source_column,
            rel.target_table,
            rel.target_column,
        )
        for rel in relationships
    }
    for source_table, source_column, target_table, target_column in sorted(
        unique_relationships
    ):
        label = _escape_dot_html(f"{source_column} -> {target_column}")
        lines.append(
            f'    "{source_table}" -> "{target_table}" [label="{label}"];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def build_cytoscape_html(*, include_columns: bool = True) -> str:
    schema_classes = get_consist_schemas()
    columns_by_table, relationships = _extract_columns_and_relationships(schema_classes)

    entity_names: Set[str] = set(columns_by_table.keys())
    for rel in relationships:
        entity_names.add(rel.source_table)
        entity_names.add(rel.target_table)

    nodes = []
    for table_name in sorted(entity_names):
        columns = columns_by_table.get(table_name, [])
        columns_payload = (
            [{"name": col.name, "type": _sanitize_type(col.type_name)} for col in columns]
            if include_columns
            else []
        )
        nodes.append(
            {
                "data": {
                    "id": table_name,
                    "label": table_name,
                    "column_count": len(columns),
                    "columns": columns_payload,
                }
            }
        )

    unique_relationships = {
        (
            rel.source_table,
            rel.source_column,
            rel.target_table,
            rel.target_column,
        )
        for rel in relationships
    }
    edges = []
    for i, (source_table, source_column, target_table, target_column) in enumerate(
        sorted(unique_relationships), start=1
    ):
        edges.append(
            {
                "data": {
                    "id": f"e{i}",
                    "source": source_table,
                    "target": target_table,
                    "label": f"{source_column} -> {target_column}",
                }
            }
        )

    elements_json = json.dumps(nodes + edges, indent=2)
    include_columns_flag = "true" if include_columns else "false"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PILATES Schema ERD</title>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      width: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: #f7f8fa;
      color: #1f2937;
    }}
    #app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      height: 100%;
      width: 100%;
      overflow: hidden;
    }}
    #cy {{
      height: 100%;
      width: 100%;
      min-width: 0;
      overflow: hidden;
      border-right: 1px solid #e5e7eb;
      background: #ffffff;
    }}
    #panel {{
      padding: 12px;
      box-sizing: border-box;
      min-width: 0;
      overflow: auto;
      background: #fafafa;
    }}
    #toolbar {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 6px 10px;
      background: #ffffff;
      cursor: pointer;
      font-size: 12px;
    }}
    button:hover {{
      background: #f1f5f9;
    }}
    .muted {{
      color: #6b7280;
      font-size: 12px;
      line-height: 1.4;
      margin: 0 0 10px;
    }}
    .section-title {{
      margin: 12px 0 6px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #475569;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 8px;
      max-width: 100%;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div id="app">
    <div id="cy"></div>
    <aside id="panel">
      <div id="toolbar">
        <button id="fitBtn" type="button">Fit</button>
        <button id="layoutBtn" type="button">Re-layout</button>
        <button id="resetBtn" type="button">Reset Selection</button>
      </div>
      <p class="muted">
        Scroll to zoom, drag background to pan, drag nodes to reposition.
      </p>
      <div class="section-title">Selection</div>
      <pre id="selectionInfo">Nothing selected.</pre>
      <div class="section-title">Graph</div>
      <pre id="graphInfo"></pre>
    </aside>
  </div>

  <script src="../diagrams/node_modules/cytoscape/dist/cytoscape.min.js"></script>
  <script>
    (function () {{
      if (typeof cytoscape === "undefined") {{
        document.getElementById("selectionInfo").textContent =
          "Failed to load local Cytoscape asset. Ensure docs/diagrams/node_modules is present.";
        return;
      }}

      const includeColumns = {include_columns_flag};
      const elements = {elements_json};
      const nodeCount = elements.filter((e) => e.data && e.data.id && !e.data.source).length;
      const edgeCount = elements.filter((e) => e.data && e.data.source).length;
      document.getElementById("graphInfo").textContent =
        `nodes: ${{nodeCount}}\\nedges: ${{edgeCount}}\\ncolumns shown: ${{includeColumns}}`;

      const cy = cytoscape({{
        container: document.getElementById("cy"),
        elements,
        style: [
          {{
            selector: "node",
            style: {{
              "shape": "round-rectangle",
              "background-color": "#2563eb",
              "label": "data(label)",
              "color": "#ffffff",
              "text-wrap": "wrap",
              "text-max-width": "180px",
              "text-valign": "center",
              "text-halign": "center",
              "font-size": 11,
              "padding": "10px",
              "width": "label",
              "height": "label",
              "border-width": 1,
              "border-color": "#1e40af"
            }}
          }},
          {{
            selector: "edge",
            style: {{
              "curve-style": "bezier",
              "width": 1.2,
              "line-color": "#64748b",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "#64748b",
              "arrow-scale": 0.8,
              "label": "data(label)",
              "font-size": 8,
              "text-rotation": "autorotate",
              "text-background-color": "#ffffff",
              "text-background-opacity": 0.9,
              "text-background-padding": 1
            }}
          }},
          {{
            selector: ":selected",
            style: {{
              "overlay-opacity": 0.08,
              "overlay-color": "#f59e0b",
              "border-color": "#f59e0b",
              "line-color": "#f59e0b",
              "target-arrow-color": "#f59e0b"
            }}
          }}
        ],
        layout: {{
          name: "cose",
          animate: false,
          randomize: true,
          fit: true,
          padding: 30
        }},
        wheelSensitivity: 0.2
      }});

      function updateSelectionInfo() {{
        const selection = cy.$(":selected");
        const el = document.getElementById("selectionInfo");
        if (selection.length === 0) {{
          el.textContent = "Nothing selected.";
          return;
        }}
        const first = selection[0];
        if (first.isNode()) {{
          const cols = first.data("columns") || [];
          let txt = `TABLE: ${{first.data("label")}}\\ncolumns: ${{first.data("column_count")}}`;
          if (includeColumns && cols.length) {{
            txt += "\\n\\n";
            txt += cols.map((c) => `${{c.name}} : ${{c.type}}`).join("\\n");
          }}
          el.textContent = txt;
        }} else {{
          el.textContent = `RELATION: ${{first.data("label")}}\\nsource: ${{first.data("source")}}\\ntarget: ${{first.data("target")}}`;
        }}
      }}

      cy.on("select unselect tap", updateSelectionInfo);
      updateSelectionInfo();

      document.getElementById("fitBtn").addEventListener("click", function () {{
        cy.fit(undefined, 30);
      }});
      document.getElementById("layoutBtn").addEventListener("click", function () {{
        cy.layout({{ name: "cose", animate: "end", fit: true, padding: 30 }}).run();
      }});
      document.getElementById("resetBtn").addEventListener("click", function () {{
        cy.$(":selected").unselect();
        updateSelectionInfo();
      }});
    }})();
  </script>
</body>
</html>
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Mermaid ERD from PILATES curated SQLModel schemas."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output diagram file path. Defaults by format under docs/database/.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("mermaid", "dot", "html"),
        default="mermaid",
        help="Diagram output format.",
    )
    parser.add_argument(
        "--no-columns",
        action="store_true",
        help="Emit only relationships (skip entity column details).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated diagram text to stdout.",
    )
    parser.add_argument(
        "--render",
        choices=("svg", "png", "pdf"),
        default=None,
        help="Render Graphviz output via `dot` (requires --format dot).",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        default=None,
        help="Rendered file path for --render. Defaults beside --output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.format == "mermaid":
        content = build_mermaid_erd(include_columns=not args.no_columns)
        default_output = Path("docs/database/pilates_schema_erd.mmd")
    elif args.format == "dot":
        content = build_graphviz_dot(include_columns=not args.no_columns)
        default_output = Path("docs/database/pilates_schema_erd.dot")
    else:
        content = build_cytoscape_html(include_columns=not args.no_columns)
        default_output = Path("docs/database/pilates_schema_erd.html")

    output_path: Path = args.output or default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    if args.stdout:
        print(content, end="")

    print(f"Wrote {args.format} ERD: {output_path}")

    if args.render:
        if args.format != "dot":
            raise SystemExit("--render is only supported with --format dot.")
        render_output = args.render_output or output_path.with_suffix(f".{args.render}")
        render_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["dot", f"-T{args.render}", str(output_path), "-o", str(render_output)],
            check=True,
        )
        print(f"Rendered Graphviz {args.render}: {render_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
