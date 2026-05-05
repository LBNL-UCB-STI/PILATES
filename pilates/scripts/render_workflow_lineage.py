from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pilates.workflows.lineage_render import render_plan_json, render_plan_mermaid
from pilates.workflows.lineage_render import render_plan_html
from pilates.workflows.planning import build_static_execution_plan_from_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a static workflow lineage plan from a PILATES settings YAML."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the PILATES settings YAML.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "mermaid", "html"),
        default="json",
        help="Render format. JSON is the primary debugging surface.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="Omit the postprocessing step from the static plan.",
    )
    parser.add_argument(
        "--hide-terminal-artifacts",
        action="store_true",
        help="Hide artifact boxes that are never consumed by another step.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = build_static_execution_plan_from_file(
        args.config,
        include_postprocessing=not args.no_postprocessing,
    )
    if args.format == "mermaid":
        rendered = render_plan_mermaid(
            plan,
            hide_terminal_artifacts=args.hide_terminal_artifacts,
        )
    elif args.format == "html":
        rendered = render_plan_html(
            plan,
            hide_terminal_artifacts=args.hide_terminal_artifacts,
        )
    else:
        rendered = render_plan_json(plan)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(str(output_path))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
