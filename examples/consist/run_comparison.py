from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ANALYSIS_SRC = Path(__file__).resolve().parents[2] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from pilates_consist_analysis import difference, open_archive, rank


def _parse_filter(raw: str) -> tuple[str, Any]:
    key, separator, value = str(raw).partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("Filters must use KEY=VALUE form.")
    text = value.strip()
    if text.isdigit():
        return key.strip(), int(text)
    return key.strip(), text


def _unique(values: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def build_mode_share_comparison(
    archive,
    *,
    table_name: str,
    compare: str,
    baseline: str | None,
    group_by: Sequence[str],
    filters: Mapping[str, Any],
    mode_column: str = "trip_mode",
):
    import ibis

    measured_by = _unique([*group_by, compare, mode_column])
    facets = _unique([*measured_by, *filters.keys()])
    table = archive.table(table_name, facets=facets, where=dict(filters))
    counts = archive.measure(
        table,
        by=measured_by,
        measures={"trip_count": lambda table_expr: table_expr.count()},
    )
    denominator_by = [column for column in measured_by if column != mode_column]
    denominator_window = ibis.window(
        group_by=[counts[column] for column in denominator_by],
    )
    total_trips = counts["trip_count"].sum().over(denominator_window)
    measured = counts.mutate(
        total_trips=total_trips,
        mode_share=counts["trip_count"] / total_trips,
    )
    comparison_by = [column for column in measured_by if column != compare]
    if baseline is not None:
        return difference(
            measured,
            value="mode_share",
            compare=compare,
            baseline=baseline,
            at=filters,
            by=comparison_by,
        )
    return rank(measured, value="mode_share", by=comparison_by)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure ActivitySim trip mode share and compare results over one facet.",
    )
    parser.add_argument("archive_run_dir", help="Archive run directory.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root mounted as Consist inputs. Defaults to the current PILATES checkout.",
    )
    parser.add_argument(
        "--table",
        default="activitysim.trips",
        help="Logical analysis table in '<model>.<logical_name>' form.",
    )
    parser.add_argument(
        "--compare",
        default="scenario_id",
        help="Facet to compare, for example scenario_id or pricing_policy.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline value for --compare. If omitted, mode shares are ranked instead.",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        default=None,
        help=(
            "Facet column to group by before adding --compare and trip_mode. "
            "Defaults to year and iteration. Repeatable."
        ),
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        type=_parse_filter,
        metavar="KEY=VALUE",
        help="Facet filter. Repeatable.",
    )
    parser.add_argument(
        "--mode-column",
        default="trip_mode",
        help="Trip mode column from the ActivitySim trips schema.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum result rows to print.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    filters = dict(args.where)
    group_by = args.group_by or ["year", "iteration"]

    archive = open_archive(
        Path(args.archive_run_dir),
        project_root=Path(args.project_root),
    )
    result = build_mode_share_comparison(
        archive,
        table_name=args.table,
        compare=args.compare,
        baseline=args.baseline,
        group_by=group_by,
        filters=filters,
        mode_column=args.mode_column,
    )

    print("Archive summary")
    print(archive.summary().to_string(index=False))
    print("\nMode share comparison")
    print(result.to_pandas().head(args.limit).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
