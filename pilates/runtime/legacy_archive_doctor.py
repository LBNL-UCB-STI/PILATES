from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is available in the PILATES env.
    yaml = None


DOCTOR_DIR = ".workflow/legacy_archive_doctor"
LIFECYCLE_AUDIT = ".workflow/diagnostics/artifact_lifecycle_audit.jsonl"
ASIM_INPUT_DIR_RE = re.compile(
    r"^inputs-year-(?P<year>\d{4})-iteration-(?P<iteration>\d+)$"
)
ASIM_INPUT_FILES = (
    "households.csv",
    "persons.csv",
    "land_use.csv",
    "skims.omx",
    "skims.zarr",
)
HOUSEHOLD_ID_COLUMNS = (
    "household_id",
    "householdid",
    "householdId",
    "HHID",
    "hh_id",
)
DEFAULT_MAX_POPULATION_FILE_BYTES = 5_000_000
DEFAULT_MAX_POPULATION_ROWS = 10_000


@dataclass(slots=True)
class DoctorAction:
    action: str
    reason: str
    source: str
    destination: str
    status: str = "planned"
    applied: bool = False
    detail: Optional[str] = None


@dataclass(slots=True)
class DoctorConflict:
    reason: str
    source: str
    destination: str
    detail: str


@dataclass(slots=True)
class PopulationSample:
    role: str
    path: str
    status: str
    household_id_count: int = 0
    skipped_reason: Optional[str] = None
    ids: set[str] = field(default_factory=set, repr=False)

    def to_report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("ids", None)
        return payload


@dataclass(slots=True)
class DoctorResult:
    archive_run_dir: str
    mode: str
    generated_at: str
    doctor_dir: str
    run_state_path: Optional[str]
    lifecycle_audit_path: Optional[str]
    run_state: dict[str, Any]
    lifecycle_event_count: int
    activitysim_input_dirs: list[dict[str, Any]]
    h5_path_mismatches: list[dict[str, Any]]
    actions: list[DoctorAction]
    conflicts: list[DoctorConflict]
    mixed_population_risk: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "archive_run_dir": self.archive_run_dir,
            "mode": self.mode,
            "generated_at": self.generated_at,
            "doctor_dir": self.doctor_dir,
            "run_state_path": self.run_state_path,
            "lifecycle_audit_path": self.lifecycle_audit_path,
            "run_state": self.run_state,
            "lifecycle_event_count": self.lifecycle_event_count,
            "activitysim_input_dirs": self.activitysim_input_dirs,
            "h5_path_mismatches": self.h5_path_mismatches,
            "actions": [asdict(action) for action in self.actions],
            "conflicts": [asdict(conflict) for conflict in self.conflicts],
            "mixed_population_risk": self.mixed_population_risk,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_run_state(run_dir: Path) -> tuple[Optional[Path], dict[str, Any]]:
    path = run_dir / "run_state.yaml"
    if not path.exists():
        return None, {}
    if yaml is None:
        return path, {"_error": "PyYAML is unavailable; run_state.yaml was not parsed"}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return path, {"_error": "run_state.yaml did not contain a mapping"}
    return path, loaded


def _read_lifecycle_events(
    run_dir: Path,
) -> tuple[Optional[Path], list[dict[str, Any]]]:
    path = run_dir / LIFECYCLE_AUDIT
    if not path.exists():
        return None, []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return path, events


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_forecast_years(
    run_state: Mapping[str, Any],
    lifecycle_events: Iterable[Mapping[str, Any]],
) -> set[int]:
    years: set[int] = set()
    for key in ("forecast_year", "target_year"):
        year = _safe_int(run_state.get(key))
        if year is not None:
            years.add(year)
    for event in lifecycle_events:
        if event.get("artifact_family") != "asim_input_archived":
            continue
        for key in ("forecast_year", "target_year", "year"):
            year = _safe_int(event.get(key))
            if year is not None:
                years.add(year)
    return years


def _discover_activitysim_input_dirs(run_dir: Path) -> list[dict[str, Any]]:
    dirs: list[dict[str, Any]] = []
    candidates = sorted(
        path
        for path in (run_dir / "activitysim" / "output").glob(
            "inputs-year-*-iteration-*"
        )
        if path.is_dir()
    )
    if not candidates:
        candidates = sorted(
            path
            for path in run_dir.rglob("inputs-year-*-iteration-*")
            if path.is_dir() and ".workflow" not in path.parts
        )
    for path in candidates:
        match = ASIM_INPUT_DIR_RE.match(path.name)
        if not match:
            continue
        files = [name for name in ASIM_INPUT_FILES if (path / name).exists()]
        dirs.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(run_dir)),
                "year": int(match.group("year")),
                "iteration": int(match.group("iteration")),
                "files": files,
            }
        )
    return dirs


def _discover_h5_path_mismatches(
    *,
    run_dir: Path,
    lifecycle_events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for event in lifecycle_events:
        semantic_year = _safe_int(event.get("year") or event.get("forecast_year"))
        if semantic_year is None:
            continue
        path_value = event.get("path") or event.get("src") or event.get("dest")
        if not path_value:
            continue
        path_text = str(path_value)
        match = re.search(r"model_data_(?P<path_year>\d{4})\.h5$", path_text)
        if match is None:
            continue
        path_year = int(match.group("path_year"))
        if path_year == semantic_year:
            continue
        key = (path_text, semantic_year, path_year)
        if key in seen:
            continue
        seen.add(key)
        path = Path(path_text)
        try:
            relative_path = str(path.relative_to(run_dir))
        except ValueError:
            relative_path = None
        mismatches.append(
            {
                "reason": "h5_forecast_year_path_mismatch",
                "event_type": event.get("event_type"),
                "artifact_key": event.get("key"),
                "artifact_family": event.get("artifact_family"),
                "semantic_year": semantic_year,
                "path_year": path_year,
                "path": path_text,
                "relative_path": relative_path,
                "canonical_filename": f"model_data_{semantic_year}.h5",
                "repairable": False,
                "detail": (
                    "H5 datastore path year does not match lifecycle semantic year; "
                    "the doctor reports this but does not rewrite H5 semantics."
                ),
            }
        )
    return mismatches


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_payload(source: Path, destination: Path) -> bool:
    if source.is_dir() or destination.is_dir():
        return source.is_dir() and destination.is_dir()
    if source.stat().st_size != destination.stat().st_size:
        return False
    return _file_digest(source) == _file_digest(destination)


def _copy_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def _plan_activitysim_aliases(
    *,
    run_dir: Path,
    run_state: Mapping[str, Any],
    lifecycle_events: Iterable[Mapping[str, Any]],
    input_dirs: Iterable[Mapping[str, Any]],
) -> tuple[list[DoctorAction], list[DoctorConflict]]:
    actions: list[DoctorAction] = []
    conflicts: list[DoctorConflict] = []
    forecast_years = _infer_forecast_years(run_state, lifecycle_events)
    if not forecast_years:
        return actions, conflicts

    current_year = _safe_int(run_state.get("year") or run_state.get("current_year"))
    state_iteration = _safe_int(
        run_state.get("iteration") or run_state.get("current_inner_iter")
    )
    for source_info in input_dirs:
        source_year = _safe_int(source_info.get("year"))
        source_iteration = _safe_int(source_info.get("iteration"))
        if state_iteration is not None and source_iteration != state_iteration:
            continue
        if current_year is not None and source_year not in {
            current_year,
            *forecast_years,
        }:
            continue
        if not {"households.csv", "persons.csv", "land_use.csv"}.issubset(
            set(source_info.get("files") or [])
        ):
            continue

        source_dir = run_dir / str(source_info["relative_path"])
        for forecast_year in sorted(forecast_years):
            if forecast_year == source_year:
                continue
            destination_dir = source_dir.parent / (
                f"inputs-year-{forecast_year}-iteration-{source_iteration}"
            )
            for name in source_info.get("files") or []:
                source = source_dir / name
                destination = destination_dir / name
                if destination.exists():
                    if _same_payload(source, destination):
                        actions.append(
                            DoctorAction(
                                action="copy",
                                reason="activitysim_forecast_year_alias",
                                source=str(source),
                                destination=str(destination),
                                status="already_present",
                                detail="destination already exists with matching payload",
                            )
                        )
                    else:
                        conflicts.append(
                            DoctorConflict(
                                reason="activitysim_forecast_year_alias_conflict",
                                source=str(source),
                                destination=str(destination),
                                detail="destination exists with different payload",
                            )
                        )
                    continue
                actions.append(
                    DoctorAction(
                        action="copy",
                        reason="activitysim_forecast_year_alias",
                        source=str(source),
                        destination=str(destination),
                    )
                )
    return actions, conflicts


def _role_for_population_path(path: Path) -> Optional[str]:
    lowered = "/".join(path.parts).lower()
    name_lower = path.name.lower()
    if "vehicles2" in name_lower:
        return "atlas"
    if "household" not in name_lower:
        return None
    if "activitysim" in lowered or "asim" in lowered:
        return "activitysim"
    if "beam" in lowered:
        return "beam"
    if "atlas" in lowered:
        return "atlas"
    return None


def _population_candidate_paths(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in run_dir.rglob("*"):
        if path.is_dir() or ".workflow" in path.parts:
            continue
        if _role_for_population_path(path) is None:
            continue
        suffixes = "".join(path.suffixes)
        if (
            suffixes.endswith(".csv")
            or suffixes.endswith(".csv.gz")
            or suffixes.endswith(".parquet")
        ):
            candidates.append(path)
    return sorted(candidates)


def _normalize_household_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _find_household_id_column(columns: Iterable[Any]) -> Optional[str]:
    by_lower = {str(column).lower(): str(column) for column in columns}
    for candidate in HOUSEHOLD_ID_COLUMNS:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    for column in columns:
        lowered = str(column).lower()
        if lowered.replace("_", "") == "householdid":
            return str(column)
    return None


def _open_text_table(path: Path):
    if "".join(path.suffixes).endswith(".csv.gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_csv_household_ids(
    path: Path, max_rows: int
) -> tuple[set[str], Optional[str]]:
    with _open_text_table(path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return set(), "missing header"
        column = _find_household_id_column(reader.fieldnames)
        if column is None:
            return set(), "missing household id column"
        ids: set[str] = set()
        for row_number, row in enumerate(reader, start=1):
            if row_number > max_rows:
                return ids, f"stopped after {max_rows} rows"
            value = row.get(column)
            if value not in (None, ""):
                ids.add(_normalize_household_id(value))
        return ids, None


def _read_parquet_household_ids(
    path: Path, max_rows: int
) -> tuple[set[str], Optional[str]]:
    try:
        import pandas as pd
    except ImportError:
        return set(), "pandas unavailable for parquet"
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - engine-specific detail.
        return set(), f"could not read parquet: {exc}"
    column = _find_household_id_column(frame.columns)
    if column is None:
        return set(), "missing household id column"
    ids = {
        _normalize_household_id(value)
        for value in frame[column].head(max_rows)
        if value is not None and str(value).strip() != ""
    }
    skipped = None
    if len(frame.index) > max_rows:
        skipped = f"stopped after {max_rows} rows"
    return ids, skipped


def _read_population_sample(
    path: Path,
    *,
    max_bytes: int,
    max_rows: int,
) -> PopulationSample:
    role = _role_for_population_path(path) or "unknown"
    if path.stat().st_size > max_bytes:
        return PopulationSample(
            role=role,
            path=str(path),
            status="skipped",
            skipped_reason=f"file larger than {max_bytes} bytes",
        )
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".parquet"):
        ids, skipped = _read_parquet_household_ids(path, max_rows)
    else:
        ids, skipped = _read_csv_household_ids(path, max_rows)
    status = "sampled" if ids else "skipped"
    return PopulationSample(
        role=role,
        path=str(path),
        status=status,
        household_id_count=len(ids),
        skipped_reason=skipped,
        ids=ids,
    )


def _detect_mixed_population_risk(run_dir: Path) -> dict[str, Any]:
    samples = [
        _read_population_sample(
            path,
            max_bytes=DEFAULT_MAX_POPULATION_FILE_BYTES,
            max_rows=DEFAULT_MAX_POPULATION_ROWS,
        )
        for path in _population_candidate_paths(run_dir)
    ]
    role_ids: dict[str, set[str]] = {}
    for sample in samples:
        if sample.status != "sampled":
            continue
        role_ids.setdefault(sample.role, set()).update(sample.ids)

    comparisons: list[dict[str, Any]] = []
    roles = sorted(role_ids)
    risk = False
    for index, left_role in enumerate(roles):
        for right_role in roles[index + 1 :]:
            left_ids = role_ids[left_role]
            right_ids = role_ids[right_role]
            if left_ids == right_ids:
                status = "match"
            else:
                status = "mismatch"
                risk = True
            comparisons.append(
                {
                    "left_role": left_role,
                    "right_role": right_role,
                    "status": status,
                    "left_only_sample": sorted(left_ids - right_ids)[:10],
                    "right_only_sample": sorted(right_ids - left_ids)[:10],
                }
            )

    if risk:
        status = "risk"
    elif len(roles) >= 2:
        status = "clear"
    else:
        status = "unknown"
    return {
        "status": status,
        "sampled_roles": roles,
        "samples": [sample.to_report() for sample in samples],
        "comparisons": comparisons,
    }


def _apply_actions(
    actions: list[DoctorAction], conflicts: list[DoctorConflict]
) -> None:
    conflicted_destinations = {conflict.destination for conflict in conflicts}
    for action in actions:
        if action.status == "already_present":
            action.applied = False
            continue
        if action.destination in conflicted_destinations:
            action.status = "blocked"
            action.detail = "destination has a recorded conflict"
            continue
        _copy_path(Path(action.source), Path(action.destination))
        action.status = "applied"
        action.applied = True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def inspect_legacy_archive(
    archive_run_dir: str | Path,
    *,
    apply: bool = False,
) -> DoctorResult:
    run_dir = Path(archive_run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Archive run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Archive run path is not a directory: {run_dir}")

    run_state_path, run_state = _read_run_state(run_dir)
    lifecycle_path, lifecycle_events = _read_lifecycle_events(run_dir)
    input_dirs = _discover_activitysim_input_dirs(run_dir)
    h5_path_mismatches = _discover_h5_path_mismatches(
        run_dir=run_dir,
        lifecycle_events=lifecycle_events,
    )
    actions, conflicts = _plan_activitysim_aliases(
        run_dir=run_dir,
        run_state=run_state,
        lifecycle_events=lifecycle_events,
        input_dirs=input_dirs,
    )
    if apply:
        _apply_actions(actions, conflicts)

    doctor_dir = run_dir / DOCTOR_DIR
    result = DoctorResult(
        archive_run_dir=str(run_dir),
        mode="apply" if apply else "dry-run",
        generated_at=_utc_now(),
        doctor_dir=str(doctor_dir),
        run_state_path=str(run_state_path) if run_state_path else None,
        lifecycle_audit_path=str(lifecycle_path) if lifecycle_path else None,
        run_state=dict(run_state),
        lifecycle_event_count=len(lifecycle_events),
        activitysim_input_dirs=input_dirs,
        h5_path_mismatches=h5_path_mismatches,
        actions=actions,
        conflicts=conflicts,
        mixed_population_risk=_detect_mixed_population_risk(run_dir),
    )
    _write_json(doctor_dir / "report.json", result.to_report())
    _write_jsonl(
        doctor_dir / ("actions.jsonl" if apply else "candidate_actions.jsonl"),
        (asdict(action) for action in actions),
    )
    if apply:
        _write_json(
            doctor_dir / "manifest.json",
            {
                "archive_run_dir": result.archive_run_dir,
                "generated_at": result.generated_at,
                "mode": result.mode,
                "report": "report.json",
                "actions": "actions.jsonl",
                "conflicts": "conflicts.json",
                "action_count": len(actions),
                "conflict_count": len(conflicts),
                "mutated_consist_db_metadata": False,
            },
        )
        _write_json(
            doctor_dir / "conflicts.json",
            {"conflicts": [asdict(conflict) for conflict in conflicts]},
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and repair safe filesystem-only legacy PILATES archive issues.",
    )
    parser.add_argument(
        "--archive-run-dir",
        required=True,
        help="Archived PILATES run directory to inspect.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="Inspect and write a report only."
    )
    mode.add_argument(
        "--apply", action="store_true", help="Create safe filesystem aliases/copies."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = inspect_legacy_archive(args.archive_run_dir, apply=args.apply)
    print(
        json.dumps(
            {"report": str(Path(result.doctor_dir) / "report.json")}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
