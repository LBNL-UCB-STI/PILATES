from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import FileRecord, RecordStore, sanitize_artifact_key
from pilates.impacts.outputs import ImpactsPreprocessOutputs
from pilates.utils.path_utils import find_project_root
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


def _find_first_existing(base_dir: Path, patterns: tuple[str, ...]) -> Optional[Path]:
    if not base_dir.exists():
        return None
    for pattern in patterns:
        matches = sorted(base_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _impacts_source_data_dir(settings: Any) -> Path:
    impacts_settings = getattr(settings, "impacts", None)
    if impacts_settings is None:
        raise ValueError("Impacts config is missing")

    data_dir = getattr(impacts_settings, "local_input_folder", None)
    if not data_dir:
        raise ValueError(
            "Impacts local_input_folder is not configured. "
            "Please set impacts.local_input_folder in settings."
        )

    if os.path.isabs(data_dir):
        return Path(data_dir)

    project_root = find_project_root(start_path=os.path.dirname(__file__))
    if not project_root:
        project_root = os.path.realpath(os.getcwd())
        logger.warning(
            "[NOT IDEAL] Could not locate PILATES project root via markers; falling back to cwd='%s'.",
            project_root,
        )
    return Path(project_root) / data_dir


def _build_seed_copy_records(
    *,
    source_dir: Path,
    dest_dir: Path,
) -> Tuple[RecordStore, RecordStore]:
    input_records = RecordStore()
    output_records = RecordStore()

    input_records.add_record(
        FileRecord(
            file_path=str(source_dir),
            short_name="impacts_bootstrap_data_root",
            description="Impacts seed data root",
        )
    )
    output_records.add_record(
        FileRecord(
            file_path=str(dest_dir),
            short_name="impacts_bootstrap_data_root",
            description="Mutable Impacts input root",
        )
    )

    for source_path in sorted(source_dir.rglob("*")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_dir)
        short_name = sanitize_artifact_key(relative_path.as_posix())
        if short_name is None:
            short_name = relative_path.as_posix().replace("/", "_")
        dest_path = dest_dir / relative_path
        input_records.add_record(
            FileRecord(
                file_path=str(source_path),
                short_name=short_name,
                description=f"Impacts seed input file: {relative_path.as_posix()}",
            )
        )
        output_records.add_record(
            FileRecord(
                file_path=str(dest_path),
                short_name=short_name,
                description=f"Mutable Impacts input file: {relative_path.as_posix()}",
            )
        )

    return input_records, output_records


class ImpactsPreprocessor(
    GenericPreprocessor[Optional[Dict[str, Any]], ImpactsPreprocessOutputs]
):
    """Stage downstream inputs for Docker-backed impacts execution."""

    def copy_data_to_mutable_location(
        self,
        settings: Any,
        output_dir: str,
        workspace: Optional[Workspace] = None,
    ) -> Tuple[RecordStore, RecordStore]:
        del workspace
        source_dir = _impacts_source_data_dir(settings)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Impacts seed data directory not found at {source_dir}"
            )

        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[ImpactsPreprocessor] Copying seed data from %s to %s",
            source_dir,
            dest_dir,
        )
        shutil.copytree(
            source_dir,
            dest_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".git*"),
        )
        return _build_seed_copy_records(source_dir=source_dir, dest_dir=dest_dir)

    def _preprocess(
        self,
        workspace: Workspace,
        previous_records: Optional[Dict[str, Any]],
    ) -> ImpactsPreprocessOutputs:
        del previous_records
        settings = self.state.full_settings
        cfg = settings.impacts
        if cfg is None:
            raise ValueError("Impacts config is missing")

        input_dir = Path(workspace.get_impacts_input_dir())
        input_dir.mkdir(parents=True, exist_ok=True)

        beam_output_dir = Path(workspace.get_beam_output_dir())
        beam_input_dir = Path(workspace.get_beam_mutable_data_dir())
        asim_output_dir = Path(workspace.get_asim_output_dir())
        asim_mutable_dir = Path(workspace.get_asim_mutable_data_dir())

        staged_inputs = {
            "beam_network": str(
                _find_first_existing(
                    beam_output_dir,
                    ("*network*.csv*", "*network*.xml*", "*network*.parquet"),
                )
                or ""
            ),
            "beam_emissions_skims": str(
                _find_first_existing(
                    beam_output_dir,
                    ("*emissions*.csv*", "*emissions*.parquet", "*emissions*.omx"),
                )
                or ""
            ),
            "beam_osm_pbf": str(
                _find_first_existing(
                    beam_input_dir,
                    ("*.osm.pbf", "*.pbf", "*.mapdb"),
                )
                or ""
            ),
            "activitysim_households": str(
                _find_first_existing(asim_output_dir, ("households*.parquet", "households*.csv"))
                or _find_first_existing(asim_mutable_dir, ("households*.csv", "households*.parquet"))
                or ""
            ),
            "activitysim_persons": str(
                _find_first_existing(asim_output_dir, ("persons*.parquet", "persons*.csv"))
                or _find_first_existing(asim_mutable_dir, ("persons*.csv", "persons*.parquet"))
                or ""
            ),
            "activitysim_land_use": str(
                _find_first_existing(asim_output_dir, ("land_use*.parquet", "land_use*.csv"))
                or _find_first_existing(asim_mutable_dir, ("land_use*.csv", "land_use*.parquet"))
                or ""
            ),
        }

        manifest_path = input_dir / cfg.input_manifest_filename
        manifest_payload = {
            "model": "impacts",
            "status": "scaffolded",
            "workspace": workspace.full_path,
            "inputs": staged_inputs,
            "missing_inputs": [
                key for key, value in staged_inputs.items() if not value or not os.path.exists(value)
            ],
            "notes": [
                "Impacts integration scaffolded for Docker execution.",
                "Replace placeholder input discovery with the concrete impacts contract once finalized.",
            ],
        }
        manifest_path.write_text(yaml.safe_dump(manifest_payload, sort_keys=True), encoding="utf-8")

        return ImpactsPreprocessOutputs(
            input_dir=input_dir,
            input_manifest=manifest_path,
            staged_inputs=staged_inputs,
        )
