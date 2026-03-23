from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import yaml

from pilates.generic.preprocessor import GenericPreprocessor
from pilates.generic.records import RecordStore
from pilates.impacts.outputs import ImpactsPreprocessOutputs
from pilates.workspace import Workspace


def _find_first_existing(base_dir: Path, patterns: tuple[str, ...]) -> Optional[Path]:
    if not base_dir.exists():
        return None
    for pattern in patterns:
        matches = sorted(base_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


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
        del settings
        del workspace
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return RecordStore(), RecordStore()

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
