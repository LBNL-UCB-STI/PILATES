from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Tuple

from pilates.generic.records import RecordStore
from pilates.workflows.artifact_keys import (
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_INPUT_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_RUN_MANIFEST,
)
from pilates.workflows.outputs_base import StepOutputsBase


@dataclass
class ImpactsPreprocessOutputs(StepOutputsBase):
    """Outputs from impacts preprocess staging."""

    primary_output_attr: ClassVar[str] = "input_manifest"
    declared_outputs: ClassVar[Tuple[str, ...]] = (IMPACTS_INPUT_MANIFEST,)
    record_keys: ClassVar[Dict[str, str]] = {
        "input_manifest": IMPACTS_INPUT_MANIFEST,
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "input_manifest": "Impacts staged-input manifest",
    }
    required_path_fields: ClassVar[Tuple[str, ...]] = ("input_dir", "input_manifest")

    input_dir: Path
    input_manifest: Path
    staged_inputs: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ImpactsPreprocessOutputs":
        from pilates.workspace import Workspace

        del Workspace
        mapping = record_store.to_mapping() if record_store is not None else {}
        manifest = mapping.get(IMPACTS_INPUT_MANIFEST)
        if manifest is None:
            raise ValueError("Impacts preprocess record store is missing input manifest")
        return cls(
            input_dir=Path(workspace.get_impacts_input_dir()),
            input_manifest=Path(str(manifest)),
        )


@dataclass
class ImpactsRunOutputs(StepOutputsBase):
    """Outputs from impacts Docker execution."""

    primary_output_attr: ClassVar[str] = "run_manifest"
    declared_outputs: ClassVar[Tuple[str, ...]] = (IMPACTS_RUN_MANIFEST,)
    record_keys: ClassVar[Dict[str, str]] = {
        "run_manifest": IMPACTS_RUN_MANIFEST,
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "run_manifest": "Impacts Docker run manifest",
    }
    required_path_fields: ClassVar[Tuple[str, ...]] = (
        "output_dir",
        "run_manifest",
        "raw_exposure_table",
    )

    output_dir: Path
    run_manifest: Path
    raw_exposure_table: Path
    docker_command: str = ""


@dataclass
class ImpactsPostprocessOutputs(StepOutputsBase):
    """Finalized impacts exposure outputs."""

    primary_output_attr: ClassVar[str] = "exposure_table"
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        IMPACTS_EXPOSURE_TABLE,
        IMPACTS_POSTPROCESS_MANIFEST,
    )
    record_keys: ClassVar[Dict[str, str]] = {
        "exposure_table": IMPACTS_EXPOSURE_TABLE,
        "postprocess_manifest": IMPACTS_POSTPROCESS_MANIFEST,
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "exposure_table": "Impacts finalized exposure table",
        "postprocess_manifest": "Impacts postprocess manifest",
    }
    required_path_fields: ClassVar[Tuple[str, ...]] = (
        "output_dir",
        "exposure_table",
        "postprocess_manifest",
    )

    output_dir: Path
    exposure_table: Path
    postprocess_manifest: Path
