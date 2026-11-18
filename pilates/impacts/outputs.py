from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Tuple

from pilates.workflows.artifact_keys import (
    IMPACTS_EXPOSURE_TABLE,
    IMPACTS_PIPELINE_MANIFEST,
    IMPACTS_POSTPROCESS_MANIFEST,
    IMPACTS_POSTSIM_SETTINGS,
    IMPACTS_PRESIM_SETTINGS,
)
from pilates.workflows.outputs_base import StepOutputsBase


@dataclass
class ImpactsPresimOutputs(StepOutputsBase):
    """Outputs from the IMPACTS presim stage (EMFAC activities + fleet)."""

    primary_output_attr: ClassVar[str] = "settings_path"
    declared_outputs: ClassVar[Tuple[str, ...]] = (IMPACTS_PRESIM_SETTINGS,)
    record_keys: ClassVar[Dict[str, str]] = {
        "settings_path": IMPACTS_PRESIM_SETTINGS,
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "settings_path": "Derived IMPACTS presim settings file",
    }
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_dir", "settings_path")

    output_dir: Path
    settings_path: Path
    bridge_settings_path: Path


@dataclass
class ImpactsPostsimOutputs(StepOutputsBase):
    """Outputs from the IMPACTS postsim stage (emissions/inmap/aermod/exposure)."""

    primary_output_attr: ClassVar[str] = "exposure_table"
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        IMPACTS_POSTSIM_SETTINGS,
        IMPACTS_PIPELINE_MANIFEST,
        IMPACTS_POSTPROCESS_MANIFEST,
        IMPACTS_EXPOSURE_TABLE,
    )
    record_keys: ClassVar[Dict[str, str]] = {
        "settings_path": IMPACTS_POSTSIM_SETTINGS,
        "pipeline_manifest": IMPACTS_PIPELINE_MANIFEST,
        "postprocess_manifest": IMPACTS_POSTPROCESS_MANIFEST,
        "exposure_table": IMPACTS_EXPOSURE_TABLE,
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "settings_path": "Derived IMPACTS postsim settings file",
        "pipeline_manifest": "Impacts pipeline manifest",
        "postprocess_manifest": "Impacts postprocess manifest",
        "exposure_table": "Impacts finalized exposure table",
    }
    required_path_fields: ClassVar[Tuple[str, ...]] = (
        "output_dir",
        "settings_path",
        "pipeline_manifest",
        "postprocess_manifest",
    )

    output_dir: Path
    settings_path: Path
    bridge_settings_path: Path
    pipeline_manifest: Path
    postprocess_manifest: Path
    exposure_table: Path
