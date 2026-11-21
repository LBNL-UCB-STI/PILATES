"""IMPACTS pipeline configuration models."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ImpactsPipelinePresimConfig(BaseModel):
    """Select IMPACTS presim sub-steps."""

    activities: bool = True
    fleet: bool = True


class ImpactsPipelinePostsimConfig(BaseModel):
    """Select IMPACTS postsim sub-steps."""

    emissions: bool = True
    inmap: bool = True
    aermod: bool = True
    exposure: bool = True


class ImpactsConfig(BaseModel):
    """Settings passed to the external IMPACTS pipeline."""

    local_input_folder: str = Field(..., description="Workspace-relative input folder")
    local_output_folder: str = Field(..., description="Workspace-relative output folder")
    local_inmap_data_folder: Optional[str] = Field(
        None,
        description=(
            "Optional host directory containing the InMAP dataset, mounted read-only "
            "at /app/inmap in IMPACTS containers."
        ),
    )
    shared_input_cache_folder: Optional[str] = Field(
        None,
        description=(
            "Optional durable host directory shared by PILATES runs for cached IMPACTS inputs; "
            "defaults to a cache directory under run.output_directory."
        ),
    )
    seed: int = Field(0, description="Random seed passed to IMPACTS")
    pipeline_presim: ImpactsPipelinePresimConfig = Field(
        default_factory=ImpactsPipelinePresimConfig
    )
    pipeline_postsim: ImpactsPipelinePostsimConfig = Field(
        default_factory=ImpactsPipelinePostsimConfig
    )
    population_fallback_to_beam_inputs: bool = True
    overrides: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("shared_input_cache_folder", "local_inmap_data_folder")
    @classmethod
    def expand_host_folder(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        if value is None:
            return None
        expanded = os.path.expandvars(value).strip()
        if "$" in expanded:
            raise ValueError(
                f"{info.field_name} contains an unset environment variable: {value}"
            )
        return expanded

    def to_consist_facet(self) -> Dict[str, Any]:
        return self.model_dump()
