"""IMPACTS pipeline configuration models."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


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
    seed: int = Field(0, description="Random seed passed to IMPACTS")
    pipeline_presim: ImpactsPipelinePresimConfig = Field(
        default_factory=ImpactsPipelinePresimConfig
    )
    pipeline_postsim: ImpactsPipelinePostsimConfig = Field(
        default_factory=ImpactsPipelinePostsimConfig
    )
    population_fallback_to_beam_inputs: bool = True
    overrides: Dict[str, Any] = Field(default_factory=dict)

    def to_consist_facet(self) -> Dict[str, Any]:
        return self.model_dump()
