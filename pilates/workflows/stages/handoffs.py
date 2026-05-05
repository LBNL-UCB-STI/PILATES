from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Union

from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
)


ArtifactPath = Union[str, os.PathLike]


@dataclass(frozen=True)
class LandUseToSupplyDemandHandoff(Mapping[str, ArtifactPath]):
    """
    Typed UrbanSim handoff between the land-use and supply-demand stages.

    The stage boundary only needs a small set of semantic UrbanSim roles. This
    object keeps those roles explicit at the call site while still allowing
    downstream ActivitySim/BEAM helpers to materialize the legacy mapping shape
    they already consume internally.
    """

    usim_datastore_base_h5: Optional[ArtifactPath] = None
    usim_datastore_current_h5: Optional[ArtifactPath] = None
    usim_population_source_h5: Optional[ArtifactPath] = None
    usim_forecast_output: Optional[ArtifactPath] = None

    def to_input_mapping(self) -> Dict[str, ArtifactPath]:
        mapping: Dict[str, ArtifactPath] = {}
        if self.usim_datastore_base_h5 is not None:
            mapping[USIM_DATASTORE_BASE_H5] = self.usim_datastore_base_h5
        if self.usim_datastore_current_h5 is not None:
            mapping[USIM_DATASTORE_CURRENT_H5] = self.usim_datastore_current_h5
        if self.usim_population_source_h5 is not None:
            mapping[USIM_POPULATION_SOURCE_H5] = self.usim_population_source_h5
        if self.usim_forecast_output is not None:
            mapping[USIM_FORECAST_OUTPUT] = self.usim_forecast_output
        return mapping

    def __getitem__(self, key: str) -> ArtifactPath:
        return self.to_input_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_input_mapping())

    def __len__(self) -> int:
        return len(self.to_input_mapping())

    @classmethod
    def from_mapping(
        cls,
        mapping: Optional[
            Union["LandUseToSupplyDemandHandoff", Mapping[str, ArtifactPath]]
        ],
    ) -> "LandUseToSupplyDemandHandoff":
        if isinstance(mapping, cls):
            return mapping
        if not mapping:
            return cls()
        return cls(
            usim_datastore_base_h5=mapping.get(USIM_DATASTORE_BASE_H5),
            usim_datastore_current_h5=mapping.get(USIM_DATASTORE_CURRENT_H5),
            usim_population_source_h5=mapping.get(USIM_POPULATION_SOURCE_H5),
            usim_forecast_output=mapping.get(USIM_FORECAST_OUTPUT),
        )

    @property
    def is_empty(self) -> bool:
        return not self.to_input_mapping()
