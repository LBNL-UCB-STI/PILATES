from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_H5,
)
from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    pass


@dataclass
class UrbanSimPreprocessOutputs(StepOutputsBase):
    """
    Outputs from the UrbanSim preprocess step.

    Attributes
    ----------
    usim_mutable_data_dir : Path
        UrbanSim mutable data directory prepared for the runner.
    prepared_inputs : dict
        Mapping of input short_name to prepared input path.
    """

    primary_output_attr: ClassVar[str] = "usim_mutable_data_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("usim_mutable_data_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)
    usim_mutable_data_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield prepared UrbanSim input records.
        """
        for key, path in self.prepared_inputs.items():
            yield key, path, f"UrbanSim prepared input: {key}"


@dataclass
class UrbanSimRunOutputs(StepOutputsBase):
    """
    Outputs from the UrbanSim run step.

    Attributes
    ----------
    usim_datastore_h5 : Path, optional
        UrbanSim datastore output for the forecast year.
    raw_outputs : dict
        Mapping of short_name to raw output path.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    declared_outputs: ClassVar[Tuple[str, ...]] = (USIM_DATASTORE_H5,)
    required_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    usim_datastore_h5: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield UrbanSim raw output records.
        """
        for key, path in self.raw_outputs.items():
            yield key, path, f"UrbanSim raw output: {key}"


@dataclass
class UrbanSimPostprocessOutputs(StepOutputsBase):
    """
    Outputs from the UrbanSim postprocess step.

    Attributes
    ----------
    usim_datastore_h5 : Path, optional
        UrbanSim datastore prepared for the next iteration.
    processed_outputs : dict
        Mapping of short_name to postprocessed output path.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    usim_datastore_h5: Path
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield UrbanSim postprocessed output records.
        """
        for key, path in self.processed_outputs.items():
            yield key, path, f"UrbanSim postprocess output: {key}"
