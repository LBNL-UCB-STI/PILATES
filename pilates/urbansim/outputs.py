from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_INPUT_MERGED_PREFIX,
)
from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    from pilates.workspace import Workspace


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

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "UrbanSimPreprocessOutputs":
        """
        Build outputs from a RecordStore.

        Parameters
        ----------
        record_store : RecordStore
            RecordStore produced by preprocessing.
        workspace : Workspace
            Workspace used to resolve paths.

        Returns
        -------
        UrbanSimPreprocessOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        prepared_inputs: Dict[str, Path] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            prepared_inputs[key] = Path(path)
        return cls(
            usim_mutable_data_dir=Path(workspace.get_usim_mutable_data_dir()),
            prepared_inputs=prepared_inputs,
        )


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
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    usim_datastore_h5: Optional[Path]
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield UrbanSim raw output records.
        """
        for key, path in self.raw_outputs.items():
            yield key, path, f"UrbanSim raw output: {key}"

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "UrbanSimRunOutputs":
        """
        Build outputs from a RecordStore.

        Parameters
        ----------
        record_store : RecordStore
            RecordStore produced by the runner.
        workspace : Workspace
            Workspace used to resolve paths.

        Returns
        -------
        UrbanSimRunOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        raw_outputs: Dict[str, Path] = {}
        usim_datastore_h5 = None
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
            if key == USIM_FORECAST_OUTPUT:
                usim_datastore_h5 = Path(path)
        return cls(
            usim_datastore_h5=usim_datastore_h5,
            raw_outputs=raw_outputs,
        )


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
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    usim_datastore_h5: Optional[Path]
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield UrbanSim postprocessed output records.
        """
        for key, path in self.processed_outputs.items():
            yield key, path, f"UrbanSim postprocess output: {key}"

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "UrbanSimPostprocessOutputs":
        """
        Build outputs from a RecordStore.

        Parameters
        ----------
        record_store : RecordStore
            RecordStore produced by the postprocessor.
        workspace : Workspace
            Workspace used to resolve paths.

        Returns
        -------
        UrbanSimPostprocessOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        processed_outputs: Dict[str, Path] = {}
        usim_datastore_h5 = None
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            processed_outputs[key] = Path(path)
            if key.startswith(USIM_INPUT_MERGED_PREFIX):
                usim_datastore_h5 = Path(path)
        return cls(
            usim_datastore_h5=usim_datastore_h5,
            processed_outputs=processed_outputs,
        )
