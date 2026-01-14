from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
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
    usim_mutable_data_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected input paths exist.
        """
        assert (
            self.usim_mutable_data_dir.exists()
        ), f"usim_mutable_data_dir missing: {self.usim_mutable_data_dir}"
        for key, path in self.prepared_inputs.items():
            if not path.exists():
                raise AssertionError(
                    f"urbansim preprocess input missing for {key}: {path}"
                )

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
    usim_datastore_h5: Optional[Path]
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        if self.usim_datastore_h5 is not None:
            assert (
                self.usim_datastore_h5.exists()
            ), f"usim_datastore_h5 missing: {self.usim_datastore_h5}"
        for key, path in self.raw_outputs.items():
            if not path.exists():
                raise AssertionError(f"urbansim run output missing for {key}: {path}")

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
            if key == "usim_forecast_output":
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
    usim_datastore_h5: Optional[Path]
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        if self.usim_datastore_h5 is not None:
            assert (
                self.usim_datastore_h5.exists()
            ), f"usim_datastore_h5 missing: {self.usim_datastore_h5}"
        for key, path in self.processed_outputs.items():
            if not path.exists():
                raise AssertionError(
                    f"urbansim postprocess output missing for {key}: {path}"
                )

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
            if key.startswith("usim_input_merged_"):
                usim_datastore_h5 = Path(path)
        return cls(
            usim_datastore_h5=usim_datastore_h5,
            processed_outputs=processed_outputs,
        )
