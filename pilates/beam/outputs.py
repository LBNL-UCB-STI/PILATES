from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_constants import FINAL_SKIMS_OMX, ZARR_SKIMS
from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    from pilates.workspace import Workspace


@dataclass
class BeamPreprocessOutputs(StepOutputsBase):
    """
    Outputs from the BEAM preprocess step.

    Attributes
    ----------
    beam_mutable_data_dir : Path
        BEAM mutable data directory populated for the runner.
    prepared_inputs : dict
        Mapping of input short_name to prepared input path.
    """

    primary_output_attr: ClassVar[str] = "beam_mutable_data_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("beam_mutable_data_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)
    beam_mutable_data_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield prepared BEAM input records.
        """
        for key, path in self.prepared_inputs.items():
            yield key, path, f"BEAM prepared input: {key}"

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "BeamPreprocessOutputs":
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
        BeamPreprocessOutputs
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
            beam_mutable_data_dir=Path(workspace.get_beam_mutable_data_dir()),
            prepared_inputs=prepared_inputs,
        )


@dataclass
class BeamRunOutputs(StepOutputsBase):
    """
    Outputs from the BEAM run step.

    Attributes
    ----------
    beam_output_dir : Path
        BEAM output directory for the run.
    raw_outputs : dict
        Mapping of short_name to raw output path.
    """

    primary_output_attr: ClassVar[str] = "beam_output_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("beam_output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    beam_output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield BEAM raw output records.
        """
        for key, path in self.raw_outputs.items():
            yield key, path, f"BEAM raw output: {key}"

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "BeamRunOutputs":
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
        BeamRunOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        raw_outputs: Dict[str, Path] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
        return cls(
            beam_output_dir=Path(workspace.get_beam_output_dir()),
            raw_outputs=raw_outputs,
        )


@dataclass
class BeamPostprocessOutputs(StepOutputsBase):
    """
    Outputs from the BEAM postprocess step.

    Attributes
    ----------
    zarr_skims : Path, optional
        Zarr skims updated by BEAM.
    final_skims_omx : Path, optional
        Final OMX skims for downstream models. When present, it is treated as
        the primary output to log.
    """

    primary_output_attr: ClassVar[str] = "zarr_skims"
    optional_path_fields: ClassVar[Tuple[str, ...]] = (
        "zarr_skims",
        "final_skims_omx",
    )

    zarr_skims: Optional[Path] = None
    final_skims_omx: Optional[Path] = None

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield only the skim file updated for downstream use.
        """
        if self.final_skims_omx is not None:
            yield (
                FINAL_SKIMS_OMX,
                self.final_skims_omx,
                "Final skims OMX for downstream models",
            )
            return
        if self.zarr_skims is not None:
            yield (
                ZARR_SKIMS,
                self.zarr_skims,
                "Zarr skims updated with BEAM outputs",
            )
