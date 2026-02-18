from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    BEAM_FULL_SKIMS,
    FINAL_SKIMS_OMX,
    ZARR_SKIMS,
)
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
    split_events : dict
        Mapping of split BEAM events parquet short_names to paths.
    split_event_links : dict
        Mapping of derived link-level tables from split events.
    """

    primary_output_attr: ClassVar[str] = "zarr_skims"
    optional_path_fields: ClassVar[Tuple[str, ...]] = (
        "zarr_skims",
        "final_skims_omx",
    )
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("split_events", "split_event_links")

    zarr_skims: Optional[Path] = None
    final_skims_omx: Optional[Path] = None
    split_events: Dict[str, Path] = field(default_factory=dict)
    split_event_links: Dict[str, Path] = field(default_factory=dict)

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

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "BeamPostprocessOutputs":
        """
        Build outputs from a RecordStore.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        values: Dict[str, Any] = {}
        zarr_path = artifact_to_path(mapping.get(ZARR_SKIMS), workspace)
        if zarr_path is not None:
            values["zarr_skims"] = Path(zarr_path)
        omx_path = artifact_to_path(mapping.get(FINAL_SKIMS_OMX), workspace)
        if omx_path is not None:
            values["final_skims_omx"] = Path(omx_path)

        split_events: Dict[str, Path] = {}
        split_event_links: Dict[str, Path] = {}
        for key, value in mapping.items():
            if not key.startswith("events_parquet_") or "_type_" not in key:
                continue
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            split_events[key] = Path(path)
        for key, value in mapping.items():
            if not key.startswith("path_traversal_links_"):
                continue
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            split_event_links[key] = Path(path)
        values["split_events"] = split_events
        values["split_event_links"] = split_event_links
        return cls(**values)


@dataclass
class BeamFullSkimOutputs(StepOutputsBase):
    """
    Outputs from the BEAM full-skim step.

    Attributes
    ----------
    full_skims : Path
        Full-skim output file produced by FullSkimsCreatorApp.
    """

    primary_output_attr: ClassVar[str] = "full_skims"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("full_skims",)

    full_skims: Path

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        yield (
            BEAM_FULL_SKIMS,
            self.full_skims,
            "BEAM full-skim background skims output",
        )

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "BeamFullSkimOutputs":
        mapping = record_store.to_mapping() if record_store is not None else {}
        path = artifact_to_path(mapping.get(BEAM_FULL_SKIMS), workspace)
        if path is None:
            raise ValueError("Missing beam_full_skims in record store.")
        return cls(full_skims=Path(path))
