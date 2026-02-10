from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union
import json

from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
)
from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    from pilates.workspace import Workspace


# Map legacy ActivitySim output keys (file stems) to namespaced keys.
ASIM_OUTPUT_KEY_MAP: Dict[str, str] = {
    "accessibility": "accessibility_asim_out",
    "beam_plans": "beam_plans_asim_out",
    "disaggregate_accessibility": "disaggregate_accessibility_asim_out",
    "households": "households_asim_out",
    "joint_tour_participants": "joint_tour_participants_asim_out",
    "land_use": "land_use_asim_out",
    "non_mandatory_tour_destination_accessibility": "non_mandatory_tour_destination_accessibility_asim_out",
    "person_windows": "person_windows_asim_out",
    "persons": "persons_asim_out",
    "proto_disaggregate_accessibility": "proto_disaggregate_accessibility_asim_out",
    "proto_households": "proto_households_asim_out",
    "proto_persons": "proto_persons_asim_out",
    "proto_persons_merged": "proto_persons_merged_asim_out",
    "proto_tours": "proto_tours_asim_out",
    "school_destination_size": "school_destination_size_asim_out",
    "school_modeled_size": "school_modeled_size_asim_out",
    "school_shadow_prices": "school_shadow_prices_asim_out",
    "tours": "tours_asim_out",
    "trips": "trips_asim_out",
    "workplace_destination_size": "workplace_destination_size_asim_out",
    "workplace_location_accessibility": "workplace_location_accessibility_asim_out",
    "workplace_modeled_size": "workplace_modeled_size_asim_out",
    "workplace_shadow_prices": "workplace_shadow_prices_asim_out",
}


def normalize_asim_output_key(key: str) -> str:
    return ASIM_OUTPUT_KEY_MAP.get(key, key)


def _asim_run_marker_filename(year: int, iteration: int) -> str:
    return f".pilates_asim_run_success_year_{year}_iter_{iteration}.json"


def get_asim_run_marker_path(output_dir: Union[str, Path], year: int, iteration: int) -> Path:
    return Path(output_dir) / _asim_run_marker_filename(year, iteration)


def has_asim_run_marker(output_dir: Union[str, Path], year: int, iteration: int) -> bool:
    return get_asim_run_marker_path(output_dir, year, iteration).exists()


def clear_asim_run_marker(output_dir: Union[str, Path], year: int, iteration: int) -> bool:
    path = get_asim_run_marker_path(output_dir, year, iteration)
    if path.exists():
        path.unlink()
        return True
    return False


def write_asim_run_marker(
    output_dir: Union[str, Path],
    year: int,
    iteration: int,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    path = get_asim_run_marker_path(output_dir, year, iteration)
    payload = {
        "year": year,
        "iteration": iteration,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta:
        payload["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


@dataclass
class ActivitySimPreprocessOutputs(StepOutputsBase):
    """
    Outputs from the ActivitySim preprocess step.

    Attributes
    ----------
    mutable_data_dir : Path
        ActivitySim mutable data directory.
    land_use_table : Path
        Land use input table path.
    households_table : Path
        Households input table path.
    persons_table : Path
        Persons input table path.
    omx_skims : Path, optional
        OMX skims input path when present.
    """

    primary_output_attr: ClassVar[str] = "mutable_data_dir"
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
    )
    record_keys: ClassVar[Dict[str, str]] = {
        "land_use_table": ASIM_LAND_USE_IN,
        "households_table": ASIM_HOUSEHOLDS_IN,
        "persons_table": ASIM_PERSONS_IN,
        "omx_skims": ASIM_OMX_SKIMS,
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "land_use_table": "ActivitySim land use input table",
        "households_table": "ActivitySim households input table",
        "persons_table": "ActivitySim persons input table",
        "omx_skims": "ActivitySim OMX skims input",
    }
    required_path_fields: ClassVar[Tuple[str, ...]] = (
        "mutable_data_dir",
        "land_use_table",
        "households_table",
        "persons_table",
    )
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("omx_skims",)

    mutable_data_dir: Path
    land_use_table: Path
    households_table: Path
    persons_table: Path
    omx_skims: Optional[Path] = None

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ActivitySimPreprocessOutputs":
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
        ActivitySimPreprocessOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        values: Dict[str, Any] = {}
        for field_name, record_key in cls.record_keys.items():
            path = artifact_to_path(mapping.get(record_key), workspace)
            if path is not None:
                values[field_name] = Path(path)
        values["mutable_data_dir"] = Path(workspace.get_asim_mutable_data_dir())
        return cls(**values)


@dataclass
class ActivitySimRunOutputs(StepOutputsBase):
    """
    Outputs from the ActivitySim run step.

    Attributes
    ----------
    output_dir : Path
        ActivitySim output directory.
    raw_outputs : dict
        Mapping of short_name to output path.
    raw_output_hashes : dict
        Mapping of short_name to known content hashes for raw outputs.
    """

    primary_output_attr: ClassVar[str] = "output_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)
    raw_output_hashes: Dict[str, str] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield run output records.
        """
        for key, path in self.raw_outputs.items():
            yield key, path, f"ActivitySim raw output: {key}"

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ActivitySimRunOutputs":
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
        ActivitySimRunOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        raw_outputs: Dict[str, Path] = {}
        raw_output_hashes: Dict[str, str] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
            if hasattr(value, "content_hash"):
                content_hash = getattr(value, "content_hash", None)
                if content_hash:
                    raw_output_hashes[key] = content_hash
        return cls(
            output_dir=Path(workspace.get_asim_output_dir()),
            raw_outputs=raw_outputs,
            raw_output_hashes=raw_output_hashes,
        )

    def to_record_store(self) -> RecordStore:
        """
        Convert outputs to a RecordStore with optional content hashes.
        """
        records = []
        for short_name, path, description in self._iter_record_items():
            records.append(
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                    content_hash=self.raw_output_hashes.get(short_name),
                )
            )
        return RecordStore(recordList=records)


@dataclass
class ActivitySimPostprocessOutputs(StepOutputsBase):
    """
    Outputs from the ActivitySim postprocess step.

    Attributes
    ----------
    usim_datastore_h5 : Path, optional
        Updated UrbanSim datastore path.
    asim_output_dir : Path
        ActivitySim output directory.
    processed_outputs : dict
        Mapping of short_name to postprocessed output path.
    processed_output_hashes : dict
        Mapping of short_name to known content hashes for copied outputs.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("asim_output_dir",)
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    usim_datastore_h5: Optional[Path]
    asim_output_dir: Path
    processed_outputs: Dict[str, Path] = field(default_factory=dict)
    processed_output_hashes: Dict[str, str] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield postprocessed output records.
        """
        for key, path in self.processed_outputs.items():
            yield key, path, f"ActivitySim output file: {key}"

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ActivitySimPostprocessOutputs":
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
        ActivitySimPostprocessOutputs
            Parsed outputs.
        """
        usim_path = None
        processed_outputs: Dict[str, Path] = {}
        processed_output_hashes: Dict[str, str] = {}
        allowed_outputs = set(ASIM_OUTPUT_KEY_MAP.values()) | set(
            ASIM_OUTPUT_KEY_MAP.keys()
        )
        if record_store is not None:
            for record in record_store.all_records():
                short_name = getattr(record, "short_name", "") or ""
                if short_name.startswith("usim_input_"):
                    usim_path = record.get_absolute_path(base_path=workspace.full_path)
                    continue
                normalized_name = normalize_asim_output_key(short_name)
                if short_name.startswith("asim_input_") or normalized_name in allowed_outputs:
                    record_path = record.get_absolute_path(
                        base_path=workspace.full_path
                    )
                    if record_path:
                        processed_outputs[normalized_name] = Path(record_path)
                        content_hash = getattr(record, "content_hash", None)
                        if content_hash:
                            processed_output_hashes[normalized_name] = content_hash
        return cls(
            usim_datastore_h5=Path(usim_path) if usim_path else None,
            asim_output_dir=Path(workspace.get_asim_output_dir()),
            processed_outputs=processed_outputs,
            processed_output_hashes=processed_output_hashes,
        )
