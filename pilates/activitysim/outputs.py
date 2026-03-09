from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union
import json
import re

from pilates.generic.records import RecordStore, FileRecord
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
)
from pilates.workflows.outputs_base import (
    OutputValidator,
    StepOutputsBase,
    ValidationContext,
    ValidationResult,
)

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


def _record_path(record: Any, workspace: "Workspace") -> Optional[Path]:
    """
    Resolve a RecordStore entry into an absolute filesystem path.
    """
    if record is None:
        return None
    get_absolute_path = getattr(record, "get_absolute_path", None)
    if callable(get_absolute_path):
        resolved = get_absolute_path(base_path=workspace.full_path)
        if resolved:
            return Path(resolved)
    file_path = getattr(record, "file_path", None)
    path = artifact_to_path(file_path if file_path is not None else record, workspace)
    if path is None:
        return None
    return Path(path)


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


class _UrbanSimToActivitySimBoundaryValidator:
    """
    Warn when ActivitySim preprocess table outputs drift from mutable-data layout.
    """

    name = "activitysim_preprocess_urbansim_boundary"
    level = "warning"

    def validate(
        self,
        outputs: "ActivitySimPreprocessOutputs",
        context: ValidationContext,
    ) -> list[ValidationResult]:
        upstream = context.upstream_outputs or {}
        if "urbansim_postprocess" not in upstream and "urbansim_run" not in upstream:
            return []

        mutable_data_dir = Path(outputs.mutable_data_dir)
        expected_fields = ("land_use_table", "households_table", "persons_table")
        results: list[ValidationResult] = []
        for field_name in expected_fields:
            path_value = getattr(outputs, field_name, None)
            if path_value is None:
                continue
            table_path = Path(path_value)
            if table_path.parent != mutable_data_dir:
                results.append(
                    ValidationResult(
                        message=(
                            f"{field_name} should be written under mutable_data_dir "
                            "for the UrbanSim->ActivitySim boundary."
                        ),
                        metadata={
                            "field": field_name,
                            "expected_parent": str(mutable_data_dir),
                            "actual_parent": str(table_path.parent),
                        },
                    )
                )
        return results


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
    validators: ClassVar[Tuple[OutputValidator, ...]] = (
        _UrbanSimToActivitySimBoundaryValidator(),
    )

    mutable_data_dir: Path
    land_use_table: Path
    households_table: Path
    persons_table: Path
    omx_skims: Optional[Path] = None
    input_hashes: Dict[str, str] = field(default_factory=dict)

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
        values: Dict[str, Any] = {}
        input_hashes: Dict[str, str] = {}
        records_by_key = {
            getattr(record, "short_name", None): record
            for record in (record_store.all_records() if record_store is not None else [])
        }
        for field_name, record_key in cls.record_keys.items():
            record = records_by_key.get(record_key)
            if record is None:
                continue
            record_path = _record_path(record, workspace)
            if record_path is not None:
                values[field_name] = record_path
            content_hash = getattr(record, "content_hash", None)
            if content_hash:
                input_hashes[record_key] = content_hash
        values["mutable_data_dir"] = Path(workspace.get_asim_mutable_data_dir())
        values["input_hashes"] = input_hashes
        return cls(**values)

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
                    content_hash=self.input_hashes.get(short_name),
                )
            )
        return RecordStore(recordList=records)


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
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs", "source_input_paths")
    output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)
    raw_output_hashes: Dict[str, str] = field(default_factory=dict)
    source_input_paths: Dict[str, Path] = field(default_factory=dict)
    source_input_hashes: Dict[str, str] = field(default_factory=dict)

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
        raw_outputs: Dict[str, Path] = {}
        raw_output_hashes: Dict[str, str] = {}
        for record in record_store.all_records() if record_store is not None else []:
            key = getattr(record, "short_name", None)
            if not key:
                continue
            record_path = _record_path(record, workspace)
            if record_path is None:
                continue
            raw_outputs[key] = record_path
            content_hash = getattr(record, "content_hash", None)
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

    def to_postprocess_record_store(self) -> RecordStore:
        """
        Convert outputs to a RecordStore and attach carried input metadata.
        """
        record_store = self.to_record_store()
        setattr(
            record_store,
            "activitysim_source_input_paths",
            {key: str(path) for key, path in self.source_input_paths.items()},
        )
        setattr(
            record_store,
            "activitysim_source_input_hashes",
            dict(self.source_input_hashes),
        )
        return record_store


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
    usim_datastore_key : str, optional
        Canonical coupler key for the next-iteration UrbanSim input datastore.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_path_fields: ClassVar[Tuple[str, ...]] = (
        "usim_datastore_h5",
        "asim_output_dir",
    )
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    usim_datastore_h5: Optional[Path]
    asim_output_dir: Optional[Path] = None
    processed_outputs: Dict[str, Path] = field(default_factory=dict)
    processed_output_hashes: Dict[str, str] = field(default_factory=dict)
    usim_datastore_key: Optional[str] = None

    def _resolved_usim_datastore_key(self) -> Optional[str]:
        if self.usim_datastore_key:
            return self.usim_datastore_key
        if self.usim_datastore_h5 is None:
            return None
        match = re.search(r"(\d{4})", self.usim_datastore_h5.name)
        if match:
            return f"usim_input_{match.group(1)}"
        return None

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield postprocessed output records.
        """
        usim_key = self._resolved_usim_datastore_key()
        if usim_key is not None and self.usim_datastore_h5 is not None:
            yield (
                usim_key,
                self.usim_datastore_h5,
                "New UrbanSim input data for next iteration",
            )
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
        usim_key = None
        processed_outputs: Dict[str, Path] = {}
        processed_output_hashes: Dict[str, str] = {}
        allowed_outputs = set(ASIM_OUTPUT_KEY_MAP.values()) | set(
            ASIM_OUTPUT_KEY_MAP.keys()
        )
        if record_store is not None:
            for record in record_store.all_records():
                short_name = getattr(record, "short_name", "") or ""
                if short_name.startswith("usim_input_"):
                    usim_key = short_name
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
            usim_datastore_key=usim_key,
        )
