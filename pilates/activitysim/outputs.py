from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    from pilates.workspace import Workspace


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
    record_keys: ClassVar[Dict[str, str]] = {
        "land_use_table": "land_use_asim_in",
        "households_table": "households_asim_in",
        "persons_table": "persons_asim_in",
        "omx_skims": "omx_skims",
    }
    record_descriptions: ClassVar[Dict[str, str]] = {
        "land_use_table": "ActivitySim land use input table",
        "households_table": "ActivitySim households input table",
        "persons_table": "ActivitySim persons input table",
        "omx_skims": "ActivitySim OMX skims input",
    }

    mutable_data_dir: Path
    land_use_table: Path
    households_table: Path
    persons_table: Path
    omx_skims: Optional[Path] = None

    def validate(self) -> None:
        """
        Validate that required outputs exist on disk.
        """
        assert (
            self.mutable_data_dir.exists()
        ), f"mutable_data_dir missing: {self.mutable_data_dir}"
        assert (
            self.land_use_table.exists()
        ), f"land_use_table missing: {self.land_use_table}"
        assert (
            self.households_table.exists()
        ), f"households_table missing: {self.households_table}"
        assert (
            self.persons_table.exists()
        ), f"persons_table missing: {self.persons_table}"
        if self.omx_skims is not None:
            assert self.omx_skims.exists(), f"omx_skims missing: {self.omx_skims}"

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
    """

    primary_output_attr: ClassVar[str] = "output_dir"
    output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        assert self.output_dir.exists(), f"output_dir missing: {self.output_dir}"
        for key, path in self.raw_outputs.items():
            if not path.exists():
                raise AssertionError(f"run output missing for {key}: {path}")

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
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
        return cls(
            output_dir=Path(workspace.get_asim_output_dir()),
            raw_outputs=raw_outputs,
        )


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
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    usim_datastore_h5: Optional[Path]
    asim_output_dir: Path
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        assert (
            self.asim_output_dir.exists()
        ), f"asim_output_dir missing: {self.asim_output_dir}"
        if self.usim_datastore_h5 is not None:
            assert (
                self.usim_datastore_h5.exists()
            ), f"usim_datastore_h5 missing: {self.usim_datastore_h5}"
        for key, path in self.processed_outputs.items():
            if not path.exists():
                raise AssertionError(f"postprocess output missing for {key}: {path}")

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
        allowed_outputs = {
            "beam_plans",
            "disaggregate_accessibility",
            "households",
            "joint_tour_participants",
            "land_use",
            "non_mandatory_tour_destination_accessibility",
            "person_windows",
            "persons",
            "proto_disaggregate_accessibility",
            "proto_households",
            "proto_persons",
            "proto_persons_merged",
            "proto_tours",
            "school_destination_size",
            "school_modeled_size",
            "school_shadow_prices",
            "tours",
            "trips",
            "workplace_destination_size",
            "workplace_location_accessibility",
            "workplace_modeled_size",
            "workplace_shadow_prices",
        }
        if record_store is not None:
            for record in record_store.all_records():
                short_name = getattr(record, "short_name", "") or ""
                if short_name.startswith("usim_input_"):
                    usim_path = record.get_absolute_path(base_path=workspace.full_path)
                    continue
                if short_name in allowed_outputs:
                    record_path = record.get_absolute_path(
                        base_path=workspace.full_path
                    )
                    if record_path:
                        processed_outputs[short_name] = Path(record_path)
        return cls(
            usim_datastore_h5=Path(usim_path) if usim_path else None,
            asim_output_dir=Path(workspace.get_asim_output_dir()),
            processed_outputs=processed_outputs,
        )
