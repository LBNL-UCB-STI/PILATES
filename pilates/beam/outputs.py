from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.outputs_base import (
    OutputValidator,
    StepOutputsBase,
    ValidationContext,
    ValidationResult,
)
from pilates.workflows.artifact_keys import (
    BEAM_INPUT_CONFIG_ARCHIVED,
    BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
    BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
    BEAM_INPUT_PERSONS_ARCHIVED,
    BEAM_INPUT_PLANS_ARCHIVED,
    BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
    BEAM_INPUT_VEHICLES_ARCHIVED,
    BEAM_HOUSEHOLDS_IN,
    BEAM_FULL_SKIMS,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    BEAM_PERSONS_IN,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    ZARR_SKIMS,
)

if TYPE_CHECKING:
    from pilates.workspace import Workspace


def _resolve_model_name(settings: Any, model_name: str) -> Optional[str]:
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None)
    return getattr(model_cfg, model_name, None)


class _BeamPostprocessExpectedOutputsValidator:
    """
    Require BEAM postprocess skims only when downstream consumers need them.
    """

    name = "beam_postprocess_expected_outputs"
    level = "error"

    def validate(
        self,
        outputs: "BeamPostprocessOutputs",
        context: ValidationContext,
    ) -> list[ValidationResult]:
        settings = context.settings
        if settings is None:
            return []

        activity_demand_model = _resolve_model_name(settings, "activity_demand")
        land_use_model = _resolve_model_name(settings, "land_use")
        write_omx = bool(getattr(settings, "write_skims_to_omx", False))

        results: list[ValidationResult] = []
        if activity_demand_model == "activitysim" and outputs.zarr_skims is None:
            results.append(
                ValidationResult(
                    message=(
                        "zarr_skims is required when ActivitySim is enabled because "
                        "BEAM postprocess must merge into the shared skims store."
                    ),
                    metadata={"activity_demand_model": activity_demand_model},
                )
            )
        if (write_omx or land_use_model == "urbansim") and outputs.final_skims_omx is None:
            results.append(
                ValidationResult(
                    message=(
                        "final_skims_omx is required when OMX export is enabled or "
                        "UrbanSim is active."
                    ),
                    metadata={
                        "write_skims_to_omx": write_omx,
                        "land_use_model": land_use_model,
                    },
                )
            )
        return results


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
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
        LINKSTATS_WARMSTART,
        "vehicles_beam_in",
    )
    required_outputs: ClassVar[Tuple[str, ...]] = (
        BEAM_PLANS_IN,
        BEAM_HOUSEHOLDS_IN,
        BEAM_PERSONS_IN,
    )
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

    def to_record_store(self) -> RecordStore:
        """Convert typed preprocess outputs into a local ``RecordStore``."""
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                )
                for short_name, path, description in self._iter_record_items()
            ]
        )

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
    declared_outputs: ClassVar[Tuple[str, ...]] = (LINKSTATS, BEAM_PLANS_OUT)
    optional_outputs: ClassVar[Tuple[str, ...]] = (
        BEAM_OUTPUT_PLANS_XML,
        BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
        BEAM_EXPERIENCED_PLANS_XML,
        BEAM_INPUT_PLANS_ARCHIVED,
        BEAM_INPUT_HOUSEHOLDS_ARCHIVED,
        BEAM_INPUT_PERSONS_ARCHIVED,
        BEAM_INPUT_CONFIG_ARCHIVED,
        BEAM_INPUT_VEHICLES_ARCHIVED,
        BEAM_INPUT_LINKSTATS_WARMSTART_ARCHIVED,
        BEAM_INPUT_PLANS_WARMSTART_ARCHIVED,
        BEAM_INPUT_EXPERIENCED_PLANS_WARMSTART_ARCHIVED,
    )
    required_path_fields: ClassVar[Tuple[str, ...]] = ("beam_output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    beam_output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    @staticmethod
    def _record_rank(short_name: str, prefix: str) -> Optional[Tuple[int, int, int]]:
        if short_name == prefix:
            return (0, 0, 0)
        marker = f"{prefix}_"
        if not short_name.startswith(marker):
            return None
        tail = short_name[len(marker) :]
        parts = tail.split("_")
        if len(parts) < 2:
            return None
        try:
            year = int(parts[0])
            iteration = int(parts[1])
        except ValueError:
            return None
        if len(parts) == 2:
            return (year, iteration, 10_000)
        if len(parts) == 3 and parts[2].startswith("sub"):
            try:
                sub_iteration = int(parts[2][3:])
            except ValueError:
                return None
            return (year, iteration, sub_iteration)
        return None

    def _latest_raw_output_for_prefix(
        self, prefix: str
    ) -> Optional[Tuple[str, Path]]:
        best_key: Optional[str] = None
        best_path: Optional[Path] = None
        best_rank: Optional[Tuple[int, int, int]] = None
        for short_name, path in self.raw_outputs.items():
            rank = self._record_rank(short_name, prefix)
            if rank is None:
                continue
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_key = short_name
                best_path = path
        if best_key is None or best_path is None:
            return None
        return best_key, best_path

    @staticmethod
    def _publication_rank(
        short_name: str,
        prefix: str,
    ) -> Optional[Tuple[int, int]]:
        if short_name == prefix:
            return (0, 0)
        marker = f"{prefix}_"
        if not short_name.startswith(marker):
            return None
        tail = short_name[len(marker) :]
        parts = tail.split("_")
        if len(parts) < 2 or parts[-1].startswith("sub"):
            return None
        try:
            year = int(parts[-2])
            iteration = int(parts[-1])
        except ValueError:
            return None
        return (year, iteration)

    def _latest_publication_output_for_prefix(
        self, prefix: str
    ) -> Optional[Tuple[str, Path]]:
        best_key: Optional[str] = None
        best_path: Optional[Path] = None
        best_rank: Optional[Tuple[int, int]] = None
        for short_name, path in self.raw_outputs.items():
            rank = self._publication_rank(short_name, prefix)
            if rank is None:
                continue
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_key = short_name
                best_path = path
        if best_key is None or best_path is None:
            return None
        return best_key, best_path

    def promoted_linkstats_for_publication(self) -> Optional[Tuple[str, Path]]:
        return self._latest_publication_output_for_prefix(LINKSTATS)

    def promoted_plans_for_publication(self) -> Optional[Tuple[str, Path]]:
        return self._latest_publication_output_for_prefix(BEAM_PLANS_OUT)

    def promoted_output_plans_xml_for_publication(self) -> Optional[Tuple[str, Path]]:
        return self._latest_publication_output_for_prefix(BEAM_OUTPUT_PLANS_XML)

    def promoted_output_experienced_plans_xml_for_publication(
        self,
    ) -> Optional[Tuple[str, Path]]:
        return self._latest_publication_output_for_prefix(
            BEAM_OUTPUT_EXPERIENCED_PLANS_XML
        )

    def promoted_experienced_plans_xml_for_publication(
        self,
    ) -> Optional[Tuple[str, Path]]:
        return self._latest_publication_output_for_prefix(
            BEAM_EXPERIENCED_PLANS_XML
        )

    def iter_linkstats_parquet_outputs(self) -> Iterable[Tuple[str, Path]]:
        for key, path in self.raw_outputs.items():
            if key.startswith("linkstats_parquet_"):
                yield key, path

    def iter_unmodified_phys_sim_outputs(self) -> Iterable[Tuple[str, Path]]:
        for key, path in self.raw_outputs.items():
            if key.startswith(
                "linkstats_unmodified_phys_sim_iter_parquet_"
            ) or key.startswith("linkstats_unmodified_parquet__"):
                yield key, path

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield BEAM raw output records.
        """
        latest_linkstats = self._latest_raw_output_for_prefix(LINKSTATS)
        if latest_linkstats is None:
            # BEAM can emit only parquet linkstats in some configs. Promote the
            # latest parquet artifact to canonical `linkstats` so the step output
            # contract remains stable for downstream workflow steps.
            latest_linkstats = self._latest_raw_output_for_prefix("linkstats_parquet")
        if latest_linkstats is not None:
            _, path = latest_linkstats
            yield (
                LINKSTATS,
                path,
                "BEAM linkstats output for downstream runs",
            )
        latest_plans = self._latest_raw_output_for_prefix(BEAM_PLANS_OUT)
        if latest_plans is not None:
            _, path = latest_plans
            yield (
                BEAM_PLANS_OUT,
                path,
                "BEAM plans output for downstream runs",
            )
        latest_output_plans_xml = self._latest_publication_output_for_prefix(
            BEAM_OUTPUT_PLANS_XML
        )
        if latest_output_plans_xml is not None:
            _, path = latest_output_plans_xml
            yield (
                BEAM_OUTPUT_PLANS_XML,
                path,
                "BEAM output plans XML for downstream warm-start reuse",
            )
        latest_output_experienced_plans_xml = (
            self._latest_publication_output_for_prefix(
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML
            )
        )
        if latest_output_experienced_plans_xml is not None:
            _, path = latest_output_experienced_plans_xml
            yield (
                BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
                path,
                "BEAM output experienced plans XML for downstream warm-start reuse",
            )
        latest_experienced_plans_xml = self._latest_publication_output_for_prefix(
            BEAM_EXPERIENCED_PLANS_XML
        )
        if latest_experienced_plans_xml is not None:
            _, path = latest_experienced_plans_xml
            yield (
                BEAM_EXPERIENCED_PLANS_XML,
                path,
                "BEAM experienced plans XML for downstream warm-start reuse",
            )
        for key, path in self.raw_outputs.items():
            yield key, path, f"BEAM raw output: {key}"

    def to_record_store(self) -> RecordStore:
        """Convert typed runner outputs into a local ``RecordStore``."""
        return RecordStore(
            recordList=[
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                )
                for short_name, path, description in self._iter_record_items()
            ]
        )

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
        Zarr skims updated by BEAM when an ActivitySim-backed shared skims
        target is active.
    final_skims_omx : Path, optional
        Final OMX skims for downstream models. When present, it is treated as
        the primary output to log.
    split_events : dict
        Mapping of split BEAM events parquet short_names to paths.
    split_event_links : dict
        Mapping of derived link-level tables from split events.
    """

    primary_output_attr: ClassVar[str] = "zarr_skims"
    declared_outputs: ClassVar[Tuple[str, ...]] = (ZARR_SKIMS,)
    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_outputs: ClassVar[Tuple[str, ...]] = (ZARR_SKIMS, FINAL_SKIMS_OMX)
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("zarr_skims", "final_skims_omx")
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("split_events", "split_event_links")
    validators: ClassVar[Tuple[OutputValidator, ...]] = (
        _BeamPostprocessExpectedOutputsValidator(),
    )

    zarr_skims: Optional[Path] = None
    final_skims_omx: Optional[Path] = None
    split_events: Dict[str, Path] = field(default_factory=dict)
    split_event_links: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield all BEAM postprocess artifacts for downstream serialization.
        """
        if self.final_skims_omx is not None:
            yield (
                FINAL_SKIMS_OMX,
                self.final_skims_omx,
                "Final skims OMX for downstream models",
            )
        if self.zarr_skims is not None:
            yield (
                ZARR_SKIMS,
                self.zarr_skims,
                "Zarr skims updated with BEAM outputs",
            )
        for key, path in self.split_events.items():
            yield key, path, f"Split BEAM events parquet: {key}"
        for key, path in self.split_event_links.items():
            yield key, path, f"Derived split-event links parquet: {key}"

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
    declared_outputs: ClassVar[Tuple[str, ...]] = (BEAM_FULL_SKIMS,)
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
