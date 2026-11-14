from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple

from pilates.utils.settings_helper import get as get_setting
from pilates.workflows.outputs_base import (
    OutputValidator,
    StepOutputsBase,
    ValidationContext,
    ValidationResult,
)
from pilates.workflows.artifact_keys import (
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PLANS_OUT,
    LINKSTATS,
)


def _resolve_model_name(settings: Any, model_name: str) -> Optional[str]:
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None)
    return getattr(model_cfg, model_name, None)


def requires_omx_skim_output(settings: Any) -> bool:
    """Return whether a configured downstream model requires BEAM OMX skims."""
    return (
        bool(get_setting(settings, "write_skims_to_omx", False))
        or (_resolve_model_name(settings, "land_use") == "urbansim")
        or (_resolve_model_name(settings, "vehicle_ownership") == "atlas")
    )


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
        """
        Return error results if consumer-required BEAM skims are missing.

        Requires ``zarr_skims`` when ``activity_demand`` is ``"activitysim"``,
        and requires ``final_skims_omx`` when OMX export is enabled or UrbanSim
        or ATLAS is active. Skips checks if ``context.settings`` is ``None``.
        """
        settings = context.settings
        if settings is None:
            return []

        activity_demand_model = _resolve_model_name(settings, "activity_demand")
        land_use_model = _resolve_model_name(settings, "land_use")
        vehicle_ownership_model = _resolve_model_name(settings, "vehicle_ownership")
        write_omx = bool(get_setting(settings, "write_skims_to_omx", False))

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
        if requires_omx_skim_output(settings) and outputs.final_skims_omx is None:
            results.append(
                ValidationResult(
                    message=(
                        "final_skims_omx is required when OMX export is enabled or "
                        "UrbanSim or ATLAS is active."
                    ),
                    metadata={
                        "write_skims_to_omx": write_omx,
                        "land_use_model": land_use_model,
                        "vehicle_ownership_model": vehicle_ownership_model,
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

    required_path_fields: ClassVar[Tuple[str, ...]] = ("beam_mutable_data_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)
    beam_mutable_data_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)
    prepared_input_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


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

    def _latest_raw_output_for_prefix(self, prefix: str) -> Optional[Tuple[str, Path]]:
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
        return self._latest_publication_output_for_prefix(BEAM_EXPERIENCED_PLANS_XML)

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

    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("zarr_skims", "final_skims_omx")
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("split_events", "split_event_links")
    validators: ClassVar[Tuple[OutputValidator, ...]] = (
        _BeamPostprocessExpectedOutputsValidator(),
    )

    zarr_skims: Optional[Path] = None
    final_skims_omx: Optional[Path] = None
    split_events: Dict[str, Path] = field(default_factory=dict)
    split_event_links: Dict[str, Path] = field(default_factory=dict)


@dataclass
class BeamFullSkimOutputs(StepOutputsBase):
    """
    Outputs from the BEAM full-skim step.

    Attributes
    ----------
    full_skims : Path
        Full-skim output file produced by FullSkimsCreatorApp.
    final_skims_omx : Optional[Path]
        Final OMX skims output when full skims produces an OMX file.
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ("full_skims",)
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("final_skims_omx",)

    full_skims: Path
    final_skims_omx: Optional[Path] = None
