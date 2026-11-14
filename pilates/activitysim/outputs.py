from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import json
from pilates.utils.io import is_land_use_enabled
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_H5,
)
from pilates.workflows.outputs_base import (
    OutputValidator,
    StepOutputsBase,
    ValidationContext,
    ValidationResult,
)


# Map legacy ActivitySim output keys (file stems) to namespaced keys.
ASIM_OUTPUT_KEY_MAP: Dict[str, str] = {
    "accessibility": "accessibility_asim_out",
    "beam_plans": "beam_plans_asim_out",
    "plans": "beam_plans_asim_out",
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

# ``plans`` was a historical PILATES configuration spelling.  Current
# ActivitySim publishes the generated BEAM plan table as ``beam_plans`` in the
# checkpoint final-pipeline directory.
ASIM_PIPELINE_TABLE_ALIASES: Dict[str, str] = {"plans": "beam_plans"}

ASIM_OPTIONAL_RUN_OUTPUT_KEYS: Tuple[str, ...] = (
    ASIM_OUTPUT_KEY_MAP["accessibility"],
    ASIM_OUTPUT_KEY_MAP["disaggregate_accessibility"],
    ASIM_OUTPUT_KEY_MAP["non_mandatory_tour_destination_accessibility"],
    ASIM_OUTPUT_KEY_MAP["person_windows"],
    ASIM_OUTPUT_KEY_MAP["proto_disaggregate_accessibility"],
    ASIM_OUTPUT_KEY_MAP["proto_households"],
    ASIM_OUTPUT_KEY_MAP["proto_persons"],
    ASIM_OUTPUT_KEY_MAP["proto_persons_merged"],
    ASIM_OUTPUT_KEY_MAP["proto_tours"],
    ASIM_OUTPUT_KEY_MAP["school_modeled_size"],
    ASIM_OUTPUT_KEY_MAP["school_destination_size"],
    ASIM_OUTPUT_KEY_MAP["school_shadow_prices"],
    ASIM_OUTPUT_KEY_MAP["workplace_destination_size"],
    ASIM_OUTPUT_KEY_MAP["workplace_location_accessibility"],
    ASIM_OUTPUT_KEY_MAP["workplace_modeled_size"],
    ASIM_OUTPUT_KEY_MAP["workplace_shadow_prices"],
)
ASIM_REQUIRED_RUN_OUTPUT_KEYS: Tuple[str, ...] = tuple(
    key
    for key in dict.fromkeys(ASIM_OUTPUT_KEY_MAP.values())
    if key not in set(ASIM_OPTIONAL_RUN_OUTPUT_KEYS)
)


def normalize_asim_output_key(key: str) -> str:
    return ASIM_OUTPUT_KEY_MAP.get(key, key)


def configured_asim_output_tables(settings: Any) -> Dict[str, str]:
    """Return physical final-pipeline table names indexed by semantic output key.

    ``output_tables.tables`` is the execution contract: it determines which
    final-pipeline tables ActivitySim actually creates.  The mapping preserves
    the physical source table name while exposing PILATES' stable semantic key
    (``beam_plans_asim_out``).  The legacy ``plans`` selector is resolved to
    ActivitySim's actual ``beam_plans`` final-pipeline directory.
    """

    activitysim = settings.activitysim
    output_tables = activitysim.output_tables
    if not isinstance(output_tables, Mapping):
        raise TypeError("activitysim.output_tables must be a mapping")
    tables = output_tables.get("tables", ())
    if not isinstance(tables, (list, tuple)):
        raise TypeError("activitysim.output_tables.tables must be a sequence")

    configured: Dict[str, str] = {}
    for table in tables:
        table_name = str(table)
        output_key = normalize_asim_output_key(table_name)
        if output_key not in ASIM_OUTPUT_KEY_MAP.values():
            continue
        pipeline_table_name = ASIM_PIPELINE_TABLE_ALIASES.get(table_name, table_name)
        existing = configured.setdefault(output_key, pipeline_table_name)
        if existing != pipeline_table_name:
            raise ValueError(
                "activitysim.output_tables.tables maps multiple source tables to "
                f"the same semantic output {output_key!r}: {existing!r}, {pipeline_table_name!r}"
            )
    return configured


def configured_asim_output_keys(settings: Any) -> Tuple[str, ...]:
    """Return the semantic output keys enabled by the active ActivitySim config."""

    return tuple(configured_asim_output_tables(settings))


def _normalize_activitysim_run_output_key(key: str) -> str:
    """
    Normalize raw ActivitySim runner keys into canonical workflow keys.

    The runner emits ``*_asim_out_temp`` filenames; the stable contract uses
    the normalized ``*_asim_out`` names.
    """
    if key.endswith("_asim_out_temp"):
        key = key[: -len("_temp")]
    return normalize_asim_output_key(key)


def _asim_run_marker_filename(year: int, iteration: int) -> str:
    return f".pilates_asim_run_success_year_{year}_iter_{iteration}.json"


def get_asim_run_marker_path(
    output_dir: Union[str, Path], year: int, iteration: int
) -> Path:
    return Path(output_dir) / _asim_run_marker_filename(year, iteration)


def has_asim_run_marker(
    output_dir: Union[str, Path], year: int, iteration: int
) -> bool:
    return get_asim_run_marker_path(output_dir, year, iteration).exists()


def clear_asim_run_marker(
    output_dir: Union[str, Path], year: int, iteration: int
) -> bool:
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
        upstream = (
            context.upstream_outputs if context.upstream_outputs is not None else {}
        )
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


@dataclass
class ActivitySimRunOutputs(StepOutputsBase):
    """
    Outputs from the ActivitySim run step.

    Attributes
    ----------
    output_dir : Path
        ActivitySim output directory.
    declared_outputs : tuple[str, ...]
        Canonical workflow-facing outputs derived from ``ASIM_OUTPUT_KEY_MAP``.
    raw_outputs : dict
        Mapping of short_name to output path.
    raw_output_hashes : dict
        Mapping of short_name to known content hashes for raw outputs.
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ("output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    validators: ClassVar[Tuple[OutputValidator, ...]] = ()
    output_dir: Path
    zarr_skims: Optional[Path] = None
    raw_outputs: Dict[str, Path] = field(default_factory=dict)
    raw_output_hashes: Dict[str, str] = field(default_factory=dict)
    source_input_paths: Dict[str, Path] = field(default_factory=dict)
    source_input_hashes: Dict[str, str] = field(default_factory=dict)


class _ActivitySimRunOutputsValidator:
    """
    Warn when raw outputs drift away from the canonical ActivitySim contract.

    The validator is advisory only. It keeps legacy or exploratory outputs
    visible without failing runs that already succeeded.
    """

    name = "activitysim_run_output_contract"
    level = "warning"

    def validate(
        self,
        outputs: "ActivitySimRunOutputs",
        context: ValidationContext,
    ) -> list[ValidationResult]:
        declared = set(ASIM_OUTPUT_KEY_MAP.values())
        results: list[ValidationResult] = []
        for raw_key in outputs.raw_outputs:
            normalized_key = _normalize_activitysim_run_output_key(raw_key)
            if normalized_key in declared:
                continue
            results.append(
                ValidationResult(
                    message=(
                        f"Unrecognized ActivitySim run output key '{raw_key}'. "
                        "It will remain in raw_outputs, but it is not part of the "
                        "stable canonical contract."
                    ),
                    metadata={
                        "raw_key": raw_key,
                        "normalized_key": normalized_key,
                        "declared_outputs": tuple(sorted(declared)),
                    },
                )
            )
        return results


ActivitySimRunOutputs.validators = (_ActivitySimRunOutputsValidator(),)


def _activitysim_postprocess_requires_usim_output(context: ValidationContext) -> bool:
    state = getattr(context, "state", None)
    if state is not None:
        stage_enum = getattr(state, "Stage", None)
        land_use_stage = getattr(stage_enum, "land_use", "land_use")
        is_enabled = getattr(state, "is_enabled", None)
        if callable(is_enabled):
            try:
                return bool(is_enabled(land_use_stage))
            except Exception:
                return False

    settings = getattr(context, "settings", None)
    if settings is not None and is_land_use_enabled(settings):
        return True

    return False


class _ActivitySimPostprocessOutputsValidator:
    """
    Require updated UrbanSim datastore outputs only when land use is active.
    """

    name = "activitysim_postprocess_expected_outputs"
    level = "error"

    def validate(
        self,
        outputs: "ActivitySimPostprocessOutputs",
        context: ValidationContext,
    ) -> list[ValidationResult]:
        if not _activitysim_postprocess_requires_usim_output(context):
            return []
        if outputs.usim_datastore_h5 is not None:
            return []
        return [
            ValidationResult(
                message=(
                    "usim_datastore_h5 is required when land use is "
                    "enabled because ActivitySim postprocess must update the "
                    "UrbanSim datastore for downstream steps."
                ),
                metadata={
                    "land_use_enabled": True,
                    "step_name": context.step_name,
                },
            )
        ]


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
        Canonical coupler key for the current UrbanSim datastore handoff
        (``usim_datastore_h5``).
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_path_fields: ClassVar[Tuple[str, ...]] = (
        "usim_datastore_h5",
        "asim_output_dir",
    )
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    validators: ClassVar[Tuple[OutputValidator, ...]] = (
        _ActivitySimPostprocessOutputsValidator(),
    )
    usim_datastore_h5: Optional[Path]
    asim_output_dir: Optional[Path] = None
    processed_outputs: Dict[str, Path] = field(default_factory=dict)
    processed_output_hashes: Dict[str, str] = field(default_factory=dict)
    usim_datastore_key: Optional[str] = None

    def _resolved_usim_datastore_key(self) -> Optional[str]:
        if self.usim_datastore_h5 is None:
            return None
        return USIM_DATASTORE_H5
