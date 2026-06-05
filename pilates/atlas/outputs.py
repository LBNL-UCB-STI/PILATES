from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.utils.usim_h5 import resolve_usim_h5_table_key
from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.outputs_base import StepOutputsBase, ValidationContext

if TYPE_CHECKING:
    pass


@dataclass
class AtlasPreprocessOutputs(StepOutputsBase):
    """
    Outputs from the ATLAS preprocess step.

    Attributes
    ----------
    atlas_mutable_input_dir : Path
        ATLAS mutable input directory prepared for the runner.
    prepared_inputs : dict
        Mapping of input short_name to prepared input path.
    """

    primary_output_attr: ClassVar[str] = "atlas_mutable_input_dir"
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        "atlas_households_csv",
        "atlas_blocks_csv",
        "atlas_persons_csv",
        "atlas_residential_csv",
        "atlas_jobs_csv",
    )
    required_outputs: ClassVar[Tuple[str, ...]] = declared_outputs
    required_path_fields: ClassVar[Tuple[str, ...]] = ("atlas_mutable_input_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)
    atlas_mutable_input_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)
    prepared_input_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield prepared ATLAS input records.
        """
        for key, path in self.prepared_inputs.items():
            yield key, path, f"ATLAS prepared input: {key}"

    def validate(self, context: Optional[ValidationContext] = None) -> None:
        super().validate(context)
        state = getattr(context, "state", None) if context is not None else None
        if state is None:
            return

        year = getattr(state, "year", getattr(state, "current_year", None))
        start_year = getattr(state, "start_year", None)
        needs_grave_csv = (
            year is not None and start_year is not None and int(year) > int(start_year)
        )

        if needs_grave_csv and "atlas_grave_csv" not in self.prepared_inputs:
            raise AssertionError(
                "AtlasPreprocessOutputs must include atlas_grave_csv when atlas year exceeds the global start_year."
            )
        _validate_atlas_households_match_selected_h5(self, context)


def _validate_atlas_households_match_selected_h5(
    outputs: AtlasPreprocessOutputs,
    context: Optional[ValidationContext],
) -> None:
    if context is None:
        return
    state = context.state
    workspace = context.workspace
    if state is None or workspace is None:
        return

    households_csv = outputs.prepared_inputs.get("atlas_households_csv")
    selected_h5 = getattr(state, "atlas_usim_datastore_h5", None)
    if households_csv is None or selected_h5 is None:
        return
    h5_path = artifact_to_existing_path(selected_h5, workspace=workspace)
    if h5_path is None:
        return

    year = getattr(state, "year", getattr(state, "current_year", None))
    if year is None:
        return

    atlas_households = pd.read_csv(households_csv, usecols=["household_id"])
    with pd.HDFStore(h5_path, mode="r") as store:
        h5_table_path = resolve_usim_h5_table_key(
            store,
            year=int(year),
            table="households",
        )
        h5_households = store[h5_table_path]

    atlas_ids = pd.Index(
        pd.to_numeric(atlas_households["household_id"], errors="raise").astype("int64")
    )
    h5_ids = pd.Index(h5_households.index.astype("int64"))
    missing_in_h5 = atlas_ids.difference(h5_ids)
    missing_in_atlas = h5_ids.difference(atlas_ids)
    if len(missing_in_h5) or len(missing_in_atlas):
        raise AssertionError(
            "ATLAS preprocess households CSV does not match the selected UrbanSim "
            "H5 household table. This usually means cached/restored ATLAS inputs "
            "belong to a different population universe. "
            f"year={year} h5_path={h5_path} h5_table={h5_table_path} "
            f"atlas_households_csv={households_csv} "
            f"missing_in_h5={len(missing_in_h5)} "
            f"missing_in_atlas={len(missing_in_atlas)} "
            f"sample_missing_in_h5={missing_in_h5.tolist()[:10]} "
            f"sample_missing_in_atlas={missing_in_atlas.tolist()[:10]}"
        )


@dataclass
class AtlasRunOutputs(StepOutputsBase):
    """
    Outputs from the ATLAS run step.

    Attributes
    ----------
    atlas_output_dir : Path
        ATLAS output directory for the run.
    raw_outputs : dict
        Mapping of short_name to raw output path.
    """

    primary_output_attr: ClassVar[str] = "atlas_output_dir"
    required_output_families: ClassVar[Tuple[str, ...]] = (
        "householdv_{year}",
        "vehicles_{year}",
    )
    required_path_fields: ClassVar[Tuple[str, ...]] = ("atlas_output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    atlas_output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield ATLAS raw output records.
        """
        for key, path in self.raw_outputs.items():
            yield key, path, f"ATLAS raw output: {key}"

    def validate(self, context: Optional[Any] = None) -> None:
        super().validate(context)
        has_households = any(key.startswith("householdv_") for key in self.raw_outputs)
        has_vehicles = any(key.startswith("vehicles_") for key in self.raw_outputs)
        if not has_households or not has_vehicles:
            raise AssertionError(
                "AtlasRunOutputs must include current-year householdv_* and vehicles_* outputs."
            )


@dataclass
class AtlasPostprocessOutputs(StepOutputsBase):
    """
    Outputs from the ATLAS postprocess step.

    Attributes
    ----------
    atlas_output_dir : Path
        ATLAS output directory after postprocessing.
    usim_datastore_h5 : Path, optional
        Updated UrbanSim datastore after ATLAS postprocessing.
    processed_outputs : dict
        Mapping of short_name to postprocessed output path.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    declared_outputs: ClassVar[Tuple[str, ...]] = (
        USIM_POPULATION_SOURCE_H5,
        ATLAS_VEHICLES2_OUTPUT,
    )
    required_outputs: ClassVar[Tuple[str, ...]] = declared_outputs
    required_path_fields: ClassVar[Tuple[str, ...]] = ("atlas_output_dir",)
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    atlas_output_dir: Path
    usim_datastore_h5: Optional[Path]
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield ATLAS postprocessed output records.
        """
        if self.usim_datastore_h5 is not None:
            yield (
                USIM_POPULATION_SOURCE_H5,
                self.usim_datastore_h5,
                f"ATLAS postprocess output: {USIM_POPULATION_SOURCE_H5}",
            )
        for key, path in self.processed_outputs.items():
            yield key, path, f"ATLAS postprocess output: {key}"

    def validate(self, context: Optional[Any] = None) -> None:
        super().validate(context)
        if self.usim_datastore_h5 is None:
            raise AssertionError(
                "AtlasPostprocessOutputs must include the updated UrbanSim datastore H5."
            )
        if ATLAS_VEHICLES2_OUTPUT not in self.processed_outputs:
            raise AssertionError(
                f"AtlasPostprocessOutputs must include {ATLAS_VEHICLES2_OUTPUT}."
            )
