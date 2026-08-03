from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path
import re
import zlib

import numpy as np
import pandas as pd

from pilates.atlas.inputs import atlas_selected_scenario
from pilates.atlas.outputs import AtlasPostprocessOutputs, AtlasRunOutputs
from pilates.atlas.preprocessor import _resolve_atlas_h5_table_key
from pilates.config import PilatesConfig
from pilates.workspace import Workspace
from pilates.utils.coupler_helpers import artifact_to_existing_path
from pilates.utils.state_access import uses_input_datastore
from pilates.workflows.artifact_keys import (
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
)
from workflow_state import WorkflowState
from pilates.generic.postprocessor import GenericPostprocessor

logger = logging.getLogger(__name__)


_ATLAS_VEHICLE_TYPE_MAPPING_BY_SCENARIO = {
    "baseline": "vehicle_type_mapping_baseline.csv",
    "ess_cons": "vehicle_type_mapping_ESS_const_220_price.csv",
    "zev_mandate": "vehicle_type_mapping_evMandForced2.csv",
}

_ATLAS_SCENARIO_ALIASES = {
    "baseline": "baseline",
    "ess_cons": "ess_cons",
    "ess_const_220_price": "ess_cons",
    "zev_mandate": "zev_mandate",
    "evmandforced2": "zev_mandate",
}

_ATLAS_BODYTYPE_ALIASES = {
    "car": "car",
    "sedan": "car",
    "coupe": "car",
    "wagon": "car",
    "hatchback": "car",
    "convertible": "car",
    "sports_car": "car",
    "sportscar": "car",
    "minivan": "minvan",
    "mini_van": "minvan",
    "minvan": "minvan",
    "suv": "suv",
    "cuv": "suv",
    "truck": "truck",
    "pickup": "truck",
    "pickup_truck": "truck",
    "van": "van",
    "cargo_van": "van",
    "passenger_van": "van",
}

_ATLAS_FUEL_ALIASES = {
    "conv": "conv",
    "conventional": "conv",
    "gas": "conv",
    "gasoline": "conv",
    "ice": "conv",
    "diesel": "conv",
    "ev": "ev",
    "electric": "ev",
    "electricity": "ev",
    "bev": "ev",
    "hybrid": "hybrid",
    "phev": "phev",
    "plugin_hybrid": "phev",
    "plug_in_hybrid": "phev",
    "fuelcell": "fuelcell",
    "fuel_cell": "fuelcell",
    "hydrogen": "fuelcell",
    "cng": "cng",
}


def _normalize_lookup_key(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _settings_value(settings: Any, field_name: str) -> Optional[Any]:
    if isinstance(settings, dict):
        return settings.get(field_name)
    return getattr(settings, field_name, None)


def _normalized_atlas_scenario(settings: Any) -> str:
    scenario = atlas_selected_scenario(settings)
    if scenario is None:
        for field_name in ("atlas_vehicles_scenario", "atlas_scenario"):
            raw_value = _settings_value(settings, field_name)
            normalized = _ATLAS_SCENARIO_ALIASES.get(_normalize_lookup_key(raw_value))
            if normalized is not None:
                scenario = normalized
                break
    if scenario is None:
        logger.warning(
            "[AtlasPostprocessor] No ATLAS scenario configured for vehicle type mapping; defaulting to baseline."
        )
        return "baseline"
    return scenario


def resolve_atlas_vehicle_type_mapping_path(
    settings: Any, workspace: Optional[Workspace] = None
) -> Path:
    scenario = _normalized_atlas_scenario(settings)
    mapping_name = _ATLAS_VEHICLE_TYPE_MAPPING_BY_SCENARIO.get(scenario)
    if mapping_name is None:
        raise RuntimeError(
            f"Unsupported ATLAS scenario for vehicle type mapping: {scenario}"
        )

    candidates = []
    if workspace is not None:
        candidates.append(Path(workspace.get_atlas_mutable_input_dir()) / mapping_name)
    candidates.append(Path(__file__).resolve().parent / "atlas_input" / mapping_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "ATLAS vehicle type mapping file not found. Looked for: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _normalize_bodytype_for_mapping(value: Any) -> Optional[str]:
    normalized = _normalize_lookup_key(value)
    if normalized is None:
        return None
    return _ATLAS_BODYTYPE_ALIASES.get(normalized, normalized)


def _normalize_fuel_for_mapping(value: Any) -> Optional[str]:
    normalized = _normalize_lookup_key(value)
    if normalized is None:
        return None
    return _ATLAS_FUEL_ALIASES.get(normalized, normalized)


def _prepare_vehicle_type_mapping(mapping_csv_path: str) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_csv_path)
    mapping = mapping.copy()
    mapping["modelyear"] = mapping["modelyear"].astype(int)
    mapping["bodytype_key"] = mapping["bodytype"].map(_normalize_bodytype_for_mapping)
    mapping["fuel_key"] = mapping["adopt_fuel"].map(_normalize_fuel_for_mapping)
    mapping["sampleProbabilityWithinCategory"] = pd.to_numeric(
        mapping["sampleProbabilityWithinCategory"], errors="coerce"
    ).fillna(0.0)
    mapping = mapping.dropna(subset=["vehicleTypeId", "modelyear"])
    mapping = mapping.drop_duplicates(
        subset=["fuel_key", "bodytype_key", "modelyear", "vehicleTypeId"],
        keep="first",
    )
    return mapping


def _nearest_modelyear_subset(mapping: pd.DataFrame, modelyear: int) -> pd.DataFrame:
    nearest_year = (mapping["modelyear"] - modelyear).abs().sort_values().index[0]
    target_year = int(mapping.loc[nearest_year, "modelyear"])
    return mapping.loc[mapping["modelyear"] == target_year]


def _select_vehicle_type_candidates(
    mapping: pd.DataFrame,
    fuel_key: Optional[str],
    bodytype_key: Optional[str],
    modelyear: int,
) -> pd.DataFrame:
    attempts = (
        (fuel_key, bodytype_key, False),
        (fuel_key, bodytype_key, True),
        (fuel_key, None, False),
        (fuel_key, None, True),
        (None, bodytype_key, True),
        (None, None, True),
    )

    for candidate_fuel, candidate_bodytype, nearest_year in attempts:
        matched = mapping
        if candidate_fuel is not None:
            matched = matched.loc[matched["fuel_key"] == candidate_fuel]
        if candidate_bodytype is not None:
            matched = matched.loc[matched["bodytype_key"] == candidate_bodytype]
        if matched.empty:
            continue
        if nearest_year:
            matched = _nearest_modelyear_subset(matched, modelyear)
        else:
            matched = matched.loc[matched["modelyear"] == modelyear]
            if matched.empty:
                continue
        return matched.sort_values("vehicleTypeId").reset_index(drop=True)

    raise RuntimeError(
        "Unable to map ATLAS vehicle row to a BEAM vehicle type. "
        f"fuel={fuel_key} bodytype={bodytype_key} modelyear={modelyear}"
    )


def _sample_vehicle_type_ids(
    candidates: pd.DataFrame,
    sample_size: int,
    *,
    output_year: int,
    fuel_key: Optional[str],
    bodytype_key: Optional[str],
    modelyear: int,
) -> pd.Series:
    weights = candidates["sampleProbabilityWithinCategory"]
    if float(weights.sum()) <= 0:
        weights = None
    seed_value = zlib.crc32(
        f"{output_year}|{fuel_key}|{bodytype_key}|{modelyear}|{sample_size}".encode(
            "utf-8"
        )
    )
    return candidates.sample(
        n=sample_size,
        replace=True,
        weights=weights,
        random_state=seed_value,
    )["vehicleTypeId"].reset_index(drop=True)


def atlas_add_vehileTypeId(
    settings: dict,
    output_year: int,
    vehicles_csv_path: str,
    output_vehicles2_csv_path: str,
    mapping_csv_path: Optional[str] = None,
):
    """Add a 'vehicleTypeId' column to the ATLAS vehicles CSV.

    Reads the main ATLAS vehicles output, samples a BEAM-compatible
    ``vehicleTypeId`` from the scenario-specific mapping table, and writes the
    result to a new ``vehicles2_{year}.csv`` file.

    Args:
        settings (dict): The simulation settings.
        output_year (int): The forecast year being processed.
        vehicles_csv_path (str): Path to the input ATLAS vehicles CSV file.
        output_vehicles2_csv_path (str): Path to write the output CSV file to.
    """
    if not os.path.exists(vehicles_csv_path):
        logger.error(
            f"[AtlasPostprocessor] Missing input file for vehicleTypeId addition: {vehicles_csv_path}"
        )
        return

    df = pd.read_csv(vehicles_csv_path)
    df["modelyear"] = df["modelyear"].astype(int)
    mapping_path = (
        Path(mapping_csv_path)
        if mapping_csv_path is not None
        else resolve_atlas_vehicle_type_mapping_path(settings)
    )
    mapping = _prepare_vehicle_type_mapping(str(mapping_path))

    fuel_source = None
    if "adopt_fuel" in df.columns:
        fuel_source = df["adopt_fuel"]
    if "pred_power" in df.columns:
        fuel_source = (
            fuel_source.combine_first(df["pred_power"])
            if fuel_source is not None
            else df["pred_power"]
        )
    df["_fuel_key"] = (
        fuel_source.map(_normalize_fuel_for_mapping)
        if fuel_source is not None
        else None
    )
    df["_bodytype_key"] = (
        df["bodytype"].map(_normalize_bodytype_for_mapping)
        if "bodytype" in df.columns
        else None
    )
    df["vehicleTypeId"] = pd.Series(index=df.index, dtype="object")

    grouped = df.groupby(
        ["_fuel_key", "_bodytype_key", "modelyear"], sort=False, dropna=False
    )
    for (fuel_key, bodytype_key, modelyear), vehicle_subset in grouped:
        candidates = _select_vehicle_type_candidates(
            mapping,
            fuel_key=fuel_key,
            bodytype_key=bodytype_key,
            modelyear=int(modelyear),
        )
        sampled_ids = _sample_vehicle_type_ids(
            candidates,
            sample_size=len(vehicle_subset),
            output_year=output_year,
            fuel_key=fuel_key,
            bodytype_key=bodytype_key,
            modelyear=int(modelyear),
        )
        df.loc[vehicle_subset.index, "vehicleTypeId"] = sampled_ids.values

    df.drop(columns=["_fuel_key", "_bodytype_key"], inplace=True)
    df.to_csv(output_vehicles2_csv_path, index=False)


def _coerce_atlas_integer_column(
    values: pd.Series,
    *,
    column_name: str,
    path: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | (numeric % 1 != 0)
    if invalid.any():
        sample = values.loc[invalid].head(10).tolist()
        raise ValueError(
            "ATLAS household/vehicle contract violation: invalid integer values "
            f"for {column_name!r} in {path}; sample={sample}"
        )
    return numeric.astype("int64")


def validate_atlas_household_vehicle_contract(
    *,
    householdv_csv_path: str,
    vehicles_csv_path: str,
    atlas_subyear: int,
    state: Any,
) -> None:
    """Ensure ATLAS ownership totals and its vehicle inventory agree exactly."""
    householdv_columns = set(pd.read_csv(householdv_csv_path, nrows=0).columns)
    required_householdv_columns = {"household_id", "nvehicles", "year"}
    missing_householdv_columns = required_householdv_columns - householdv_columns
    if missing_householdv_columns:
        raise ValueError(
            "ATLAS household/vehicle contract violation: householdv output is "
            f"missing required columns {sorted(missing_householdv_columns)}; "
            f"householdv_path={householdv_csv_path}"
        )

    vehicle_columns = set(pd.read_csv(vehicles_csv_path, nrows=0).columns)
    required_vehicle_columns = {"household_id", "vehicle_id", "year"}
    missing_vehicle_columns = required_vehicle_columns - vehicle_columns
    if missing_vehicle_columns:
        raise ValueError(
            "ATLAS household/vehicle contract violation: vehicles output is "
            f"missing required columns {sorted(missing_vehicle_columns)}; "
            f"vehicles_path={vehicles_csv_path}"
        )

    householdv = pd.read_csv(
        householdv_csv_path,
        usecols=["household_id", "nvehicles", "year"],
    )
    household_ids = _coerce_atlas_integer_column(
        householdv["household_id"],
        column_name="householdv.household_id",
        path=householdv_csv_path,
    )
    expected_vehicle_counts = _coerce_atlas_integer_column(
        householdv["nvehicles"],
        column_name="householdv.nvehicles",
        path=householdv_csv_path,
    )
    if (expected_vehicle_counts < 0).any():
        sample = expected_vehicle_counts.loc[expected_vehicle_counts < 0].head(10).tolist()
        raise ValueError(
            "ATLAS household/vehicle contract violation: householdv.nvehicles "
            f"must be non-negative; householdv_path={householdv_csv_path} "
            f"sample={sample}"
        )
    if household_ids.duplicated().any():
        sample = household_ids.loc[household_ids.duplicated(keep=False)].head(10).tolist()
        raise ValueError(
            "ATLAS household/vehicle contract violation: householdv output has "
            f"duplicate household_id values; householdv_path={householdv_csv_path} "
            f"sample={sample}"
        )
    householdv_years = _coerce_atlas_integer_column(
        householdv["year"],
        column_name="householdv.year",
        path=householdv_csv_path,
    )
    if (householdv_years != atlas_subyear).any():
        sample = householdv_years.loc[householdv_years != atlas_subyear].head(10).tolist()
        raise ValueError(
            "ATLAS household/vehicle contract violation: householdv.year must "
            f"equal atlas_subyear={atlas_subyear}; householdv_path={householdv_csv_path} "
            f"sample={sample}"
        )

    expected_by_household = pd.Series(
        expected_vehicle_counts.to_numpy(),
        index=pd.Index(household_ids.to_numpy(), name="household_id"),
        dtype="int64",
    )
    vehicle_usecols = ["household_id", "vehicle_id", "year"]

    vehicle_counts = pd.Series(dtype="int64")
    unexpected_vehicle_rows = 0
    unexpected_household_samples: list[int] = []
    year_mismatch_samples: list[int] = []
    total_vehicle_rows = 0
    for vehicles_chunk in pd.read_csv(
        vehicles_csv_path,
        usecols=vehicle_usecols,
        chunksize=250_000,
    ):
        vehicle_household_ids = _coerce_atlas_integer_column(
            vehicles_chunk["household_id"],
            column_name="vehicles.household_id",
            path=vehicles_csv_path,
        )
        _coerce_atlas_integer_column(
            vehicles_chunk["vehicle_id"],
            column_name="vehicles.vehicle_id",
            path=vehicles_csv_path,
        )
        total_vehicle_rows += len(vehicles_chunk)
        unexpected_mask = ~vehicle_household_ids.isin(expected_by_household.index)
        unexpected_vehicle_rows += int(unexpected_mask.sum())
        if unexpected_mask.any() and len(unexpected_household_samples) < 10:
            unexpected_household_samples.extend(
                vehicle_household_ids.loc[unexpected_mask]
                .head(10 - len(unexpected_household_samples))
                .tolist()
            )
        vehicle_years = _coerce_atlas_integer_column(
            vehicles_chunk["year"],
            column_name="vehicles.year",
            path=vehicles_csv_path,
        )
        mismatched_year = vehicle_years != atlas_subyear
        if mismatched_year.any() and len(year_mismatch_samples) < 10:
            year_mismatch_samples.extend(
                vehicle_years.loc[mismatched_year]
                .head(10 - len(year_mismatch_samples))
                .tolist()
            )
        chunk_counts = vehicle_household_ids.value_counts(sort=False).astype("int64")
        vehicle_counts = vehicle_counts.add(chunk_counts, fill_value=0).astype("int64")

    observed_vehicle_counts = vehicle_counts.reindex(
        expected_by_household.index,
        fill_value=0,
    ).astype("int64")
    shortfall_mask = expected_by_household > observed_vehicle_counts
    surplus_mask = expected_by_household < observed_vehicle_counts
    if (
        unexpected_vehicle_rows
        or year_mismatch_samples
        or shortfall_mask.any()
        or surplus_mask.any()
    ):
        count_frame = pd.DataFrame(
            {
                "nvehicles": expected_by_household,
                "vehicle_row_count": observed_vehicle_counts,
            }
        )
        shortfall_samples = (
            count_frame.loc[shortfall_mask]
            .head(10)
            .reset_index()
            .to_dict(orient="records")
        )
        surplus_samples = (
            count_frame.loc[surplus_mask]
            .head(10)
            .reset_index()
            .to_dict(orient="records")
        )
        raise ValueError(
            "ATLAS household/vehicle contract violation: "
            f"atlas_interval_start_year={getattr(state, 'atlas_interval_start_year', None)} "
            f"atlas_subyear={atlas_subyear} "
            f"main_forecast_year={getattr(state, 'main_forecast_year', None)} "
            f"forecast_year={getattr(state, 'forecast_year', None)} "
            f"householdv_path={householdv_csv_path} vehicles_path={vehicles_csv_path} "
            f"households={len(expected_by_household)} expected_vehicle_rows={int(expected_by_household.sum())} "
            f"vehicle_rows={total_vehicle_rows} unexpected_vehicle_rows={unexpected_vehicle_rows} "
            f"sample_unexpected_households={unexpected_household_samples} "
            f"sample_vehicle_year_mismatches={year_mismatch_samples} "
            f"households_with_vehicle_shortfall={int(shortfall_mask.sum())} "
            f"sample_household_shortfalls={shortfall_samples} "
            f"households_with_vehicle_surplus={int(surplus_mask.sum())} "
            f"sample_household_surpluses={surplus_samples}"
        )


atlas_add_vehicleTypeId = atlas_add_vehileTypeId


def get_usim_datastore_fname(settings: PilatesConfig, io, year=None):
    """Construct the UrbanSim datastore filename based on settings.

    Args:
        settings (dict): The simulation settings.
        io (str): The direction of I/O, either 'input' or 'output'.
        year (int, optional): The simulation year. Required if io is 'output'.

    Returns:
        str: The formatted UrbanSim datastore filename.
    """
    if io == "output":
        datastore_name = settings.urbansim.output_file_template.format(year=year)
    elif io == "input":
        region = settings.run.region
        region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
        usim_base_fname = settings.urbansim.input_file_template
        datastore_name = usim_base_fname.format(region_id=region_id)
    else:
        raise ValueError(
            f"Invalid io parameter: {io}. Must be either 'input' or 'output'"
        )

    return datastore_name


def resolve_atlas_usim_datastore_path(
    settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
) -> Path:
    """Resolve the UrbanSim datastore ATLAS should read/update for this subrun."""
    explicit_value = getattr(state, "atlas_usim_datastore_h5", None)
    resolved_explicit = artifact_to_existing_path(explicit_value, workspace=workspace)
    if resolved_explicit:
        return Path(resolved_explicit)
    if isinstance(explicit_value, (str, os.PathLike)):
        return Path(os.fspath(explicit_value))

    usim_mutable_data_dir = workspace.get_usim_mutable_data_dir()
    if uses_input_datastore(state):
        usim_datastore_fname = get_usim_datastore_fname(settings, io="input")
    else:
        usim_datastore_fname = get_usim_datastore_fname(
            settings, io="output", year=state.forecast_year
        )
    return Path(usim_mutable_data_dir) / usim_datastore_fname


class AtlasPostprocessor(GenericPostprocessor):
    """
    ATLAS-specific postprocessor that consolidates all postprocessing steps for the ATLAS vehicle ownership model.
    This includes updating UrbanSim HDF5 with new vehicle ownership and adding vehicleTypeId to ATLAS vehicle outputs.
    Produces updated UrbanSim inputs and ATLAS vehicle outputs.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this postprocessor expects from the workflow.
        """
        usim_output_path = resolve_atlas_usim_datastore_path(settings, state, workspace)
        atlas_output_dir = workspace.get_atlas_output_dir()
        return {
            ATLAS_OUTPUT_DIR: (
                atlas_output_dir if os.path.exists(atlas_output_dir) else None
            ),
            USIM_DATASTORE_CURRENT_H5: (
                usim_output_path if usim_output_path.exists() else None
            ),
        }

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this postprocessor produces.

        Notes
        -----
        Output keys
            - ``usim_population_source_h5``: Updated UrbanSim datastore (H5)
              selected as the downstream population source.
            - ``atlas_vehicles2_output``: ATLAS vehicles2 CSV emitted for BEAM.
        Related docs
            - See `pilates/atlas/inputs.py` for the corresponding input
              descriptions used by ATLAS and downstream models.
        """
        usim_output_path = resolve_atlas_usim_datastore_path(settings, state, workspace)
        output_year = getattr(state, "year", getattr(state, "forecast_year", None))
        vehicles2_path = (
            os.path.join(
                workspace.get_atlas_output_dir(), f"vehicles2_{output_year}.csv"
            )
            if output_year is not None
            else None
        )
        return {
            USIM_POPULATION_SOURCE_H5: usim_output_path,
            **(
                {ATLAS_VEHICLES2_OUTPUT: vehicles2_path}
                if vehicles2_path is not None
                else {}
            ),
        }

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)

    def _postprocess(
        self,
        raw_outputs: AtlasRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
        usim_datastore_h5: Optional[Path] = None,
    ) -> AtlasPostprocessOutputs:
        """
        Postprocess ATLAS outputs: update UrbanSim HDF5 with new vehicle ownership,
        and add vehicleTypeId to ATLAS vehicle outputs. Handles provenance tracking.

        Args:
            raw_outputs (AtlasRunOutputs): The raw outputs from the ATLAS model run.
            workspace (Workspace): The workspace object for path management.
            model_run_hash (Optional[str]): The unique hash for this postprocessor run.

        Returns:
            AtlasPostprocessOutputs: Typed outputs for the generated files.
        """
        logger.info(
            "[AtlasPostprocessor] Starting postprocessing for ATLAS for year %s",
            self.state.current_year,
        )

        settings = self.state.full_settings
        output_year = self.state.forecast_year
        output_paths: Dict[str, Path] = {}
        atlas_hh_path = raw_outputs.raw_outputs.get(f"householdv_{output_year}")
        atlas_veh_path = raw_outputs.raw_outputs.get(f"vehicles_{output_year}")
        if atlas_hh_path is None or not atlas_hh_path.exists():
            raise RuntimeError(
                "ATLAS postprocess requires the current-year householdv CSV from "
                "AtlasRunOutputs"
            )
        if atlas_veh_path is None or not atlas_veh_path.exists():
            raise RuntimeError(
                "ATLAS postprocess requires the current-year vehicles CSV from "
                "AtlasRunOutputs"
            )

        # --- HDF5 Update and Provenance ---
        if usim_datastore_h5 is None:
            raise TypeError(
                "AtlasPostprocessor._postprocess requires usim_datastore_h5"
            )
        usim_h5_file = Path(usim_datastore_h5)
        if not usim_h5_file.exists():
            raise RuntimeError(
                "ATLAS postprocess requires the UrbanSim datastore H5 selected for "
                f"year {self.state.current_year}: {usim_h5_file}"
            )

        # Validate ATLAS's raw ownership outputs before generating any derived
        # vehicle inventory or mutating the population source.  BEAM requires
        # one staged vehicle row for every declared household vehicle.
        validate_atlas_household_vehicle_contract(
            householdv_csv_path=str(atlas_hh_path),
            vehicles_csv_path=str(atlas_veh_path),
            atlas_subyear=int(output_year),
            state=self.state,
        )

        # --- vehicleTypeId addition and Provenance ---
        atlas_veh2_file = os.path.join(
            workspace.get_atlas_output_dir(), f"vehicles2_{output_year}.csv"
        )

        atlas_add_vehileTypeId(
            settings, output_year, str(atlas_veh_path), atlas_veh2_file
        )
        logger.info(
            "[AtlasPostprocessor] Added vehicleTypeId to ATLAS vehicle outputs."
        )
        if not os.path.exists(atlas_veh2_file):
            raise RuntimeError(
                "ATLAS postprocess did not produce vehicles2 output for year "
                f"{output_year}"
            )
        output_paths[ATLAS_VEHICLES2_OUTPUT] = Path(atlas_veh2_file)

        # Perform the update only after the raw vehicle inventory has been
        # proved consistent with the ownership totals used to set ``cars``.
        updated_households_table = self.atlas_update_h5_vehicle(
            settings, output_year, str(usim_h5_file), str(atlas_hh_path)
        )
        if not updated_households_table:
            raise RuntimeError(
                "ATLAS postprocess failed to update UrbanSim HDF5 with vehicle ownership"
            )
        self._validate_updated_h5_table(
            h5_file_path=str(usim_h5_file),
            table_path=updated_households_table,
            output_year=output_year,
        )
        logger.info(
            "[AtlasPostprocessor] Updated UrbanSim HDF5 with new vehicle ownership."
        )
        updated_usim_h5 = Path(usim_h5_file)

        return AtlasPostprocessOutputs(
            atlas_output_dir=Path(workspace.get_atlas_output_dir()),
            usim_datastore_h5=updated_usim_h5,
            processed_outputs=output_paths,
        )

    def postprocess(
        self,
        raw_outputs: AtlasRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
        usim_datastore_h5: Optional[Path] = None,
    ) -> AtlasPostprocessOutputs:
        if not isinstance(raw_outputs, AtlasRunOutputs):
            raise TypeError("AtlasPostprocessor.postprocess expects AtlasRunOutputs")
        if usim_datastore_h5 is None:
            raise TypeError("AtlasPostprocessor.postprocess requires usim_datastore_h5")
        self.state.set_sub_stage_progress("postprocessor")
        return self._postprocess(
            raw_outputs,
            workspace,
            model_run_hash,
            usim_datastore_h5=usim_datastore_h5,
        )

    def atlas_update_h5_vehicle(
        self,
        settings: PilatesConfig,
        output_year: int,
        h5_file_path: str,
        household_v_csv_path: str,
    ) -> Optional[str]:
        """Update the UrbanSim HDF5 file with vehicle ownership data from ATLAS.

        Reads vehicle ownership data from the given CSV file and updates the 'cars'
        and 'hh_cars' columns in the 'households' table within the specified HDF5 file.

        Args:
            settings (dict): The simulation settings.
            output_year (int): The forecast year being processed.
            h5_file_path (str): The absolute path to the UrbanSim HDF5 file to update.
            household_v_csv_path (str): The absolute path to the ATLAS householdv CSV file.
        """
        if not os.path.exists(h5_file_path) or not os.path.exists(household_v_csv_path):
            logger.error(
                f"[AtlasPostprocessor] Missing input files for H5 update. H5: {h5_file_path}, CSV: {household_v_csv_path}"
            )
            return None

        logger.info(f"ATLAS is updating urbansim outputs for Year {output_year}")

        # Read and format ATLAS vehicle ownership output
        df = pd.read_csv(household_v_csv_path)
        df = (
            df.rename(columns={"nvehicles": "cars"})
            .set_index("household_id")
            .sort_index(ascending=True)
        )
        df["hh_cars"] = pd.cut(
            df["cars"],
            bins=[-0.5, 0.5, 1.5, np.inf],
            labels=["none", "one", "two or more"],
        )

        logger.info(f"Writing updated household vehicle info to h5 file {h5_file_path}")

        # Read original h5 files and update
        with pd.HDFStore(h5_file_path, mode="r+") as h5:
            # Keep write target resolution aligned with ATLAS preprocess so
            # subyear runs can update the nearest available year-scoped table.
            key = _resolve_atlas_h5_table_key(
                h5,
                year=output_year,
                table="households",
                is_start_year=self.state.is_start_year(),
            )

            try:
                olddf = h5[key]
            except KeyError:
                logger.error(f"Table '{key}' not found in HDF5 file: {h5_file_path}")
                return None

            olddf.index = olddf.index.astype(int)
            atlas_ids = pd.Index(df.index.astype(int))
            h5_ids = pd.Index(olddf.index.astype(int))
            missing_in_h5 = atlas_ids.difference(h5_ids)
            missing_in_atlas = h5_ids.difference(atlas_ids)

            if len(missing_in_h5) or len(missing_in_atlas):
                logger.error(
                    "ATLAS household_id mismatch found - NOT updating h5 datastore. "
                    "missing_in_h5=%s missing_in_atlas=%s sample_missing_in_h5=%s "
                    "sample_missing_in_atlas=%s",
                    len(missing_in_h5),
                    len(missing_in_atlas),
                    missing_in_h5.tolist()[:10],
                    missing_in_atlas.tolist()[:10],
                )
                return None

            olddf = olddf.reindex(atlas_ids)
            olddf["cars"] = df["cars"].values
            olddf["hh_cars"] = df["hh_cars"].values
            for col in olddf.columns:
                if olddf[col].dtype.name == "category":
                    logger.info(f"Converting column {col} from category to str")
                    olddf[col] = olddf[col].astype(str)
            h5[key] = olddf
            logger.info(f"ATLAS update h5 datastore table {key} - done")
            return key if str(key).startswith("/") else f"/{key}"

    @staticmethod
    def _validate_updated_h5_table(
        *,
        h5_file_path: str,
        table_path: str,
        output_year: int,
    ) -> None:
        normalized_table_path = (
            table_path if str(table_path).startswith("/") else f"/{table_path}"
        )
        with pd.HDFStore(h5_file_path, mode="r") as store:
            if normalized_table_path not in store:
                raise RuntimeError(
                    "ATLAS postprocess reported an updated UrbanSim H5 table that "
                    "is not present after write. "
                    f"h5_path={h5_file_path} year={output_year} "
                    f"table={normalized_table_path} available={sorted(store.keys())}"
                )
