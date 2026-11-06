import re
from typing import Optional, Set, Tuple

from pilates.config.models import PilatesConfig
from pilates.generic.records import sanitize_artifact_key
from pilates.utils.settings_helper import get as get_setting

# Auto-generated from pilates/atlas/atlas_input; update if files change.
ATLAS_STATIC_INPUTS_COMMON = [
    "accessbility2017.RData",
    "accessbility_2015.RData",
    "cpi.csv",
    "modeaccessibility.csv",
    "psid_names.Rdat",
    "sfb Baseline.csv",
    "taz_to_tract_sfbay.csv",
    "vehicle_type_mapping_ESS_const_220_price.csv",
    "vehicle_type_mapping_baseline.csv",
    "vehicle_type_mapping_evMandForced2.csv",
]

ATLAS_STATIC_INPUTS_BY_SCENARIO = {
    "baseline": [
        "adopt/baseline/new_vehicle_annual_medians.csv",
        "adopt/baseline/new_vehicle_representative_vehicle.csv",
        "adopt/baseline/new_vehicles.csv",
        "adopt/baseline/new_vehicles_biannual_values_2017.csv",
        "adopt/baseline/new_vehicles_biannual_values_2019.csv",
        "adopt/baseline/new_vehicles_biannual_values_2021.csv",
        "adopt/baseline/new_vehicles_biannual_values_2023.csv",
        "adopt/baseline/new_vehicles_biannual_values_2025.csv",
        "adopt/baseline/new_vehicles_biannual_values_2027.csv",
        "adopt/baseline/new_vehicles_biannual_values_2029.csv",
        "adopt/baseline/new_vehicles_biannual_values_2031.csv",
        "adopt/baseline/new_vehicles_biannual_values_2033.csv",
        "adopt/baseline/new_vehicles_biannual_values_2035.csv",
        "adopt/baseline/new_vehicles_biannual_values_2037.csv",
        "adopt/baseline/new_vehicles_biannual_values_2039.csv",
        "adopt/baseline/new_vehicles_biannual_values_2041.csv",
        "adopt/baseline/new_vehicles_biannual_values_2043.csv",
        "adopt/baseline/new_vehicles_biannual_values_2045.csv",
        "adopt/baseline/new_vehicles_biannual_values_2047.csv",
        "adopt/baseline/new_vehicles_biannual_values_2049.csv",
        "adopt/baseline/used_vehicles.csv",
        "adopt/baseline/used_vehicles_2017.csv",
        "adopt/baseline/used_vehicles_2019.csv",
        "adopt/baseline/used_vehicles_2021.csv",
        "adopt/baseline/used_vehicles_2023.csv",
        "adopt/baseline/used_vehicles_2025.csv",
        "adopt/baseline/used_vehicles_2027.csv",
        "adopt/baseline/used_vehicles_2029.csv",
        "adopt/baseline/used_vehicles_2031.csv",
        "adopt/baseline/used_vehicles_2033.csv",
        "adopt/baseline/used_vehicles_2035.csv",
        "adopt/baseline/used_vehicles_2037.csv",
        "adopt/baseline/used_vehicles_2039.csv",
        "adopt/baseline/used_vehicles_2041.csv",
        "adopt/baseline/used_vehicles_2043.csv",
        "adopt/baseline/used_vehicles_2045.csv",
        "adopt/baseline/used_vehicles_2047.csv",
        "adopt/baseline/used_vehicles_2049.csv",
    ],
    "ess_cons": [
        "adopt/ess_cons/new_vehicle_annual_medians.csv",
        "adopt/ess_cons/new_vehicle_representative_vehicle.csv",
        "adopt/ess_cons/new_vehicles.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2017.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2019.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2021.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2023.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2025.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2027.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2029.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2031.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2033.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2035.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2037.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2039.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2041.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2043.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2045.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2047.csv",
        "adopt/ess_cons/new_vehicles_biannual_values_2049.csv",
        "adopt/ess_cons/used_vehicles.csv",
        "adopt/ess_cons/used_vehicles_2017.csv",
        "adopt/ess_cons/used_vehicles_2019.csv",
        "adopt/ess_cons/used_vehicles_2021.csv",
        "adopt/ess_cons/used_vehicles_2023.csv",
        "adopt/ess_cons/used_vehicles_2025.csv",
        "adopt/ess_cons/used_vehicles_2027.csv",
        "adopt/ess_cons/used_vehicles_2029.csv",
        "adopt/ess_cons/used_vehicles_2031.csv",
        "adopt/ess_cons/used_vehicles_2033.csv",
        "adopt/ess_cons/used_vehicles_2035.csv",
        "adopt/ess_cons/used_vehicles_2037.csv",
        "adopt/ess_cons/used_vehicles_2039.csv",
        "adopt/ess_cons/used_vehicles_2041.csv",
        "adopt/ess_cons/used_vehicles_2043.csv",
        "adopt/ess_cons/used_vehicles_2045.csv",
        "adopt/ess_cons/used_vehicles_2047.csv",
        "adopt/ess_cons/used_vehicles_2049.csv",
    ],
    "zev_mandate": [
        "adopt/zev_mandate/new_vehicle_annual_medians.csv",
        "adopt/zev_mandate/new_vehicle_representative_vehicle.csv",
        "adopt/zev_mandate/new_vehicles.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2017.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2019.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2021.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2023.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2025.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2027.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2029.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2031.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2033.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2035.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2037.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2039.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2041.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2043.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2045.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2047.csv",
        "adopt/zev_mandate/new_vehicles_biannual_values_2049.csv",
        "adopt/zev_mandate/used_vehicles.csv",
        "adopt/zev_mandate/used_vehicles_2017.csv",
        "adopt/zev_mandate/used_vehicles_2019.csv",
        "adopt/zev_mandate/used_vehicles_2021.csv",
        "adopt/zev_mandate/used_vehicles_2023.csv",
        "adopt/zev_mandate/used_vehicles_2025.csv",
        "adopt/zev_mandate/used_vehicles_2027.csv",
        "adopt/zev_mandate/used_vehicles_2029.csv",
        "adopt/zev_mandate/used_vehicles_2031.csv",
        "adopt/zev_mandate/used_vehicles_2033.csv",
        "adopt/zev_mandate/used_vehicles_2035.csv",
        "adopt/zev_mandate/used_vehicles_2037.csv",
        "adopt/zev_mandate/used_vehicles_2039.csv",
        "adopt/zev_mandate/used_vehicles_2041.csv",
        "adopt/zev_mandate/used_vehicles_2043.csv",
        "adopt/zev_mandate/used_vehicles_2045.csv",
        "adopt/zev_mandate/used_vehicles_2047.csv",
        "adopt/zev_mandate/used_vehicles_2049.csv",
    ],
}

_YEAR_SUFFIX = re.compile(r"_(\d{4})$")
_ATLAS_SCENARIO_ALIASES = {
    "baseline": "baseline",
    "ess_cons": "ess_cons",
    "ess_const_220_price": "ess_cons",
    "zev_mandate": "zev_mandate",
    "evmandforced2": "zev_mandate",
}


def atlas_selected_scenario(settings: object) -> Optional[str]:
    """
    Resolve the effective ATLAS scenario used for static input selection.

    Prefer ``atlas.adscen`` because that is the scenario argument passed to the
    ATLAS container. Fall back to ``atlas.scenario`` for configs that do not
    populate ``adscen``.
    """
    adscen = get_setting(settings, "atlas.adscen")
    scenario = get_setting(settings, "atlas.scenario")
    raw_value = adscen if adscen not in (None, "") else scenario
    if raw_value in (None, ""):
        return None
    return _ATLAS_SCENARIO_ALIASES.get(str(raw_value).strip().lower())


def atlas_run_years(settings: PilatesConfig) -> Set[int]:
    """
    Determine ATLAS run years from global run configuration.

    ATLAS sub-runs advance in biannual cadence between configured run bounds.
    This is intentionally independent of ``run.vehicle_ownership_freq``, which
    controls higher-level stage cadence, not ATLAS internal sub-year steps.
    """
    start_year = get_setting(settings, "run.start_year")
    end_year = get_setting(settings, "run.end_year")
    if start_year is None or end_year is None:
        return set()
    return set(range(int(start_year), int(end_year) + 1, 2))


def atlas_static_input_relpaths(settings: PilatesConfig) -> Tuple[str, ...]:
    """
    Deterministic static ATLAS input relpaths based on settings.

    Rules align with `AtlasPreprocessor.copy_data_to_mutable_location()`:
    - scenario-specific adopt folder selection by `settings.atlas.scenario`
    - one selected `vehicle_type_mapping_*` file for known scenarios
    - unknown/no scenario falls back to all mapping files
    """
    scenario_key = atlas_selected_scenario(settings)

    vehicle_type_mapping_by_scenario = {
        "baseline": "vehicle_type_mapping_baseline.csv",
        "ess_cons": "vehicle_type_mapping_ESS_const_220_price.csv",
        "zev_mandate": "vehicle_type_mapping_evMandForced2.csv",
    }
    selected_mapping = (
        vehicle_type_mapping_by_scenario.get(scenario_key) if scenario_key else None
    )
    relpaths = [
        relpath
        for relpath in ATLAS_STATIC_INPUTS_COMMON
        if not relpath.startswith("vehicle_type_mapping_")
    ]
    if selected_mapping is not None:
        relpaths.append(selected_mapping)
    else:
        relpaths.extend(
            relpath
            for relpath in ATLAS_STATIC_INPUTS_COMMON
            if relpath.startswith("vehicle_type_mapping_")
        )

    if scenario_key:
        relpaths.extend(ATLAS_STATIC_INPUTS_BY_SCENARIO.get(scenario_key, []))
    else:
        for values in ATLAS_STATIC_INPUTS_BY_SCENARIO.values():
            relpaths.extend(values)

    return tuple(dict.fromkeys(relpaths))


def atlas_static_input_keys(settings: PilatesConfig) -> Tuple[str, ...]:
    relpaths = atlas_static_input_relpaths(settings)

    keys = []
    for relpath in relpaths:
        rel_no_ext = relpath.rsplit(".", 1)[0]
        candidate = rel_no_ext.replace("\\", "/")
        key = sanitize_artifact_key(candidate) or candidate
        keys.append(key)
    return tuple(keys)


def atlas_static_input_keys_for_interval(
    settings: PilatesConfig,
    *,
    interval_start_year: int,
    interval_end_year: int,
) -> Tuple[str, ...]:
    """
    Return ATLAS static input keys with ADOPT year-stamped keys filtered to an interval.

    Notes
    -----
    - Filtering is applied only to keys under ``adopt/`` with a trailing
      ``_<year>`` suffix.
    - Non-ADOPT keys (for example ``accessbility2017``) are preserved so shared
      static resources remain available for all ATLAS runs.
    """
    lower = min(int(interval_start_year), int(interval_end_year))
    upper = max(int(interval_start_year), int(interval_end_year))

    filtered = []
    for key in atlas_static_input_keys(settings):
        normalized_key = key.replace("\\", "/")
        if normalized_key.startswith("adopt/"):
            match = _YEAR_SUFFIX.search(normalized_key)
            if match and not (lower <= int(match.group(1)) <= upper):
                continue
        filtered.append(key)
    return tuple(filtered)
