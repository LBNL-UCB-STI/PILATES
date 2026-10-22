"""Shared helpers, facet builders, and contract validation for workflow steps.

Step input/output contracts and semantic artifact roles are defined in
``pilates/workflows/catalog.py`` and the native ``StepDefinition`` modules
under ``pilates/workflows/steps/``.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Set,
)

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_H5,
    USIM_INPUT_ARCHIVE_PREFIX,
    USIM_INPUT_MERGED_PREFIX,
    USIM_FORECAST_OUTPUT,
    ZARR_SKIMS,
)
from pilates.workflows.catalog import (
    WORKFLOW_STEP_SPECS,
    workflow_step_spec_for_step_name,
)


def _parse_prefixed_iteration_key(
    short_name: str, prefix: str
) -> Optional[Dict[str, Any]]:
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
    payload: Dict[str, Any] = {
        "year": year,
        "iteration": iteration,
    }
    if len(parts) > 2 and parts[2].startswith("sub"):
        try:
            payload["beam_sub_iteration"] = int(parts[2][3:])
        except ValueError:
            pass
    return payload


def _beam_artifact_facets(short_name: str) -> Optional[Dict[str, Any]]:
    for prefix, family in (
        ("events_parquet", "events_parquet"),
        ("raw_od_skims", "raw_od_skims"),
        ("raw_od_skims_zarr", "raw_od_skims_zarr"),
        ("linkstats_parquet", "linkstats_parquet"),
    ):
        parsed = _parse_prefixed_iteration_key(short_name, prefix)
        if parsed is not None:
            return {"artifact_family": family, **parsed}

    if short_name.startswith("linkstats_unmodified_parquet__"):
        tokens = short_name.split("__")
        payload: Dict[str, Any] = {
            "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet"
        }
        for token in tokens[1:]:
            if token.startswith("y"):
                try:
                    payload["year"] = int(token[1:])
                except ValueError:
                    return None
            elif token.startswith("i"):
                try:
                    payload["iteration"] = int(token[1:])
                except ValueError:
                    return None
            elif token.startswith("phys_sim_iter"):
                try:
                    payload["phys_sim_iteration"] = int(token[len("phys_sim_iter") :])
                except ValueError:
                    return None
            elif token.startswith("beam_sub_iter"):
                try:
                    payload["beam_sub_iteration"] = int(token[len("beam_sub_iter") :])
                except ValueError:
                    return None
        if {"year", "iteration", "phys_sim_iteration"} <= set(payload.keys()):
            return payload
        return None

    return None


def _beam_log_facet_meta(short_name: str) -> Dict[str, Any]:
    facet = _beam_artifact_facets(short_name)
    if not facet:
        return {}
    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _beam_postprocess_split_facet_meta(short_name: str) -> Dict[str, Any]:
    if short_name.startswith("events_parquet_") and "_type_" in short_name:
        head, event_type = short_name.split("_type_", 1)
        parsed = _parse_prefixed_iteration_key(head, "events_parquet")
        if parsed:
            return {
                "facet": {
                    "artifact_family": "events_parquet_split",
                    "event_type": event_type,
                    **parsed,
                },
                "facet_schema_version": "v1",
                "facet_index": True,
            }
    if short_name.startswith("path_traversal_links_"):
        parsed = _parse_prefixed_iteration_key(short_name, "path_traversal_links")
        if parsed:
            return {
                "facet": {
                    "artifact_family": "path_traversal_links",
                    **parsed,
                },
                "facet_schema_version": "v1",
                "facet_index": True,
            }
    return {}


def _activitysim_output_facet_meta(
    short_name: str,
    *,
    year: int,
    iteration: int,
) -> Dict[str, Any]:
    family = None
    snapshot_meta: Dict[str, Any] = {}
    if short_name.endswith("_asim_out"):
        family = short_name[: -len("_asim_out")]
    elif short_name.startswith("asim_input_") and short_name.endswith("_archived"):
        family = "asim_input_archived"
        input_name = short_name.removeprefix("asim_input_").removesuffix("_archived")
        source_role_map = {
            "households_csv": ASIM_HOUSEHOLDS_IN,
            "persons_csv": ASIM_PERSONS_IN,
            "land_use_csv": ASIM_LAND_USE_IN,
            "skims_omx": ASIM_OMX_SKIMS,
            "skims_zarr": ZARR_SKIMS,
        }
        snapshot_meta = {
            "source_role": source_role_map.get(input_name, input_name),
            "snapshot_role": f"asim_input_{input_name}",
            "snapshot_reason": "exact_rewind",
            "storage_event": "snapshot_copy",
        }
    elif short_name == ZARR_SKIMS:
        family = "zarr_skims"
    if family is None:
        return {}
    return {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": year,
            "iteration": iteration,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _urbansim_output_facet_meta(
    short_name: str,
    *,
    forecast_year: int,
) -> Dict[str, Any]:
    snapshot_meta: Dict[str, Any] = {}
    if short_name.startswith(USIM_INPUT_ARCHIVE_PREFIX):
        family = "usim_input_archive"
        snapshot_meta = {
            "source_role": USIM_DATASTORE_H5,
            "snapshot_role": "usim_input_archive",
            "snapshot_reason": "pre_merge_input",
            "storage_event": "snapshot_move",
        }
    elif short_name.startswith(USIM_INPUT_MERGED_PREFIX):
        family = "usim_input_merged"
        snapshot_meta = {
            "source_role": "usim_input_archive",
            "snapshot_role": "usim_input_merged",
            "snapshot_reason": "post_merge_handoff",
            "storage_event": "merged_h5_output",
        }
    elif short_name == USIM_FORECAST_OUTPUT:
        family = "usim_forecast_output"
    elif short_name == USIM_DATASTORE_H5:
        family = "usim_datastore_h5"
    elif short_name == USIM_DATASTORE_BASE_H5:
        family = "usim_datastore_base_h5"
    else:
        return {}
    return {
        "facet": {
            "artifact_family": family,
            **snapshot_meta,
            "year": forecast_year,
        },
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _atlas_artifact_facet_meta(
    short_name: str,
    *,
    run_scenario: Optional[str],
    forecast_year: int,
    artifact_family: str = "atlas_input",
) -> Dict[str, Any]:
    key = short_name.replace("\\", "/")
    key_compact = key.replace("/", "_")

    input_group = "global"
    parsed_scenario = None
    input_year = None

    if key.startswith("adopt/") or key_compact.startswith("adopt_"):
        input_group = "adopt"
        if key.startswith("adopt/"):
            parts = key.split("/")
            if len(parts) >= 2:
                parsed_scenario = parts[1]
        else:
            parts = key_compact.split("_")
            if len(parts) >= 2:
                parsed_scenario = parts[1]
    elif key_compact.startswith("vehicle_type_mapping_"):
        input_group = "vehicle_type_mapping"
        if "baseline" in key_compact:
            parsed_scenario = "baseline"
        elif "evMandForced2" in key_compact:
            parsed_scenario = "zev_mandate"
        elif "ESS_const_220_price" in key_compact:
            parsed_scenario = "ess_cons"
    elif key_compact.startswith("atlas_vehicles2"):
        input_group = "vehicles2"
    elif key_compact.startswith("usim_"):
        input_group = "usim"

    tail = key_compact.rsplit("_", 1)
    if len(tail) == 2 and len(tail[1]) == 4 and tail[1].isdigit():
        input_year = int(tail[1])

    facet: Dict[str, Any] = {
        "artifact_family": artifact_family,
        "input_group": input_group,
        "forecast_year": forecast_year,
    }
    scenario_value = parsed_scenario or run_scenario
    if scenario_value:
        facet["scenario"] = str(scenario_value)
    if input_year is not None:
        facet["input_year"] = input_year

    return {
        "facet": facet,
        "facet_schema_version": "v1",
        "facet_index": True,
    }


def _declared_step_model(step_func: Callable[..., Any]) -> Optional[str]:
    meta = getattr(step_func, "__consist_step__", None)
    model = getattr(meta, "model", None)
    if isinstance(model, str) and model:
        return model
    return None


def _declared_step_name(step_func: Callable[..., Any]) -> Optional[str]:
    """Return the registered native identity for a decorated step callable."""
    for spec in WORKFLOW_STEP_SPECS:
        if spec.definition.function is step_func:
            return spec.step_name
    return None


def validate_workflow_step_contracts(
    *,
    declared_steps: Optional[Iterable[Callable[..., Any]]] = None,
    allow_untracked_declared: Optional[Set[str]] = None,
    require_all_tracked_declared: bool = True,
) -> None:
    """Validate native decorated steps against the catalog contract."""
    if declared_steps is None:
        return

    errors: list[str] = []
    declared_by_name: Dict[str, Callable[..., Any]] = {}
    for step in declared_steps:
        model = _declared_step_model(step)
        if model is None:
            errors.append(
                "Declared step callable is missing __consist_step__.model metadata"
            )
            continue
        name = _declared_step_name(step) or model
        if name in declared_by_name:
            errors.append(f"Duplicate declared step identity: {name}")
            continue
        declared_by_name[name] = step

    expected_names = {spec.step_name for spec in WORKFLOW_STEP_SPECS}
    declared_names = set(declared_by_name)
    if require_all_tracked_declared:
        missing = expected_names - declared_names
        if missing:
            errors.append(
                "Native definitions missing from declared steps: "
                + ", ".join(sorted(missing))
            )
    allowed = {"postprocessing"} | set(allow_untracked_declared or ())
    unexpected = declared_names - expected_names - allowed
    if unexpected:
        errors.append("Undeclared native step models: " + ", ".join(sorted(unexpected)))

    for name, step in declared_by_name.items():
        spec = workflow_step_spec_for_step_name(name)
        if spec is None:
            continue
        step_meta = getattr(step, "__consist_step__", None)
        schema_outputs = (
            getattr(step_meta, "schema_outputs", ()) if step_meta is not None else ()
        )
        if schema_outputs is not None and not set(spec.output_keys).issubset(
            set(schema_outputs)
        ):
            errors.append(
                f"Step '{name}' metadata schema outputs omit catalog outputs "
                + ", ".join(sorted(set(spec.output_keys) - set(schema_outputs)))
            )

    if errors:
        raise RuntimeError(
            "Workflow step contract validation failed:\n- " + "\n- ".join(errors)
        )
