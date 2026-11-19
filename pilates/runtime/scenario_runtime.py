from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

from pilates.config import PilatesConfig
from pilates.workflows.catalog import schema_step_specs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface


def coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_scenario_id(settings: PilatesConfig) -> str:
    run_cfg = getattr(settings, "run", None)
    candidates = [
        getattr(run_cfg, "scenario", None),
        getattr(run_cfg, "output_run_name", None),
    ]
    for candidate in candidates:
        text = str(candidate).strip() if candidate is not None else ""
        if text:
            return text
    return "unknown_scenario"


def resolve_seed(settings: PilatesConfig) -> Optional[int]:
    candidates = [
        getattr(getattr(settings, "activitysim", None), "random_seed", None),
        getattr(getattr(settings, "run", None), "seed", None),
        getattr(settings, "seed", None),
    ]
    for candidate in candidates:
        value = coerce_int(candidate)
        if value is not None:
            return value
    return None


def merge_tag_list(existing: Any, additions: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        for value in existing:
            text = str(value).strip()
            if text and text not in seen:
                merged.append(text)
                seen.add(text)
    for value in additions:
        text = str(value).strip()
        if text and text not in seen:
            merged.append(text)
            seen.add(text)
    return merged


def facet_to_mapping(facet: Any) -> Dict[str, Any]:
    if isinstance(facet, Mapping):
        return dict(facet)
    model_dump = getattr(facet, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def merge_epoch_facet(
    *,
    existing: Any,
    scenario_id: str,
    seed: Optional[int],
    model: Optional[str],
    year: Optional[int],
    iteration: Optional[int],
) -> Dict[str, Any]:
    merged = facet_to_mapping(existing)
    merged["scenario_id"] = scenario_id
    if seed is not None:
        merged["seed"] = seed
    if model:
        merged["model"] = model
    if year is not None:
        merged["year"] = year
    if iteration is not None:
        merged["iteration"] = iteration
    return merged


DEFAULT_CACHE_EPOCH = 2


def resolve_cache_epoch(settings: PilatesConfig) -> int:
    value = getattr(getattr(settings, "run", None), "cache_epoch", DEFAULT_CACHE_EPOCH)
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_CACHE_EPOCH


def build_schema_steps() -> List[Callable[..., Any]]:
    """Return catalog-ordered committed native Consist callables for schema checks."""
    return [spec.definition.function for spec in schema_step_specs()]


def filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    *,
    surface: "EnabledWorkflowSurface",
    include_optional: bool = True,
) -> List[Callable[..., Any]]:
    enabled_step_names = surface.enabled_schema_step_names(
        include_optional=include_optional
    )
    enabled_step_functions = {
        spec.definition.function
        for spec in schema_step_specs(include_optional=include_optional)
        if spec.step_name in enabled_step_names
    }
    return [step_func for step_func in steps if step_func in enabled_step_functions]


def _required_output_keys_for_surface(
    *,
    surface: "EnabledWorkflowSurface",
    state: Any = None,
) -> List[str]:
    required_output_keys: List[str] = []
    seen = set()

    min_order = _required_output_restart_frontier_order(
        surface=surface,
        state=state,
    )
    for spec in schema_step_specs(include_optional=True):
        if min_order is not None and spec.order < min_order:
            continue
        step_name = spec.step_name
        if not surface.step_enabled(step_name, include_optional=False):
            continue
        step_surface = surface.step_surface(step_name)
        if step_surface is None:
            continue
        for key in step_surface.required_output_keys:
            if key in seen:
                continue
            required_output_keys.append(key)
            seen.add(key)
    return required_output_keys


def _stage_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, str) and raw_value:
        return raw_value
    if isinstance(value, str) and value:
        return value
    return None


def _state_stage_matches(state: Any, value: Any, stage_name: str) -> bool:
    stage_enum = getattr(state, "Stage", None)
    expected = getattr(stage_enum, stage_name, None) if stage_enum is not None else None
    if expected is not None and value == expected:
        return True
    return _stage_name(value) == stage_name


def _restart_frontier_stage_name(
    *,
    surface: "EnabledWorkflowSurface",
    state: Any,
) -> Optional[str]:
    raw_run_mode = getattr(surface, "run_mode", None)
    run_mode = str(getattr(raw_run_mode, "value", raw_run_mode))
    if run_mode != "restart":
        return None

    current_major_stage = getattr(state, "current_major_stage", None)
    if current_major_stage is None:
        return None

    if _state_stage_matches(state, current_major_stage, "supply_demand_loop"):
        current_sub_stage = getattr(state, "current_sub_stage", None)
        current_sub_stage_name = _stage_name(current_sub_stage)
        if current_sub_stage_name in {"activity_demand", "traffic_assignment"}:
            return current_sub_stage_name
        for stage_name in ("activity_demand", "traffic_assignment"):
            if any(
                spec.stage_name == stage_name
                and surface.step_enabled(spec.step_name, include_optional=False)
                for spec in schema_step_specs(include_optional=True)
            ):
                return stage_name
        return None

    return _stage_name(current_major_stage)


def _required_output_restart_frontier_order(
    *,
    surface: "EnabledWorkflowSurface",
    state: Any = None,
) -> Optional[int]:
    if state is None:
        return None

    frontier_stage_name = _restart_frontier_stage_name(surface=surface, state=state)
    if frontier_stage_name is None:
        return None

    orders = [
        spec.order
        for spec in schema_step_specs(include_optional=True)
        if spec.stage_name == frontier_stage_name
        and surface.step_enabled(spec.step_name, include_optional=False)
    ]
    if not orders:
        return None
    return min(orders)


def build_scenario_runtime_contract(
    *,
    settings: PilatesConfig,
    state: Any,
    workspace: Any,
    scenario_id: str,
    seed: Optional[int],
    cache_epoch: int,
    surface: "EnabledWorkflowSurface",
    build_scenario_consist_kwargs_fn: Callable[[Any], Dict[str, Any]],
    build_coupler_schema_fn: Callable[..., Dict[str, str]],
    validate_workflow_step_contracts_fn: Callable[..., None],
    build_schema_steps_fn: Callable[[], List[Callable[..., Any]]],
    filter_schema_steps_for_enabled_models_fn: Callable[..., List[Callable[..., Any]]],
    merge_epoch_facet_fn: Callable[..., Dict[str, Any]],
    scenario_name_template: str,
) -> Dict[str, Any]:
    scenario_kwargs = build_scenario_consist_kwargs_fn(settings)
    scenario_step_tags = [f"scenario_id:{scenario_id}"]
    if seed is not None:
        scenario_step_tags.append(f"seed:{seed}")
    scenario_step_facet: Dict[str, Any] = {"scenario_id": scenario_id}
    if seed is not None:
        scenario_step_facet["seed"] = seed
    scenario_kwargs["step_tags"] = merge_tag_list(
        scenario_kwargs.get("step_tags"),
        scenario_step_tags,
    )
    scenario_kwargs["step_facet"] = {
        **facet_to_mapping(scenario_kwargs.get("step_facet")),
        **scenario_step_facet,
    }
    scenario_kwargs["facet"] = merge_epoch_facet_fn(
        existing=scenario_kwargs.get("facet"),
        scenario_id=scenario_id,
        seed=seed,
        model="pilates_orchestrator",
        year=None,
        iteration=None,
    )
    scenario_kwargs.setdefault("name_template", scenario_name_template)
    scenario_kwargs.setdefault("cache_epoch", cache_epoch)

    schema_steps_all = build_schema_steps_fn()
    schema_steps_enabled = filter_schema_steps_for_enabled_models_fn(
        schema_steps_all,
        include_optional=True,
        surface=surface,
    )
    validate_workflow_step_contracts_fn(
        declared_steps=schema_steps_enabled,
        require_all_tracked_declared=False,
    )
    coupler_schema = build_coupler_schema_fn(schema_steps_enabled, settings=settings)
    required_output_keys = _required_output_keys_for_surface(
        surface=surface,
        state=state,
    )
    scenario_kwargs["require_outputs"] = required_output_keys
    return {
        "scenario_kwargs": scenario_kwargs,
        "schema_steps_all": schema_steps_all,
        "schema_steps_enabled": schema_steps_enabled,
        "coupler_schema": coupler_schema,
        "required_output_keys": required_output_keys,
    }
