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

if TYPE_CHECKING:
    from pilates.workflows.state import WorkflowState

from pilates.config import PilatesConfig, SettingsLike
from pilates.utils.io import (
    compute_model_enabled_flags,
    get_activity_demand_model,
    get_traffic_assignment_model,
)
from pilates.utils.settings_helper import get as get_setting
from pilates.workflows.steps import STEP_DEFINITIONS

logger = logging.getLogger(__name__)


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
    """Return committed native Consist callables in canonical step order."""
    return [definition.function for definition in STEP_DEFINITIONS.values()]


def _enabled_schema_step_names(
    settings: SettingsLike,
    *,
    include_optional: bool = True,
) -> set[str]:
    flags = compute_model_enabled_flags(settings)
    enabled: set[str] = set()

    if flags.get("land_use_enabled"):
        enabled.update({"urbansim_run", "urbansim_postprocess"})
    if flags.get("vehicle_ownership_model_enabled"):
        enabled.update({"atlas_preprocess", "atlas_run", "atlas_postprocess"})

    activity_model = get_activity_demand_model(settings)
    if flags.get("activity_demand_enabled") and (
        activity_model is None or activity_model == "activitysim"
    ):
        enabled.update(
            {
                "activitysim_preprocess",
                "activitysim_run",
                "activitysim_postprocess",
            }
        )

    traffic_model = get_traffic_assignment_model(settings)
    if flags.get("traffic_assignment_enabled") and (
        traffic_model is None or traffic_model == "beam"
    ):
        schedule = str(
            get_setting(settings, "beam.full_skim.run_schedule", default="disabled")
            or "disabled"
        )
        if schedule == "standalone":
            if include_optional:
                enabled.add("beam_preprocess")
                enabled.add("beam_full_skim")
        else:
            enabled.update({"beam_preprocess", "beam_run", "beam_postprocess"})
            if include_optional and schedule in {
                "after_each_iteration",
                "after_final_iteration",
            }:
                enabled.add("beam_full_skim")

    if "beam_run" in enabled and get_setting(settings, "run.models.impacts"):
        enabled.update(("impacts_presim", "impacts_postsim"))

    if include_optional and bool(
        get_setting(settings, "postprocessing.enabled", default=False)
    ):
        enabled.add("postprocessing")

    return enabled


def filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    *,
    settings: SettingsLike,
    include_optional: bool = True,
) -> List[Callable[..., Any]]:
    enabled_step_names = _enabled_schema_step_names(
        settings,
        include_optional=include_optional,
    )
    enabled_step_functions = {
        definition.function
        for step_name, definition in STEP_DEFINITIONS.items()
        if step_name in enabled_step_names
    }
    return [step_func for step_func in steps if step_func in enabled_step_functions]


def build_scenario_runtime_contract(
    *,
    settings: PilatesConfig,
    state: "WorkflowState",
    workspace: Any,
    scenario_id: str,
    seed: Optional[int],
    cache_epoch: int,
    build_scenario_consist_kwargs_fn: Callable[[Any], Dict[str, Any]],
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
        settings=settings,
        include_optional=True,
    )
    required_output_keys: List[str] = []
    scenario_kwargs["require_outputs"] = required_output_keys
    return {
        "scenario_kwargs": scenario_kwargs,
        "schema_steps_all": schema_steps_all,
        "schema_steps_enabled": schema_steps_enabled,
        "required_output_keys": required_output_keys,
    }
