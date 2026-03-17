from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pilates.workflows.catalog import enabled_schema_step_models, schema_step_names
from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_full_skim_step,
    make_beam_postprocess_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_impacts_postprocess_step,
    make_impacts_preprocess_step,
    make_impacts_run_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
)

logger = logging.getLogger(__name__)


def coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_scenario_id(settings: Any) -> str:
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


def resolve_seed(settings: Any) -> Optional[int]:
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


class EpochTaggingScenarioProxy:
    """
    Wrapper around a Consist scenario that injects epoch metadata and parent linkage.
    """

    def __init__(self, scenario: Any, *, scenario_id: str, seed: Optional[int]) -> None:
        self._scenario = scenario
        self._scenario_id = scenario_id
        self._seed = seed
        self._activitysim_run_ids: Dict[Tuple[int, int], str] = {}
        self._activitysim_step_ids: Dict[Tuple[int, int], str] = {}
        self._beam_run_ids: Dict[Tuple[int, int], str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scenario, name)

    def _infer_model_name(self, run_kwargs: Mapping[str, Any]) -> Optional[str]:
        explicit = run_kwargs.get("model")
        if explicit:
            return str(explicit)
        fn = run_kwargs.get("fn")
        meta = getattr(fn, "__consist_step__", None)
        model_name = getattr(meta, "model", None)
        if model_name:
            return str(model_name)
        return None

    def _resolve_parent_run_id(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
    ) -> Optional[str]:
        if model_name is None or year is None or iteration is None:
            return None
        model_norm = model_name.lower()
        if model_norm == "beam" or model_norm.startswith("beam_"):
            key = (year, iteration)
            return self._activitysim_run_ids.get(key) or self._activitysim_step_ids.get(
                key
            )
        if (model_norm == "activitysim" or model_norm.startswith("activitysim_")) and iteration > 0:
            return self._beam_run_ids.get((year, iteration - 1))
        return None

    def _should_expect_parent(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
    ) -> bool:
        if model_name is None or year is None or iteration is None:
            return False
        model_norm = model_name.lower()
        if model_norm == "beam" or model_norm.startswith("beam_"):
            return True
        if (model_norm == "activitysim" or model_norm.startswith("activitysim_")) and iteration > 0:
            return True
        return False

    def _remember_run_id(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
        run_id: Optional[str],
    ) -> None:
        if model_name is None or year is None or iteration is None or not run_id:
            return
        key = (year, iteration)
        model_norm = model_name.lower()
        if model_norm == "activitysim" or model_norm.startswith("activitysim_"):
            self._activitysim_step_ids[key] = run_id
            if model_norm in {"activitysim", "activitysim_run"}:
                self._activitysim_run_ids[key] = run_id
        elif model_norm in {"beam", "beam_run"}:
            self._beam_run_ids[key] = run_id

    def remember_restored_run_id(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
        run_id: Optional[str],
    ) -> None:
        self._remember_run_id(
            model_name=model_name,
            year=year,
            iteration=iteration,
            run_id=run_id,
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) > 2:
            return self._scenario.run(*args, **kwargs)

        run_kwargs: Dict[str, Any] = dict(kwargs)
        if len(args) >= 1 and "fn" not in run_kwargs:
            run_kwargs["fn"] = args[0]
        if len(args) == 2 and "name" not in run_kwargs:
            run_kwargs["name"] = args[1]

        model_name = self._infer_model_name(run_kwargs)
        year = coerce_int(run_kwargs.get("year"))
        iteration = coerce_int(run_kwargs.get("iteration"))
        if model_name and "model" not in run_kwargs:
            run_kwargs["model"] = model_name

        if not run_kwargs.get("parent_run_id"):
            resolved_parent = self._resolve_parent_run_id(
                model_name=model_name,
                year=year,
                iteration=iteration,
            )
            if resolved_parent:
                run_kwargs["parent_run_id"] = resolved_parent
            elif self._should_expect_parent(
                model_name=model_name,
                year=year,
                iteration=iteration,
            ):
                logger.debug(
                    "[RunTagging] parent_run_id unavailable for model=%s year=%s iteration=%s; leaving unset.",
                    model_name,
                    year,
                    iteration,
                )

        tag_additions = [f"scenario_id:{self._scenario_id}"]
        if self._seed is not None:
            tag_additions.append(f"seed:{self._seed}")
        if model_name:
            tag_additions.append(f"model:{model_name}")
        if year is not None:
            tag_additions.append(f"year:{year}")
        if iteration is not None:
            tag_additions.append(f"iteration:{iteration}")
        run_kwargs["tags"] = merge_tag_list(run_kwargs.get("tags"), tag_additions)
        run_kwargs["facet"] = merge_epoch_facet(
            existing=run_kwargs.get("facet"),
            scenario_id=self._scenario_id,
            seed=self._seed,
            model=model_name,
            year=year,
            iteration=iteration,
        )

        result = self._scenario.run(**run_kwargs)
        run_id = str(getattr(getattr(result, "run", None), "id", "")).strip() or None
        self._remember_run_id(
            model_name=model_name,
            year=year,
            iteration=iteration,
            run_id=run_id,
        )
        return result


class SchemaCoupler:
    """No-op coupler used to construct decorated step callables for schema introspection."""

    def get(self, _key: str, default: Optional[Any] = None) -> Any:
        return default

    def set(self, _key: str, _value: Any) -> None:
        return None

    def update(self, _mapping: Dict[str, Any]) -> None:
        return None

    def view(self, _namespace: str) -> "SchemaCoupler":
        return self

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None


def resolve_cache_epoch(settings: Any) -> int:
    value = getattr(getattr(settings, "run", None), "cache_epoch", 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def build_schema_steps() -> List[Callable[..., Any]]:
    coupler = SchemaCoupler()
    outputs_holder = StepOutputsHolder()
    step_factories: Dict[str, Callable[..., Any]] = {
        "urbansim_preprocess": make_urbansim_preprocess_step,
        "urbansim_run": make_urbansim_run_step,
        "urbansim_postprocess": make_urbansim_postprocess_step,
        "atlas_preprocess": make_atlas_preprocess_step,
        "atlas_run": make_atlas_run_step,
        "atlas_postprocess": make_atlas_postprocess_step,
        "activitysim_preprocess": make_activitysim_preprocess_step,
        "activitysim_compile": make_activitysim_compile_step,
        "activitysim_run": make_activitysim_run_step,
        "activitysim_postprocess": make_activitysim_postprocess_step,
        "beam_preprocess": make_beam_preprocess_step,
        "beam_run": make_beam_run_step,
        "beam_postprocess": make_beam_postprocess_step,
        "beam_full_skim": make_beam_full_skim_step,
        "impacts_preprocess": make_impacts_preprocess_step,
        "impacts_run": make_impacts_run_step,
        "impacts_postprocess": make_impacts_postprocess_step,
    }
    ordered_steps = schema_step_names()
    missing_factories = [name for name in ordered_steps if name not in step_factories]
    if missing_factories:
        raise RuntimeError(
            "Missing schema step factories for: " + ", ".join(missing_factories)
        )
    return [
        step_factories[step_name](coupler=coupler, outputs_holder=outputs_holder)
        for step_name in ordered_steps
    ]


def is_model_enabled(settings: Any, *, flag_attr: str, model_attr: str) -> bool:
    explicit_flag = getattr(settings, flag_attr, None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None) if run_cfg is not None else None
    return bool(getattr(model_cfg, model_attr, None))


def filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    settings: Any,
    *,
    include_optional: bool = True,
) -> List[Callable[..., Any]]:
    enabled_models = enabled_schema_step_models(
        settings,
        is_model_enabled=is_model_enabled,
        include_optional=include_optional,
    )

    filtered: List[Callable[..., Any]] = []
    for step_func in steps:
        meta = getattr(step_func, "__consist_step__", None)
        model_name = getattr(meta, "model", "") if meta is not None else ""
        if model_name not in enabled_models:
            continue
        filtered.append(step_func)
    return filtered


def build_scenario_runtime_contract(
    *,
    settings: Any,
    scenario_id: str,
    seed: Optional[int],
    cache_epoch: int,
    build_scenario_consist_kwargs_fn: Callable[[Any], Dict[str, Any]],
    build_coupler_schema_fn: Callable[..., Dict[str, str]],
    validate_workflow_step_contracts_fn: Callable[..., None],
    build_schema_steps_fn: Callable[[], List[Callable[..., Any]]],
    filter_schema_steps_for_enabled_models_fn: Callable[..., List[Callable[..., Any]]],
    merge_epoch_facet_fn: Callable[..., Dict[str, Any]],
    scenario_name_template: str,
) -> Dict[str, Any]:
    scenario_kwargs = build_scenario_consist_kwargs_fn(settings)
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
    validate_workflow_step_contracts_fn(declared_steps=schema_steps_all)
    schema_steps_enabled = filter_schema_steps_for_enabled_models_fn(
        schema_steps_all,
        settings,
        include_optional=True,
    )
    coupler_schema = build_coupler_schema_fn(schema_steps_enabled, settings=settings)
    required_schema = build_coupler_schema_fn(
        filter_schema_steps_for_enabled_models_fn(
            schema_steps_all,
            settings,
            include_optional=False,
        ),
        settings=settings,
        include_extras=False,
    )
    required_output_keys = list(required_schema.keys())
    scenario_kwargs["require_outputs"] = required_output_keys
    return {
        "scenario_kwargs": scenario_kwargs,
        "schema_steps_all": schema_steps_all,
        "schema_steps_enabled": schema_steps_enabled,
        "coupler_schema": coupler_schema,
        "required_output_keys": required_output_keys,
    }
