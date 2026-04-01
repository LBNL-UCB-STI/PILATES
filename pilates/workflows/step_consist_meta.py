"""
pilates/workflows/step_consist_meta.py

Provides lazy metadata builders for Consist step decoration.

## Why lazy resolution?

Consist's ``define_step()`` decorator accepts callables for ``adapter``,
``config``, ``facet``, ``facet_index``, ``facet_schema_version``, and
``identity_inputs``.  These callables receive a ``StepContext`` at step
execution time, giving them access to the runtime settings, state, and
workspace that only exist after the simulation has started.

Resolving these values at import/decoration time (i.e. statically) would not
work: the workspace doesn't exist yet, and config file paths can't be
discovered until the workspace is created for each run.

## What each returned key means for Consist caching

- ``adapter``: A ``ConfigAdapter`` (ActivitySimConfigAdapter or
  BeamConfigAdapter) that fingerprints the model's config files.  When any
  config file changes, Consist detects a cache miss automatically — without
  PILATES having to track config hashes manually.

- ``config``: A dict of scalar key-value pairs that form part of the run's
  cache identity (e.g. region, year, scenario settings).  Changing any value
  causes a cache miss.

- ``facet``: Structured metadata stored with every run for later querying
  (e.g. year, iteration, scenario_id, region).  Does not affect cache
  identity by default, but can be used to filter runs via
  ``tracker.find_runs(facet=...)``.

- ``facet_index``: Whether to index the facet for fast lookup. ``True`` for
  models where facet-based filtering is expected (see B1 in consist changelog).

- ``facet_schema_version``: Version tag for the facet schema; bump when facet
  structure changes to avoid querying stale runs.

- ``identity_inputs``: Additional file paths whose content hashes contribute to
  the cache identity (beyond what the adapter already covers).

## How to add a new model adapter

1. Add a ``_<model>_adapter(ctx)`` function analogous to ``_activitysim_adapter``
   and ``_beam_adapter``.  It should return a Consist ``ConfigAdapter`` instance
   or ``None`` if config discovery fails gracefully.
2. Wire it into ``_adapter(ctx)`` with a ``model.startswith("<model>_")`` branch.
3. If the model has ATLAS-style runtime identity (e.g. a subyear), add it in
   ``_resolve()`` alongside the existing ``atlas_`` branch.
4. Register the new model in ``WORKFLOW_STEP_SPECS`` (catalog.py) and implement
   the step function in ``pilates/workflows/steps/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from consist.core.step_context import StepContext

from pilates.utils.consist_config import build_step_consist_kwargs


def consist_step_meta(model: str) -> Dict[str, Any]:
    """
    Build StepContext-callable metadata for Consist step defaults.

    The callables mirror kwargs typically passed to `scenario.run(...)` and let
    Consist resolve per-step config/facet/identity input metadata at execution
    time.
    """

    cache_attr = "_pilates_step_meta_cache"

    def _runtime_value(ctx: StepContext, name: str) -> Any:
        return ctx.get_runtime(name, default=None)

    def _settings(ctx: StepContext) -> Any:
        return _runtime_value(ctx, "settings")

    def _state(ctx: StepContext) -> Any:
        return _runtime_value(ctx, "state")

    def _workspace_path_from_value(value: Any) -> Optional[str]:
        if value is None:
            return None

        full_path = getattr(value, "full_path", None)
        if isinstance(full_path, (Path, str)):
            return str(full_path)

        if isinstance(value, (Path, str)):
            return str(value)

        return None

    def _workspace(ctx: StepContext) -> Any:
        return _runtime_value(ctx, "workspace")

    def _workspace_path(ctx: StepContext) -> Optional[str]:
        return _workspace_path_from_value(_workspace(ctx))

    def _activitysim_adapter(ctx: StepContext) -> Any:
        settings = _settings(ctx)
        if settings is None:
            return None
        activitysim_settings = getattr(settings, "activitysim", None)
        if activitysim_settings is None:
            return None

        try:
            from consist.integrations.activitysim import ActivitySimConfigAdapter
        except Exception:
            return None

        workspace_obj = _workspace(ctx)
        mutable_configs_root: Optional[Path] = None
        if workspace_obj is not None and hasattr(
            workspace_obj, "get_asim_mutable_configs_dir"
        ):
            mutable_configs_root = Path(workspace_obj.get_asim_mutable_configs_dir())
        else:
            ws_path = _workspace_path(ctx)
            if ws_path:
                mutable_configs_root = (
                    Path(ws_path) / activitysim_settings.local_mutable_configs_folder
                )
        if mutable_configs_root is None:
            return None

        main_configs_dir = getattr(activitysim_settings, "main_configs_dir", "configs")
        candidates = [
            main_configs_dir,
            "configs",
            "configs_extended",
            "configs_mp",
            "configs_sh_compile",
        ]
        seen = set()
        ordered_unique_candidates = []
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            ordered_unique_candidates.append(name)

        config_roots = []
        for dirname in ordered_unique_candidates:
            candidate = mutable_configs_root / dirname
            if candidate.exists():
                config_roots.append(candidate)

        if not config_roots:
            return None

        return ActivitySimConfigAdapter(root_dirs=config_roots)

    def _beam_adapter(ctx: StepContext) -> Any:
        settings = _settings(ctx)
        if settings is None:
            return None
        run_settings = getattr(settings, "run", None)
        beam_settings = getattr(settings, "beam", None)
        if run_settings is None or beam_settings is None:
            return None

        try:
            from consist.integrations.beam import BeamConfigAdapter
        except Exception:
            return None

        workspace_obj = _workspace(ctx)
        config_root: Optional[Path] = None
        if workspace_obj is not None and hasattr(
            workspace_obj, "get_beam_mutable_data_dir"
        ):
            config_root = (
                Path(workspace_obj.get_beam_mutable_data_dir()) / run_settings.region
            )
        else:
            ws_path = _workspace_path(ctx)
            if ws_path:
                config_root = (
                    Path(ws_path)
                    / beam_settings.local_mutable_data_folder
                    / run_settings.region
                )
        if config_root is None:
            return None

        primary_config = config_root / beam_settings.config
        if not primary_config.exists():
            return None

        beam_input_root = config_root.resolve()
        pwd_candidates = [
            beam_input_root.parent,
            beam_input_root,
            beam_input_root.parent.parent,
        ]
        expected_suffix = Path("input") / run_settings.region
        pwd_root = next(
            (root for root in pwd_candidates if (root / expected_suffix).exists()),
            beam_input_root.parent,
        )
        env_overrides = {"PWD": str(pwd_root)}
        return BeamConfigAdapter(
            root_dirs=[config_root],
            primary_config=primary_config,
            env_overrides=env_overrides,
        )

    def _adapter(ctx: StepContext) -> Any:
        if model.startswith("activitysim_"):
            return _activitysim_adapter(ctx)
        if model.startswith("beam_"):
            return _beam_adapter(ctx)
        return None

    def _resolve(ctx: StepContext) -> Dict[str, Any]:
        cache: Optional[Dict[str, Dict[str, Any]]] = getattr(ctx, cache_attr, None)
        if not isinstance(cache, dict):
            cache = {}
            try:
                setattr(ctx, cache_attr, cache)
            except Exception:
                cache = None

        if cache is not None and model in cache:
            return cache[model]
        settings = _settings(ctx)
        if settings is None:
            if cache is not None:
                cache[model] = {}
            return {}
        workspace_path = _workspace_path(ctx)
        resolved = build_step_consist_kwargs(
            model=model,
            settings=settings,
            workspace_path=workspace_path,
        )
        if model.startswith("atlas_"):
            state = _state(ctx)
            atlas_runtime_identity: Dict[str, Any] = {}
            if state is not None:
                atlas_subyear = getattr(state, "year", None)
                if atlas_subyear is not None:
                    atlas_runtime_identity["atlas_subyear"] = atlas_subyear
                main_forecast_year = getattr(state, "main_forecast_year", None)
                if main_forecast_year is not None:
                    atlas_runtime_identity["main_forecast_year"] = main_forecast_year
            if atlas_runtime_identity:
                resolved["config"] = {
                    **dict(resolved.get("config") or {}),
                    **atlas_runtime_identity,
                }
                resolved["facet"] = {
                    **dict(resolved.get("facet") or {}),
                    **atlas_runtime_identity,
                }
        adapter = _adapter(ctx)
        if adapter is not None:
            resolved["adapter"] = adapter
        if cache is not None:
            cache[model] = resolved
        return resolved

    return {
        "adapter": lambda ctx: _resolve(ctx).get("adapter"),
        "config": lambda ctx: _resolve(ctx).get("config"),
        "facet": lambda ctx: _resolve(ctx).get("facet"),
        "facet_index": lambda ctx: _resolve(ctx).get("facet_index"),
        "facet_schema_version": lambda ctx: _resolve(ctx).get("facet_schema_version"),
        "identity_inputs": lambda ctx: _resolve(ctx).get("identity_inputs"),
    }
