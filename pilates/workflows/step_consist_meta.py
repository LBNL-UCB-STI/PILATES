from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from consist.core.step_context import StepContext

from pilates.utils import consist_runtime as cr
from pilates.utils.consist_config import build_step_consist_kwargs


def consist_step_meta(model: str) -> Dict[str, Any]:
    """
    Build StepContext-callable metadata for Consist step defaults.

    The callables mirror kwargs typically passed to `scenario.run(...)` and let
    Consist resolve per-step config/facet/hash input metadata at execution time.
    """

    cache: Dict[int, Dict[str, Any]] = {}

    def _settings(ctx: StepContext) -> Any:
        runtime_settings = getattr(ctx, "runtime_settings", None)
        if runtime_settings is not None:
            return runtime_settings

        runtime_kwargs = ctx.runtime_kwargs or {}
        runtime_settings_kwarg = runtime_kwargs.get("settings")
        if runtime_settings_kwarg is not None:
            return runtime_settings_kwarg

        legacy_settings = getattr(ctx, "settings", None)
        if legacy_settings is not None:
            return legacy_settings
        return getattr(ctx, "consist_settings", None)

    def _workspace_path(ctx: StepContext) -> Optional[str]:
        runtime_workspace = getattr(ctx, "runtime_workspace", None)
        if runtime_workspace is not None:
            full_path = getattr(runtime_workspace, "full_path", None)
            if isinstance(full_path, str):
                return full_path
            if isinstance(runtime_workspace, (Path, str)):
                return str(runtime_workspace)

        runtime_kwargs = ctx.runtime_kwargs or {}
        ws_runtime = runtime_kwargs.get("workspace")
        if ws_runtime is not None:
            full_path = getattr(ws_runtime, "full_path", None)
            if isinstance(full_path, str):
                return full_path
            if isinstance(ws_runtime, (Path, str)):
                return str(ws_runtime)

        ws = getattr(ctx, "workspace", None)
        if ws is None:
            ws = getattr(ctx, "consist_workspace", None)
        if ws is not None:
            if isinstance(ws, Path):
                return str(ws)
            if isinstance(ws, str):
                return ws
            full_path = getattr(ws, "full_path", None)
            if isinstance(full_path, str):
                return full_path
        return None

    def _config_plan(ctx: StepContext) -> Any:
        tracker = cr.current_tracker()
        settings = _settings(ctx)
        if tracker is None or settings is None:
            return None

        runtime_kwargs = ctx.runtime_kwargs or {}
        workspace_obj = runtime_kwargs.get("workspace")
        prepare_config_resolver = getattr(tracker, "prepare_config_resolver", None)
        if not callable(prepare_config_resolver):
            return None

        def _resolve_plan_from_adapter(
            adapter: Any,
            config_root: Path,
            env_overrides: Optional[Dict[str, str]] = None,
        ) -> Any:
            try:
                resolver = prepare_config_resolver(
                    adapter=adapter,
                    config_dirs=[config_root],
                )
            except TypeError:
                try:
                    resolver = prepare_config_resolver(adapter, [config_root])
                except TypeError:
                    return None
            except Exception:
                return None

            if not callable(resolver):
                return None

            if not env_overrides:
                try:
                    return resolver(ctx)
                except Exception:
                    return None

            original_env: Dict[str, Optional[str]] = {}
            for key, value in env_overrides.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            try:
                return resolver(ctx)
            except Exception:
                return None
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        if model in {"activitysim_compile", "activitysim_run"}:
            try:
                from consist.integrations.activitysim import ActivitySimConfigAdapter
            except Exception:
                return None

            config_root: Optional[Path] = None
            if workspace_obj is not None and hasattr(workspace_obj, "get_asim_mutable_configs_dir"):
                config_root = (
                    Path(workspace_obj.get_asim_mutable_configs_dir())
                    / settings.activitysim.main_configs_dir
                )
            else:
                ws_path = _workspace_path(ctx)
                if ws_path:
                    config_root = (
                        Path(ws_path)
                        / settings.activitysim.local_mutable_configs_folder
                        / settings.activitysim.main_configs_dir
                    )
            if config_root is None or not config_root.exists():
                return None

            return _resolve_plan_from_adapter(
                ActivitySimConfigAdapter(),
                config_root,
            )

        if model in {"beam_run", "beam_full_skim"}:
            try:
                from consist.integrations.beam import BeamConfigAdapter
            except Exception:
                return None

            config_root: Optional[Path] = None
            if workspace_obj is not None and hasattr(workspace_obj, "get_beam_mutable_data_dir"):
                config_root = (
                    Path(workspace_obj.get_beam_mutable_data_dir()) / settings.run.region
                )
            else:
                ws_path = _workspace_path(ctx)
                if ws_path:
                    config_root = (
                        Path(ws_path)
                        / settings.beam.local_mutable_data_folder
                        / settings.run.region
                    )
            if config_root is None:
                return None

            primary_config = config_root / settings.beam.config
            if not primary_config.exists():
                return None

            beam_input_root = config_root.resolve()
            pwd_candidates = [
                beam_input_root.parent,
                beam_input_root,
                beam_input_root.parent.parent,
            ]
            expected_suffix = Path("input") / settings.run.region
            pwd_root = next(
                (root for root in pwd_candidates if (root / expected_suffix).exists()),
                beam_input_root.parent,
            )
            env_overrides = {"PWD": str(pwd_root)}
            return _resolve_plan_from_adapter(
                BeamConfigAdapter(
                    primary_config=primary_config,
                    env_overrides=env_overrides,
                ),
                config_root,
                env_overrides=env_overrides,
            )

        return None

    def _resolve(ctx: StepContext) -> Dict[str, Any]:
        cache_key = id(ctx)
        if cache_key in cache:
            return cache[cache_key]
        settings = _settings(ctx)
        if settings is None:
            cache[cache_key] = {}
            return {}
        workspace_path = _workspace_path(ctx)
        resolved = build_step_consist_kwargs(
            model=model,
            settings=settings,
            workspace_path=workspace_path,
        )
        plan = _config_plan(ctx)
        if plan is not None:
            resolved["config_plan"] = plan
            # Avoid duplicate/overlapping invalidation when Consist canonical
            # config plans are available. The plan's identity hash is now the
            # authoritative file-based config signature for these steps.
            resolved.pop("hash_inputs", None)
        cache[cache_key] = resolved
        return resolved

    return {
        "config": lambda ctx: _resolve(ctx).get("config"),
        "config_plan": lambda ctx: _resolve(ctx).get("config_plan"),
        "facet": lambda ctx: _resolve(ctx).get("facet"),
        "facet_index": lambda ctx: _resolve(ctx).get("facet_index"),
        "facet_schema_version": lambda ctx: _resolve(ctx).get("facet_schema_version"),
        "hash_inputs": lambda ctx: _resolve(ctx).get("hash_inputs"),
    }
