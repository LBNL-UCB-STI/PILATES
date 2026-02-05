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

    def _workspace_path(ctx: StepContext) -> Optional[str]:
        ws = ctx.workspace
        if ws is not None:
            if isinstance(ws, Path):
                return str(ws)
            if isinstance(ws, str):
                return ws
            full_path = getattr(ws, "full_path", None)
            if isinstance(full_path, str):
                return full_path
        runtime_kwargs = ctx.runtime_kwargs or {}
        ws_runtime = runtime_kwargs.get("workspace")
        if ws_runtime is None:
            return None
        full_path = getattr(ws_runtime, "full_path", None)
        if isinstance(full_path, str):
            return full_path
        if isinstance(ws_runtime, (Path, str)):
            return str(ws_runtime)
        return None

    def _canonicalization_identity(ctx: StepContext) -> Dict[str, Any]:
        tracker = cr.current_tracker()
        settings = ctx.settings
        if tracker is None or settings is None:
            return {}

        runtime_kwargs = ctx.runtime_kwargs or {}
        workspace_obj = runtime_kwargs.get("workspace")
        current_run = cr.current_run()
        run_id = getattr(current_run, "id", None) if current_run else None

        try:
            from consist.core.config_canonicalization import ConfigAdapterOptions
        except Exception:
            return {}

        if model == "activitysim_preprocess":
            try:
                from consist.integrations.activitysim import ActivitySimConfigAdapter
            except Exception:
                return {}

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
                return {}

            options = ConfigAdapterOptions(
                strict=False,
                bundle=True,
                ingest=True,
                allow_heuristic_refs=True,
            )
            contribution = tracker.canonicalize_config(
                ActivitySimConfigAdapter(),
                [config_root],
                run_id=run_id,
                options=options,
            )
            return {
                "canonical_config_identity_hash": contribution.identity_hash,
                "canonical_config_adapter_version": contribution.adapter_version,
            }

        if model == "beam_preprocess":
            try:
                from consist.integrations.beam import BeamConfigAdapter
            except Exception:
                return {}

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
                return {}

            primary_config = config_root / settings.beam.config
            if not primary_config.exists():
                return {}

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
            options = ConfigAdapterOptions(
                strict=False,
                bundle=False,
                ingest=True,
                allow_heuristic_refs=True,
            )
            original_env: Dict[str, Optional[str]] = {}
            for key, value in env_overrides.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            try:
                contribution = tracker.canonicalize_config(
                    BeamConfigAdapter(
                        primary_config=primary_config,
                        env_overrides=env_overrides,
                    ),
                    [config_root],
                    run_id=run_id,
                    options=options,
                )
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            return {
                "canonical_config_identity_hash": contribution.identity_hash,
                "canonical_config_adapter_version": contribution.adapter_version,
            }

        return {}

    def _resolve(ctx: StepContext) -> Dict[str, Any]:
        cache_key = id(ctx)
        if cache_key in cache:
            return cache[cache_key]
        settings = ctx.settings
        if settings is None:
            cache[cache_key] = {}
            return {}
        workspace_path = _workspace_path(ctx)
        resolved = build_step_consist_kwargs(
            model=model,
            settings=settings,
            workspace_path=workspace_path,
        )
        extra_identity = _canonicalization_identity(ctx)
        if extra_identity:
            config = resolved.get("config")
            if isinstance(config, dict):
                config = dict(config)
                config.update(extra_identity)
            else:
                config = dict(extra_identity)
            resolved["config"] = config
        cache[cache_key] = resolved
        return resolved

    return {
        "config": lambda ctx: _resolve(ctx).get("config"),
        "facet": lambda ctx: _resolve(ctx).get("facet"),
        "facet_index": lambda ctx: _resolve(ctx).get("facet_index"),
        "facet_schema_version": lambda ctx: _resolve(ctx).get("facet_schema_version"),
        "hash_inputs": lambda ctx: _resolve(ctx).get("hash_inputs"),
    }
