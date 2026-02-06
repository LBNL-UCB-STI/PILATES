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

    def _canonicalization_identity(ctx: StepContext) -> Dict[str, Any]:
        tracker = cr.current_tracker()
        settings = _settings(ctx)
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

        def _identity_payload(identity_hash: Any, adapter_version: Any) -> Dict[str, Any]:
            if not identity_hash:
                return {}
            payload: Dict[str, Any] = {
                "canonical_config_identity_hash": identity_hash,
                "canonical_config_adapter_version": adapter_version,
            }
            return payload

        def _resolve_identity_from_adapter(
            adapter: Any,
            config_root: Path,
            *,
            options: Any,
        ) -> Dict[str, Any]:
            """
            Resolve canonical config identity at metadata-resolution time.

            Prefer `prepare_config` (no active-run requirement) so this can run
            before ScenarioContext starts the step run. Fall back to legacy
            `canonicalize_config` only when available and safe.
            """
            prepare_config = getattr(tracker, "prepare_config", None)
            if callable(prepare_config):
                try:
                    plan = prepare_config(adapter, [config_root], options=options)
                except TypeError:
                    plan = prepare_config(adapter, [config_root], strict=False)
                except Exception:
                    plan = None
                if plan is not None:
                    return _identity_payload(
                        getattr(plan, "identity_hash", None),
                        getattr(plan, "adapter_version", None),
                    )

            canonicalize_config = getattr(tracker, "canonicalize_config", None)
            if not callable(canonicalize_config):
                return {}
            try:
                kwargs: Dict[str, Any] = {"options": options}
                if run_id is not None:
                    kwargs["run_id"] = run_id
                contribution = canonicalize_config(adapter, [config_root], **kwargs)
            except Exception:
                return {}
            return _identity_payload(
                getattr(contribution, "identity_hash", None),
                getattr(contribution, "adapter_version", None),
            )

        if model in {"activitysim_compile", "activitysim_run"}:
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
            return _resolve_identity_from_adapter(
                ActivitySimConfigAdapter(),
                config_root,
                options=options,
            )

        if model == "beam_run":
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
                identity = _resolve_identity_from_adapter(
                    BeamConfigAdapter(
                        primary_config=primary_config,
                        env_overrides=env_overrides,
                    ),
                    config_root,
                    options=options,
                )
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            return identity

        return {}

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
