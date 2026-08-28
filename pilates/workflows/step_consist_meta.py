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

import os
from pathlib import Path
from typing import Any, Dict, Optional

from consist.core.step_context import StepContext

from pilates.beam.config_hocon import beam_config_env_overrides, beam_config_root
from pilates.atlas.preprocessor import selected_atlas_static_input_sources
from pilates.urbansim.preprocessor import selected_urbansim_static_input_sources
from pilates.utils.consist_config import build_step_consist_kwargs
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_VEHICLES_IN,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.step_definition import InputContract

_DISABLE_BEAM_CONFIG_ADAPTER_ENV = "PILATES_DISABLE_BEAM_CONFIG_ADAPTER"


def activitysim_config_root_dirs(
    settings: Any,
    mutable_configs_root: Path,
) -> tuple[Path, ...]:
    """Return ActivitySim config roots in the same order used for identity."""
    activitysim_settings = getattr(settings, "activitysim", None)
    main_configs_dir = (
        getattr(activitysim_settings, "main_configs_dir", None) or "configs"
    )
    candidates = (
        main_configs_dir,
        "configs",
        "configs_extended",
        "configs_mp",
        "configs_sh_compile",
    )
    return tuple(
        mutable_configs_root / dirname for dirname in dict.fromkeys(candidates)
    )


def _adapter_covered_identity_input(item: Any, *, model: str) -> bool:
    if not isinstance(item, tuple) or not item:
        return False
    key = str(item[0])
    if model.startswith("beam_"):
        return key.startswith("beam_conf")
    if model.startswith("activitysim_"):
        return key == "asim_mutable_configs" or key.startswith("asim_mutable_configs/")
    return False


def _filter_adapter_covered_identity_inputs(
    identity_inputs: Any,
    *,
    model: str,
) -> Optional[list[Any]]:
    if not identity_inputs:
        return None
    filtered = [
        item
        for item in identity_inputs
        if not _adapter_covered_identity_input(item, model=model)
    ]
    return filtered or None


def consist_step_meta(
    model: str,
    *,
    input_contract: InputContract | None = None,
) -> Dict[str, Any]:
    """
    Build StepContext-callable metadata for Consist step defaults.

    The callables mirror kwargs typically passed to `scenario.run(...)` and let
    Consist resolve per-step config/facet/identity input metadata at execution
    time.
    """

    cache_attr = "_pilates_step_meta_cache"

    def _contract_report(*, adapter: Any, payload: Any) -> Dict[str, Any] | None:
        if input_contract is None:
            return None
        config_contract = input_contract.config_contract
        if config_contract is None:
            configuration = {"kind": None, "available": False}
        elif config_contract.kind == "adapter":
            configuration = {"kind": "adapter", "available": adapter is not None}
        else:
            configuration = {
                "kind": "payload",
                "available": isinstance(payload, dict),
            }
        return {
            "status": input_contract.status,
            "reason": input_contract.reason,
            "configuration": configuration,
        }

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

    def _state_is_start_year(state: Any) -> bool:
        is_start_year = getattr(state, "is_start_year", None)
        if callable(is_start_year):
            try:
                return bool(is_start_year())
            except TypeError:
                pass
        year = getattr(state, "year", None)
        start_year = getattr(state, "start_year", None)
        return (
            year is not None and start_year is not None and int(year) == int(start_year)
        )

    def _atlas_selected_usim_h5(state: Any) -> Any:
        current_h5 = getattr(state, "atlas_usim_datastore_h5", None)
        base_h5 = getattr(state, "atlas_usim_datastore_base_h5", None)
        if _state_is_start_year(state) and base_h5 is not None:
            return base_h5
        return current_h5 if current_h5 is not None else base_h5

    def _atlas_usim_households_identity(ctx: StepContext) -> Dict[str, Any]:
        state = _state(ctx)
        if state is None:
            return {}
        selected_h5 = _atlas_selected_usim_h5(state)
        if selected_h5 is None:
            return {}

        from pilates.utils.coupler_helpers import artifact_to_existing_path
        from pilates.utils.usim_h5 import fingerprint_usim_h5_table

        workspace_obj = _workspace(ctx)
        h5_path = artifact_to_existing_path(selected_h5, workspace=workspace_obj)
        if h5_path is None:
            return {
                "atlas_usim_households_identity_status": "unresolved",
                "atlas_usim_households_source": str(selected_h5),
            }

        year = getattr(state, "year", getattr(state, "current_year", None))
        fingerprint = fingerprint_usim_h5_table(
            h5_path=h5_path,
            year=int(year) if year is not None else None,
            table="households",
        )
        return {
            "atlas_usim_households_identity_status": "resolved",
            "atlas_usim_households_fingerprint_version": fingerprint[
                "fingerprint_version"
            ],
            "atlas_usim_households_sha256": fingerprint["sha256"],
            "atlas_usim_households_requested_year": fingerprint["requested_year"],
            "atlas_usim_households_table_path": fingerprint["resolved_table_path"],
            "atlas_usim_households_row_count": fingerprint["row_count"],
            "atlas_usim_households_column_count": fingerprint["column_count"],
        }

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

        config_roots = activitysim_config_root_dirs(settings, mutable_configs_root)
        missing_roots = [candidate for candidate in config_roots if not candidate.is_dir()]
        if missing_roots:
            missing = ", ".join(str(candidate) for candidate in missing_roots)
            raise RuntimeError(
                "ActivitySim configuration identity is incomplete; missing roots: "
                f"{missing}"
            )

        return ActivitySimConfigAdapter(root_dirs=list(config_roots))

    def _beam_adapter(ctx: StepContext) -> Any:
        if os.environ.get(_DISABLE_BEAM_CONFIG_ADAPTER_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None

        settings = _settings(ctx)
        if settings is None:
            return None
        run_settings = getattr(settings, "run", None)
        beam_settings = getattr(settings, "beam", None)
        if run_settings is None or beam_settings is None:
            return None

        try:
            from consist.integrations.beam import BeamConfigAdapter, BeamReferencePolicy
        except Exception:
            return None

        from pilates.beam.launch_config import BeamLaunchConfig

        workspace_obj = _workspace(ctx)
        launch_config = _runtime_value(ctx, "beam_launch_config")
        config_root: Optional[Path] = None
        workspace_root: Optional[Path] = None
        primary_config: Optional[Path] = None
        if isinstance(launch_config, BeamLaunchConfig):
            config_root = launch_config.root
            primary_config = launch_config.primary_config
        elif workspace_obj is not None and hasattr(
            workspace_obj, "get_beam_mutable_data_dir"
        ):
            workspace_root_path = _workspace_path_from_value(workspace_obj)
            if workspace_root_path:
                workspace_root = Path(workspace_root_path)
            config_root = beam_config_root(
                settings,
                workspace=workspace_obj,
            )
        else:
            ws_path = _workspace_path(ctx)
            if ws_path:
                workspace_root = Path(ws_path)
                config_root = beam_config_root(
                    settings,
                    workspace_path=ws_path,
                )
        if config_root is None:
            return None

        if primary_config is None:
            primary_config = config_root / beam_settings.config
        if not primary_config.exists():
            return None

        path_aliases: dict[str, Path] = {}
        if workspace_root is not None:
            path_aliases["workspace"] = workspace_root
        if workspace_obj is not None:
            if hasattr(workspace_obj, "get_beam_mutable_data_dir"):
                path_aliases["beam_input"] = Path(
                    workspace_obj.get_beam_mutable_data_dir()
                )
            if hasattr(workspace_obj, "get_beam_output_dir"):
                path_aliases["beam_output"] = Path(workspace_obj.get_beam_output_dir())
            if settings.activitysim is not None and hasattr(
                workspace_obj, "get_asim_output_dir"
            ):
                path_aliases["activitysim_output"] = Path(
                    workspace_obj.get_asim_output_dir()
                )
        path_aliases["beam_region_input"] = config_root

        vehicle_file_policy = BeamReferencePolicy(
            identity_policy=(
                "output_or_runtime_ignored"
                if model == "beam_preprocess"
                else "delegated_to_artifacts"
            ),
            role="beam_vehicle_input",
            required=model != "beam_preprocess",
            reason=(
                "generated_by_beam_preprocess_from_declared_atlas_vehicle_input"
                if model == "beam_preprocess"
                else "vehicles_declared_as_step_artifact"
            ),
            delegated_artifact_keys=(BEAM_VEHICLES_IN,),
        )

        reference_policies = {
            "beam.routing.r5.directory": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_r5_raw_network_directory",
                required=True,
                reason="r5_raw_osm_member_selected_and_recorded_by_beam_preprocess",
            ),
            "beam.routing.r5.osmFile": BeamReferencePolicy(
                identity_policy="ignored",
                required=False,
                reason="legacy_r5_osm_file_key_not_consulted_by_beam",
            ),
            "beam.routing.r5.osmMapdbFile": BeamReferencePolicy(
                identity_policy="output_or_runtime_ignored",
                required=False,
                reason="generated_r5_mapdb_cache_destination",
            ),
            "beam.exchange.scenario.folder": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_population_input_root",
                required=True,
                reason="population_inputs_declared_as_step_artifacts",
                delegated_artifact_keys=(
                    BEAM_PLANS_IN,
                    BEAM_HOUSEHOLDS_IN,
                    BEAM_PERSONS_IN,
                    BEAM_VEHICLES_IN,
                ),
            ),
            "beam.agentsim.agents.vehicles.vehiclesFilePath": vehicle_file_policy,
            "beam.warmStart.initialLinkstatsFilePath": BeamReferencePolicy(
                identity_policy="delegated_to_artifacts",
                role="beam_linkstats_warmstart",
                required=False,
                reason="warmstart_declared_as_optional_step_artifact",
                delegated_artifact_keys=(LINKSTATS_WARMSTART,),
            ),
        }
        dormant_matsim_keys = (
            "matsim.conversion.populationFile",
            "matsim.conversion.matsimNetworkFile",
            "matsim.conversion.scenarioDirectory",
            "matsim.conversion.shapeConfig.shapeFile",
            "matsim.conversion.vehiclesFile",
            "matsim.conversion.osmFile",
        )
        for key in dormant_matsim_keys:
            reference_policies[key] = BeamReferencePolicy(
                identity_policy="ignored",
                required=False,
                reason="dormant_matsim_example_config",
            )
        for key in (
            "beam.agentsim.agents.rideHail.managers[0].initialization.filePath",
            "beam.agentsim.agents.rideHail.managers[1].initialization.filePath",
        ):
            reference_policies[key] = BeamReferencePolicy(
                identity_policy="ignored",
                required=False,
                reason="procedural_ridehail_fleet_path_not_read_by_pilates",
            )
        for key in (
            "beam.physsim.inputNetworkFilePath",
            "matsim.modules.network.inputNetworkFile",
        ):
            reference_policies[key] = BeamReferencePolicy(
                identity_policy="output_or_runtime_ignored",
                required=False,
                reason="generated_from_declared_r5_input_before_beam_execution",
            )
        runtime_output_keys = (
            "beam.router.skim.activity-sim-skimmer.fileBaseName",
            "beam.router.skim.drive-time-skimmer.fileBaseName",
            "beam.router.skim.origin-destination-skimmer.fileBaseName",
            "beam.router.skim.taz-skimmer.fileBaseName",
            "beam.router.skim.transit-crowding-skimmer.fileBaseName",
        )
        for key in runtime_output_keys:
            reference_policies[key] = BeamReferencePolicy(
                identity_policy="output_or_runtime_ignored",
                required=False,
                reason="runtime_output_prefix_not_identity_input",
            )

        return BeamConfigAdapter(
            root_dirs=[config_root],
            primary_config=primary_config,
            env_overrides=beam_config_env_overrides(
                settings,
                config_root=config_root,
            ),
            path_aliases=path_aliases,
            reference_policies=reference_policies,
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
        if model == "urbansim_run":
            workspace = _workspace(ctx)
            if workspace is not None:
                static_identity_inputs = [
                    (f"urbansim_static/{relpath}", source_path)
                    for relpath, source_path in selected_urbansim_static_input_sources(
                        settings,
                        workspace,
                    )
                ]
                resolved["identity_inputs"] = [
                    *list(resolved.get("identity_inputs") or []),
                    *static_identity_inputs,
                ]
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
                atlas_runtime_identity.update(_atlas_usim_households_identity(ctx))
            if atlas_runtime_identity:
                resolved["config"] = {
                    **dict(resolved.get("config") or {}),
                    **atlas_runtime_identity,
                }
                resolved["facet"] = {
                    **dict(resolved.get("facet") or {}),
                    **atlas_runtime_identity,
                }
            if model == "atlas_run":
                static_identity_inputs = [
                    (f"atlas_static/{relpath}", source_path)
                    for relpath, source_path in selected_atlas_static_input_sources(
                        settings
                    )
                ]
                if static_identity_inputs:
                    resolved["identity_inputs"] = [
                        *list(resolved.get("identity_inputs") or []),
                        *static_identity_inputs,
                    ]
        adapter = _adapter(ctx)
        if adapter is not None:
            resolved["adapter"] = adapter
            identity_inputs = _filter_adapter_covered_identity_inputs(
                resolved.get("identity_inputs"),
                model=model,
            )
            if identity_inputs:
                resolved["identity_inputs"] = identity_inputs
            else:
                resolved.pop("identity_inputs", None)
        if model == "beam_preprocess":
            closure = _runtime_value(ctx, "beam_preprocess_identity_closure")
            if isinstance(closure, dict):
                resolved["config"] = {
                    **dict(resolved.get("config") or {}),
                    "beam_preprocess_identity": closure,
                }
        contract_report = _contract_report(
            adapter=adapter,
            payload=resolved.get("config"),
        )
        if contract_report is not None:
            resolved["facet"] = {
                **dict(resolved.get("facet") or {}),
                "native_input_contract": contract_report,
            }
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
