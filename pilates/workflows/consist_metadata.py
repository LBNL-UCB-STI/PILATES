"""
Consist configuration, facet, identity-input, and lazy step metadata builders.

This module centralizes:
- scenario and per-step Consist configuration/facet/identity-input mapping
  (`build_scenario_consist_kwargs`, `build_step_consist_kwargs`, and model builders)
- lazy `StepContext` metadata resolution for `@consist.define_step`
  (`consist_step_meta`, `activitysim_config_root_dirs`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from consist.core.step_context import StepContext
from consist.types import HasConsistFacet

from pilates.activitysim.config_roots import (
    configured_activitysim_main_configs_dir,
    configured_activitysim_sharrow,
    required_activitysim_config_roots,
    sync_activitysim_sharrow_configs,
)
from pilates.atlas.preprocessor import selected_atlas_static_input_sources
from pilates.beam.config_hocon import beam_config_env_overrides, beam_config_root
from pilates.config.models import (
    ActivitySimConfig,
    BeamConfig,
    FullSkimsCreatorConfig,
    PilatesConfig,
    PriorRunExpectation,
    admission_policy_fingerprint,
)
from pilates.urbansim.preprocessor import selected_urbansim_static_input_sources
from pilates.utils.settings_helper import get as get_setting
from pilates.utils.state_access import current_year, is_start_year
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_VEHICLES_IN,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.contracts import InputContract

IdentityInput = Tuple[str, Path]

_DISABLE_BEAM_CONFIG_ADAPTER_ENV = "PILATES_DISABLE_BEAM_CONFIG_ADAPTER"


def provenance_builder_key_for_step_name(step_name: str) -> Optional[str]:
    from pilates.workflows.catalog import (
        provenance_builder_key_for_step_name as _catalog_provenance_builder_key,
    )

    return _catalog_provenance_builder_key(step_name)


@runtime_checkable
class _HasFullPath(Protocol):
    full_path: str | Path


@runtime_checkable
class _HasAsimMutableConfigsDir(Protocol):
    def get_asim_mutable_configs_dir(self) -> str | Path: ...


@runtime_checkable
class _HasBeamMutableDataDir(Protocol):
    def get_beam_mutable_data_dir(self) -> str | Path: ...


@runtime_checkable
class _HasBeamOutputDir(Protocol):
    def get_beam_output_dir(self) -> str | Path: ...


@runtime_checkable
class _HasAsimOutputDir(Protocol):
    def get_asim_output_dir(self) -> str | Path: ...


@runtime_checkable
class _HasModelDump(Protocol):
    def model_dump(self) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class _StepConfigBuilder:
    identity_config_fn: Callable[[PilatesConfig], Dict[str, Any]]
    facet_fn: Callable[[PilatesConfig], Dict[str, Any]]
    default_schema_version: str
    step_schema_versions: Mapping[str, str] = field(default_factory=dict)
    identity_inputs_fn: Optional[
        Callable[[PilatesConfig, str], List[IdentityInput]]
    ] = None


def build_scenario_consist_kwargs(settings: PilatesConfig) -> Dict[str, Any]:
    """
    Build kwargs for `tracker.scenario(...)`.

    - `config`: cache identity for the scenario header
    - `facet`: queryable run metadata (execution-governing)
    """
    return {
        "config": settings.get_initialization_signature(),
        "facet": build_scenario_facet(settings),
        "facet_schema_version": "pilates_scenario_v1",
        "facet_index": True,
    }


def _coerce_consist_facet(target: Any) -> Optional[Dict[str, Any]]:
    if isinstance(target, HasConsistFacet):
        return dict(target.to_consist_facet())
    try:
        to_facet = target.to_consist_facet
    except AttributeError:
        return None
    if callable(to_facet):
        return dict(to_facet())
    return None


def build_scenario_facet(settings: PilatesConfig) -> Dict[str, Any]:
    """
    Scenario facet: store "execution-governing" config that is useful to inspect
    and query, but should not affect cache identity.
    """
    try:
        run_cfg: Any = settings.run
    except AttributeError:
        run_cfg = None
    if run_cfg is not None:
        facet = _coerce_consist_facet(run_cfg)
        if facet is not None:
            run_cfg = facet
    return {"run": run_cfg}


def build_step_consist_kwargs(
    model: str,
    settings: PilatesConfig,
    *,
    workspace_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build kwargs for `scenario.step(..., **kwargs)`.

    `workspace_path` should be the current run directory (Workspace.full_path).
    It is required for steps that use `identity_inputs` (ActivitySim/BEAM).
    """
    model_norm = (model or "").lower()

    builder_key = provenance_builder_key_for_step_name(model_norm)
    if builder_key is not None:
        builder = _CONFIG_BUILDERS.get(builder_key)
        if builder is None:
            raise ValueError(
                f"Unknown provenance builder key {builder_key!r} for model {model_norm!r}. "
                "Register it in _CONFIG_BUILDERS or remove catalog provenance metadata."
            )
    else:
        builder_key = model_norm.split("_")[0]
        builder = _CONFIG_BUILDERS.get(builder_key)

    if builder is not None:
        if builder.identity_inputs_fn is not None and workspace_path is None:
            raise ValueError(
                f"workspace_path is required for {builder_key} identity_inputs."
            )
        result: Dict[str, Any] = {
            "config": builder.identity_config_fn(settings),
            "facet": builder.facet_fn(settings),
            "facet_schema_version": builder.step_schema_versions.get(
                model_norm, builder.default_schema_version
            ),
            "facet_index": True,
        }
        if workspace_path is not None and builder.identity_inputs_fn is not None:
            identity_inputs = builder.identity_inputs_fn(settings, workspace_path)
            if identity_inputs:
                result["identity_inputs"] = identity_inputs
        return result

    return {
        "facet_schema_version": f"{model_norm or 'unknown'}_v1",
        "facet_index": True,
    }


def _only_keys(src: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: src.get(k) for k in keys if k in src}


def build_activitysim_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.activitysim
    if cfg is None:
        return {}
    database = (
        cfg.database
        if isinstance(cfg, ActivitySimConfig)
        else vars(cfg).get("database")
    )
    vo_model = get_setting(
        settings,
        "run.models.vehicle_ownership",
        default=get_setting(settings, "vehicle_ownership_model"),
    )
    atlas_vo_enabled = isinstance(vo_model, str) and vo_model.strip().lower() == "atlas"
    identity_config: Dict[str, Any] = {
        "household_sample_size": cfg.household_sample_size,
        "chunk_size": cfg.chunk_size,
        "num_processes": cfg.num_processes,
        "file_format": cfg.file_format,
        "replan_iters": cfg.replan_iters,
        "replan_hh_samp_size": cfg.replan_hh_samp_size,
        "replan_after": cfg.replan_after,
        "random_seed": cfg.random_seed,
        "main_configs_dir": configured_activitysim_main_configs_dir(cfg),
        "atlas_vehicle_ownership_enabled": atlas_vo_enabled,
        "database": (
            {
                "enabled": database.enabled,
                "use_processed_data": database.use_processed_data,
                "year": database.year,
            }
            if database is not None
            else None
        ),
    }
    sharrow = configured_activitysim_sharrow(cfg)
    if isinstance(sharrow, bool):
        identity_config["sharrow"] = sharrow
    admission = (
        cfg.admission
        if isinstance(cfg, ActivitySimConfig)
        else vars(cfg).get("admission")
    )
    initial_skims = admission.initial_skims if admission is not None else None
    if initial_skims is not None:
        identity_config["admission"] = {
            "initial_skims": admission_policy_fingerprint(initial_skims)
        }
    return identity_config


def build_activitysim_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.activitysim
    if cfg is None:
        return {}
    facet = _coerce_consist_facet(cfg)
    if facet is not None:
        return facet
    return build_activitysim_identity_config(settings)


def build_activitysim_identity_inputs(
    settings: PilatesConfig, workspace_path: str
) -> List[IdentityInput]:
    cfg = settings.activitysim
    if cfg is None:
        return []

    configs_dir = Path(workspace_path) / cfg.local_mutable_configs_folder
    if not configs_dir.exists():
        local_configs_folder = get_setting(settings, "activitysim.local_configs_folder")
        region = get_setting(settings, "run.region")
        if isinstance(local_configs_folder, str) and isinstance(region, str):
            from pilates.activitysim.preprocessor import (
                ensure_activitysim_mutable_configs,
            )
            from pilates.utils.path_utils import find_project_root

            project_root = (
                find_project_root(start_path=os.path.dirname(__file__)) or os.getcwd()
            )
            source_dir = (
                Path(local_configs_folder)
                if os.path.isabs(local_configs_folder)
                else Path(project_root) / local_configs_folder
            ) / region
            if source_dir.is_dir():
                ensure_activitysim_mutable_configs(
                    settings,
                    workspace_root=workspace_path,
                )
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"ActivitySim mutable configs dir not found: {configs_dir}"
        )

    return [("asim_mutable_configs", configs_dir)]


def build_beam_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.beam
    if cfg is None:
        return {}
    admission_config = cfg.admission
    linkstats_admission = (
        admission_config.linkstats if admission_config is not None else None
    )
    linkstats_admission_dict = admission_policy_fingerprint(linkstats_admission)
    cfg_vars = vars(cfg)
    is_typed_beam = isinstance(cfg, BeamConfig)
    identity_config: Dict[str, Any] = {
        "config": cfg.config,
        "sample": cfg.sample,
        "replanning_portion": cfg.replanning_portion,
        "memory": cfg.memory,
        "discard_plans_every_year": cfg.discard_plans_every_year,
        "max_plans_memory": cfg.max_plans_memory,
        "plan_memory_beta": (
            cfg.plan_memory_beta
            if is_typed_beam
            else cfg_vars.get("plan_memory_beta", 1.0)
        ),
        "new_plan_selection_policy": (
            cfg.new_plan_selection_policy
            if is_typed_beam
            else cfg_vars.get("new_plan_selection_policy", "force_new")
        ),
        "new_plan_initial_score": (
            cfg.new_plan_initial_score
            if is_typed_beam
            else cfg_vars.get("new_plan_initial_score", 0.0)
        ),
        "new_plan_adoption_rate": (
            cfg.new_plan_adoption_rate
            if is_typed_beam
            else cfg_vars.get("new_plan_adoption_rate", 1.0)
        ),
        "plan_selection_random_seed": (
            cfg.plan_selection_random_seed
            if is_typed_beam
            else cfg_vars.get("plan_selection_random_seed", 42)
        ),
        "write_skims_to_omx": bool(get_setting(settings, "write_skims_to_omx", False)),
        "router_directory": cfg.router_directory,
        "scenario_folder": cfg.scenario_folder,
        "linkstats_admission": linkstats_admission_dict,
    }
    full_skim = cfg.full_skim if is_typed_beam else cfg_vars.get("full_skim")
    if isinstance(full_skim, FullSkimsCreatorConfig):
        identity_config["full_skim"] = full_skim.to_consist_facet()
    return identity_config


def build_beam_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.beam
    if cfg is None:
        return {}
    facet = _coerce_consist_facet(cfg)
    if facet is not None:
        return facet
    return build_beam_identity_config(settings)


def build_beam_identity_inputs(
    settings: PilatesConfig, workspace_path: str
) -> List[IdentityInput]:
    cfg = settings.beam
    if cfg is None:
        return []

    identity_inputs: List[IdentityInput] = []
    root = Path(workspace_path) / cfg.local_mutable_data_folder
    if not root.exists():
        local_input_folder = get_setting(settings, "beam.local_input_folder")
        region = get_setting(settings, "run.region")
        if isinstance(local_input_folder, str) and isinstance(region, str):
            from pilates.beam.preprocessor import ensure_beam_mutable_workspace

            try:
                ensure_beam_mutable_workspace(settings, output_dir=str(root))
            except (ValueError, FileNotFoundError):
                pass
    if root.exists():
        config_name = cfg.config
        if isinstance(config_name, str) and config_name:
            matches = sorted(root.rglob(config_name))
            if matches:
                identity_inputs.append((f"beam_conf/{config_name}", matches[0]))
        if not identity_inputs:
            conf_files = sorted([p for p in root.rglob("*.conf") if p.is_file()])
            if conf_files:
                for path in conf_files:
                    rel = path.relative_to(root).as_posix()
                    identity_inputs.append((f"beam_conf/{rel}", path))
            else:
                identity_inputs.append(("beam_conf_dir", root))

    admission_config = cfg.admission
    linkstats_admission = (
        admission_config.linkstats if admission_config is not None else None
    )
    expectation = (
        linkstats_admission.expectation if linkstats_admission is not None else None
    )
    expected_bytes_path = (
        expectation.expected_bytes_path
        if isinstance(expectation, PriorRunExpectation)
        else None
    )
    if expected_bytes_path:
        expected_path = Path(expected_bytes_path).expanduser().resolve()
        if expected_path.is_file():
            identity_inputs.append(
                ("beam_admission_linkstats_expected_bytes", expected_path)
            )
    return identity_inputs


def build_urbansim_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.urbansim
    if cfg is None:
        return {}
    return {
        "command_template": cfg.command_template,
        "input_file_template": cfg.input_file_template,
        "input_file_template_year": cfg.input_file_template_year,
        "output_file_template": cfg.output_file_template,
        "region_id": cfg.region_id,
        "admission": (
            {
                "initial_datastore": admission_policy_fingerprint(
                    cfg.admission.initial_datastore
                )
            }
            if cfg.admission is not None
            else None
        ),
    }


def build_urbansim_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.urbansim
    if isinstance(cfg, _HasModelDump):
        return cfg.model_dump()
    return {}


def build_atlas_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.atlas
    if not isinstance(cfg, _HasModelDump):
        return {}
    return _only_keys(
        cfg.model_dump(),
        [
            "max_retries",
            "sample_size",
            "num_processes",
            "beamac",
            "mod",
            "scenario",
            "adscen",
            "rebfactor",
            "taxfactor",
            "discIncent",
            "command_template",
        ],
    )


def build_atlas_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.atlas
    if isinstance(cfg, _HasModelDump):
        return cfg.model_dump()
    return {}


def build_postprocessing_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = (
        settings.postprocessing
        if isinstance(settings, PilatesConfig)
        else vars(settings).get("postprocessing")
    )
    if isinstance(cfg, _HasModelDump):
        return cfg.model_dump()
    return {}


def build_postprocessing_facet(settings: PilatesConfig) -> Dict[str, Any]:
    return build_postprocessing_identity_config(settings)


_CONFIG_BUILDERS: Dict[str, _StepConfigBuilder] = {
    "activitysim": _StepConfigBuilder(
        identity_config_fn=build_activitysim_identity_config,
        facet_fn=build_activitysim_facet,
        identity_inputs_fn=build_activitysim_identity_inputs,
        default_schema_version="activitysim_v1",
        step_schema_versions={
            "activitysim_preprocess": "activitysim_preprocess_v1",
            "activitysim_run": "activitysim_run_v1",
            "activitysim_postprocess": "activitysim_postprocess_v1",
        },
    ),
    "beam": _StepConfigBuilder(
        identity_config_fn=build_beam_identity_config,
        facet_fn=build_beam_facet,
        identity_inputs_fn=build_beam_identity_inputs,
        default_schema_version="beam_v1",
        step_schema_versions={
            "beam_preprocess": "beam_preprocess_v1",
            "beam_run": "beam_run_v1",
            "beam_postprocess": "beam_postprocess_v1",
            "beam_full_skim": "beam_full_skim_v1",
        },
    ),
    "urbansim": _StepConfigBuilder(
        identity_config_fn=build_urbansim_identity_config,
        facet_fn=build_urbansim_facet,
        default_schema_version="urbansim_v1",
        step_schema_versions={
            "urbansim_preprocess": "urbansim_preprocess_v1",
            "urbansim_run": "urbansim_run_v1",
            "urbansim_postprocess": "urbansim_postprocess_v1",
        },
    ),
    "atlas": _StepConfigBuilder(
        identity_config_fn=build_atlas_identity_config,
        facet_fn=build_atlas_facet,
        default_schema_version="atlas_v1",
        step_schema_versions={
            "atlas_preprocess": "atlas_preprocess_v1",
            "atlas_run": "atlas_run_v1",
            "atlas_postprocess": "atlas_postprocess_v1",
        },
    ),
    "postprocessing": _StepConfigBuilder(
        identity_config_fn=build_postprocessing_identity_config,
        facet_fn=build_postprocessing_facet,
        default_schema_version="postprocessing_v1",
    ),
}


def activitysim_config_root_dirs(
    settings: ActivitySimConfig,
    mutable_configs_root: Path,
) -> tuple[Path, ...]:
    """Return ActivitySim config roots in the same order used for identity."""
    return tuple(
        config_root.path_under(mutable_configs_root)
        for config_root in required_activitysim_config_roots(settings.main_configs_dir)
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

    cached_ctx: StepContext | None = None
    cached_resolved: Dict[str, Any] | None = None

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

    def _workspace_path_from_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, _HasFullPath) and isinstance(value.full_path, (Path, str)):
            return str(value.full_path)
        if isinstance(value, (Path, str)):
            return str(value)
        return None

    def _atlas_selected_usim_h5(state: Any) -> Any:
        state_vars = vars(state)
        current_h5 = state_vars.get("atlas_usim_datastore_h5")
        base_h5 = state_vars.get("atlas_usim_datastore_base_h5")
        if is_start_year(state) and base_h5 is not None:
            return base_h5
        return current_h5 if current_h5 is not None else base_h5

    def _atlas_usim_households_identity(ctx: StepContext) -> Dict[str, Any]:
        state = ctx.get_runtime("state", default=None)
        if state is None:
            return {}
        selected_h5 = _atlas_selected_usim_h5(state)
        if selected_h5 is None:
            return {}

        from pilates.utils.coupler_helpers import artifact_to_existing_path
        from pilates.utils.usim_h5 import fingerprint_usim_h5_table

        workspace_obj = ctx.get_runtime("workspace", default=None)
        h5_path = artifact_to_existing_path(selected_h5, workspace=workspace_obj)
        if h5_path is None:
            return {
                "atlas_usim_households_identity_status": "unresolved",
                "atlas_usim_households_source": str(selected_h5),
            }

        year = current_year(state)
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
        settings = ctx.get_runtime("settings", default=None)
        if settings is None:
            return None
        activitysim_settings = (
            settings.activitysim
            if isinstance(settings, PilatesConfig)
            else vars(settings).get("activitysim")
        )
        if activitysim_settings is None:
            return None

        try:
            from consist.integrations.activitysim import ActivitySimConfigAdapter
        except Exception:
            return None

        workspace_obj = ctx.get_runtime("workspace", default=None)
        mutable_configs_root: Optional[Path] = None
        if isinstance(workspace_obj, _HasAsimMutableConfigsDir):
            mutable_configs_root = Path(workspace_obj.get_asim_mutable_configs_dir())
        else:
            ws_path = _workspace_path_from_value(workspace_obj)
            if ws_path:
                mutable_configs_root = (
                    Path(ws_path) / activitysim_settings.local_mutable_configs_folder
                )
        if mutable_configs_root is None:
            return None

        config_roots = activitysim_config_root_dirs(
            activitysim_settings, mutable_configs_root
        )
        if not all(candidate.is_dir() for candidate in config_roots):
            local_configs_folder = get_setting(
                settings, "activitysim.local_configs_folder"
            )
            region = get_setting(settings, "run.region")
            if isinstance(local_configs_folder, str) and isinstance(region, str):
                from pilates.activitysim.preprocessor import (
                    ensure_activitysim_mutable_configs,
                )

                ensure_activitysim_mutable_configs(settings, workspace=workspace_obj)
                config_roots = activitysim_config_root_dirs(
                    activitysim_settings, mutable_configs_root
                )
        if model == "activitysim_run":
            missing_roots = [
                candidate for candidate in config_roots if not candidate.is_dir()
            ]
            if missing_roots:
                missing = ", ".join(str(candidate) for candidate in missing_roots)
                raise RuntimeError(
                    "ActivitySim configuration identity is incomplete; missing roots: "
                    f"{missing}"
                )
            selected_roots = config_roots
        else:
            selected_roots = tuple(
                candidate for candidate in config_roots if candidate.is_dir()
            )
            if not selected_roots:
                return None

        sync_activitysim_sharrow_configs(
            mutable_configs_root,
            main_configs_dir=configured_activitysim_main_configs_dir(
                activitysim_settings
            ),
            sharrow=configured_activitysim_sharrow(activitysim_settings),
        )
        return ActivitySimConfigAdapter(root_dirs=list(selected_roots))

    def _beam_adapter(ctx: StepContext) -> Any:
        if os.environ.get(_DISABLE_BEAM_CONFIG_ADAPTER_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None

        settings = ctx.get_runtime("settings", default=None)
        if settings is None:
            return None
        if isinstance(settings, PilatesConfig):
            run_settings = settings.run
            beam_settings = settings.beam
        else:
            settings_vars = vars(settings)
            run_settings = settings_vars.get("run")
            beam_settings = settings_vars.get("beam")
        if run_settings is None or beam_settings is None:
            return None

        try:
            from consist.integrations.beam import BeamConfigAdapter, BeamReferencePolicy
        except Exception:
            return None

        from pilates.beam.launch_config import BeamLaunchConfig

        workspace_obj = ctx.get_runtime("workspace", default=None)
        launch_config = ctx.get_runtime("beam_launch_config", default=None)
        config_root: Optional[Path] = None
        workspace_root: Optional[Path] = None
        primary_config: Optional[Path] = None
        if isinstance(launch_config, BeamLaunchConfig):
            config_root = launch_config.root
            primary_config = launch_config.primary_config
        elif isinstance(workspace_obj, _HasBeamMutableDataDir):
            workspace_root_path = _workspace_path_from_value(workspace_obj)
            if workspace_root_path:
                workspace_root = Path(workspace_root_path)
            config_root = beam_config_root(
                settings,
                workspace=workspace_obj,
            )
        else:
            ws_path = _workspace_path_from_value(workspace_obj)
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
            local_input_folder = get_setting(settings, "beam.local_input_folder")
            region = get_setting(settings, "run.region")
            if isinstance(local_input_folder, str) and isinstance(region, str):
                from pilates.beam.preprocessor import ensure_beam_mutable_workspace

                try:
                    ensure_beam_mutable_workspace(settings, workspace=workspace_obj)
                except (ValueError, FileNotFoundError):
                    pass
        if not primary_config.exists():
            return None

        path_aliases: dict[str, Path] = {}
        if workspace_root is not None:
            path_aliases["workspace"] = workspace_root
        if workspace_obj is not None:
            if isinstance(workspace_obj, _HasBeamMutableDataDir):
                path_aliases["beam_input"] = Path(
                    workspace_obj.get_beam_mutable_data_dir()
                )
            if isinstance(workspace_obj, _HasBeamOutputDir):
                path_aliases["beam_output"] = Path(workspace_obj.get_beam_output_dir())
            if settings.activitysim is not None and isinstance(
                workspace_obj, _HasAsimOutputDir
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
            required=model not in ("beam_preprocess", "beam_full_skim"),
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
                required=model != "beam_full_skim",
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
        nonlocal cached_ctx, cached_resolved
        if cached_ctx is ctx and cached_resolved is not None:
            return cached_resolved

        settings = ctx.get_runtime("settings", default=None)
        if settings is None:
            cached_ctx = ctx
            cached_resolved = {}
            return cached_resolved
        workspace = ctx.get_runtime("workspace", default=None)
        workspace_path = _workspace_path_from_value(workspace)
        resolved = build_step_consist_kwargs(
            model=model,
            settings=settings,
            workspace_path=workspace_path,
        )
        if model == "urbansim_run":
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
            state = ctx.get_runtime("state", default=None)
            atlas_runtime_identity: Dict[str, Any] = {}
            if state is not None:
                atlas_subyear = current_year(state)
                if atlas_subyear is not None:
                    atlas_runtime_identity["atlas_subyear"] = atlas_subyear
                main_forecast_year = vars(state).get("main_forecast_year")
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
            closure = ctx.get_runtime("beam_preprocess_identity_closure", default=None)
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
        cached_ctx = ctx
        cached_resolved = resolved
        return resolved

    return {
        "adapter": lambda ctx: _resolve(ctx).get("adapter"),
        "config": lambda ctx: _resolve(ctx).get("config"),
        "facet": lambda ctx: _resolve(ctx).get("facet"),
        "facet_index": lambda ctx: _resolve(ctx).get("facet_index"),
        "facet_schema_version": lambda ctx: _resolve(ctx).get("facet_schema_version"),
        "identity_inputs": lambda ctx: _resolve(ctx).get("identity_inputs"),
    }
