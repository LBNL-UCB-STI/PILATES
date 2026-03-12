"""
pilates/utils/consist_config.py

Helpers for Consist's config channels:
- identity config (hashed, drives cache identity)
- facet (stored + queryable, does not affect cache identity)
- identity_inputs (file/dir digests folded into identity)

These helpers centralize how PILATES maps its Pydantic settings to Consist so
`run.py` stays readable and the mapping is testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pilates.config.models import PilatesConfig
from pilates.workflows.catalog import provenance_builder_key_for_model_name

try:
    from consist.types import HasConsistFacet
except Exception:  # pragma: no cover
    HasConsistFacet = None  # type: ignore[misc,assignment]


IdentityInput = Tuple[str, Path]


class ConsistConfigBuilder(Protocol):
    """
    Protocol for building Consist step config/facet/identity inputs per model.

    Each builder owns the identity config (hashed), facet (queryable), and any
    identity inputs that should be folded into the step signature.
    """

    @property
    def requires_workspace_path(self) -> bool:
        """Whether identity input construction requires a workspace path."""

    def build_identity_config(self, settings: PilatesConfig) -> Dict[str, Any]:
        """Config dict that drives cache identity."""

    def build_facet(self, settings: PilatesConfig) -> Dict[str, Any]:
        """Facet dict stored for querying."""

    def build_identity_inputs(
        self, settings: PilatesConfig, workspace_path: str
    ) -> List[IdentityInput]:
        """File/dir digests folded into identity."""

    def get_facet_schema_version(self, model: str) -> str:
        """Facet schema version for the step model identifier."""


class ActivitySimConfigBuilder:
    @property
    def requires_workspace_path(self) -> bool:
        return True

    def build_identity_config(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_activitysim_identity_config(settings)

    def build_facet(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_activitysim_facet(settings)

    def build_identity_inputs(
        self, settings: PilatesConfig, workspace_path: str
    ) -> List[IdentityInput]:
        return build_activitysim_identity_inputs(settings, workspace_path)

    def get_facet_schema_version(self, model: str) -> str:
        return {
            "activitysim_compile": "activitysim_compile_v1",
            "activitysim_preprocess": "activitysim_preprocess_v1",
            "activitysim_run": "activitysim_run_v1",
            "activitysim_postprocess": "activitysim_postprocess_v1",
        }.get(model, "activitysim_v1")


class BeamConfigBuilder:
    @property
    def requires_workspace_path(self) -> bool:
        return True

    def build_identity_config(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_beam_identity_config(settings)

    def build_facet(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_beam_facet(settings)

    def build_identity_inputs(
        self, settings: PilatesConfig, workspace_path: str
    ) -> List[IdentityInput]:
        return build_beam_identity_inputs(settings, workspace_path)

    def get_facet_schema_version(self, model: str) -> str:
        return {
            "beam_preprocess": "beam_preprocess_v1",
            "beam_run": "beam_run_v1",
            "beam_postprocess": "beam_postprocess_v1",
            "beam_full_skim": "beam_full_skim_v1",
        }.get(model, "beam_v1")


class UrbanSimConfigBuilder:
    @property
    def requires_workspace_path(self) -> bool:
        return False

    def build_identity_config(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_urbansim_identity_config(settings)

    def build_facet(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_urbansim_facet(settings)

    def build_identity_inputs(
        self, settings: PilatesConfig, workspace_path: str
    ) -> List[IdentityInput]:
        return []

    def get_facet_schema_version(self, model: str) -> str:
        return {
            "urbansim_preprocess": "urbansim_preprocess_v1",
            "urbansim_run": "urbansim_run_v1",
            "urbansim_postprocess": "urbansim_postprocess_v1",
        }.get(model, "urbansim_v1")


class AtlasConfigBuilder:
    @property
    def requires_workspace_path(self) -> bool:
        return False

    def build_identity_config(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_atlas_identity_config(settings)

    def build_facet(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_atlas_facet(settings)

    def build_identity_inputs(
        self, settings: PilatesConfig, workspace_path: str
    ) -> List[IdentityInput]:
        return []

    def get_facet_schema_version(self, model: str) -> str:
        return {
            "atlas_preprocess": "atlas_preprocess_v1",
            "atlas_run": "atlas_run_v1",
            "atlas_postprocess": "atlas_postprocess_v1",
        }.get(model, "atlas_v1")


class PostprocessingConfigBuilder:
    @property
    def requires_workspace_path(self) -> bool:
        return False

    def build_identity_config(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_postprocessing_identity_config(settings)

    def build_facet(self, settings: PilatesConfig) -> Dict[str, Any]:
        return build_postprocessing_facet(settings)

    def build_identity_inputs(
        self, settings: PilatesConfig, workspace_path: str
    ) -> List[IdentityInput]:
        return []

    def get_facet_schema_version(self, model: str) -> str:
        return "postprocessing_v1"


_CONFIG_BUILDERS: Dict[str, ConsistConfigBuilder] = {
    "activitysim": ActivitySimConfigBuilder(),
    "beam": BeamConfigBuilder(),
    "urbansim": UrbanSimConfigBuilder(),
    "atlas": AtlasConfigBuilder(),
    "postprocessing": PostprocessingConfigBuilder(),
}


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


def build_scenario_facet(settings: PilatesConfig) -> Dict[str, Any]:
    """
    Scenario facet: store "execution-governing" config that is useful to inspect
    and query, but should not affect cache identity.
    """
    run_cfg = getattr(settings, "run", None)
    if run_cfg is not None:
        if HasConsistFacet is not None and isinstance(run_cfg, HasConsistFacet):
            run_cfg = run_cfg.to_consist_facet()
        elif hasattr(run_cfg, "to_consist_facet") and callable(
            run_cfg.to_consist_facet
        ):
            run_cfg = run_cfg.to_consist_facet()
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

    if model_norm == "initialization":
        return {
            "config": settings.get_initialization_signature(),
            "facet": build_scenario_facet(settings),
            "facet_schema_version": "initialization_v1",
            "facet_index": True,
        }

    builder_key = provenance_builder_key_for_model_name(model_norm)
    if builder_key is not None:
        builder = _CONFIG_BUILDERS.get(builder_key)
        if builder is None:
            raise ValueError(
                f"Unknown provenance builder key {builder_key!r} for model {model_norm!r}. "
                "Register it in _CONFIG_BUILDERS or remove catalog provenance metadata."
            )
    else:
        # Fallback for non-catalog (or provenance-unspecified) step models.
        builder_key = model_norm.split("_")[0]
        builder = _CONFIG_BUILDERS.get(builder_key)

    if builder is not None:
        if builder.requires_workspace_path and workspace_path is None:
            raise ValueError(
                f"workspace_path is required for {builder_key} identity_inputs."
            )
        result: Dict[str, Any] = {
            "config": builder.build_identity_config(settings),
            "facet": builder.build_facet(settings),
            "facet_schema_version": builder.get_facet_schema_version(model_norm),
            "facet_index": True,
        }
        if workspace_path is not None:
            identity_inputs = builder.build_identity_inputs(settings, workspace_path)
            if identity_inputs:
                result["identity_inputs"] = identity_inputs
        return result

    # Default: no special config mapping yet.
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
    return {
        "household_sample_size": cfg.household_sample_size,
        "chunk_size": cfg.chunk_size,
        "num_processes": cfg.num_processes,
        "file_format": cfg.file_format,
        "warm_start_activities": cfg.warm_start_activities,
        "replan_iters": cfg.replan_iters,
        "replan_hh_samp_size": cfg.replan_hh_samp_size,
        "replan_after": cfg.replan_after,
        "random_seed": cfg.random_seed,
        "database": (
            {
                "enabled": cfg.database.enabled,
                "use_processed_data": cfg.database.use_processed_data,
                "year": cfg.database.year,
            }
            if getattr(cfg, "database", None) is not None
            else None
        ),
    }


def build_activitysim_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.activitysim
    if cfg is None:
        return {}
    if HasConsistFacet is not None and isinstance(cfg, HasConsistFacet):
        return cfg.to_consist_facet()
    if hasattr(cfg, "to_consist_facet") and callable(cfg.to_consist_facet):
        return cfg.to_consist_facet()
    # Fallback: shallow identity-like facet
    return build_activitysim_identity_config(settings)


def build_activitysim_identity_inputs(
    settings: PilatesConfig, workspace_path: str
) -> List[IdentityInput]:
    cfg = settings.activitysim
    if cfg is None:
        return []

    configs_dir = Path(workspace_path) / cfg.local_mutable_configs_folder
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"ActivitySim mutable configs dir not found: {configs_dir}"
        )

    return [("asim_mutable_configs", configs_dir)]


def build_beam_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.beam
    if cfg is None:
        return {}
    return {
        "config": cfg.config,
        "sample": cfg.sample,
        "replanning_portion": cfg.replanning_portion,
        "memory": cfg.memory,
        "discard_plans_every_year": cfg.discard_plans_every_year,
        "max_plans_memory": cfg.max_plans_memory,
        "router_directory": cfg.router_directory,
        "scenario_folder": cfg.scenario_folder,
    }


def build_beam_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.beam
    if cfg is None:
        return {}
    if HasConsistFacet is not None and isinstance(cfg, HasConsistFacet):
        return cfg.to_consist_facet()
    if hasattr(cfg, "to_consist_facet") and callable(cfg.to_consist_facet):
        return cfg.to_consist_facet()
    return build_beam_identity_config(settings)


def build_beam_identity_inputs(
    settings: PilatesConfig, workspace_path: str
) -> List[IdentityInput]:
    cfg = settings.beam
    if cfg is None:
        return []

    root = Path(workspace_path) / cfg.local_mutable_data_folder
    if not root.exists():
        return []

    config_name = getattr(cfg, "config", None)
    if isinstance(config_name, str) and config_name:
        matches = sorted(root.rglob(config_name))
        if matches:
            return [(f"beam_conf/{config_name}", matches[0])]

    conf_files = sorted([p for p in root.rglob("*.conf") if p.is_file()])
    if not conf_files:
        # Fallback: hash the root itself so identity still captures "something".
        return [("beam_conf_dir", root)]

    identity_inputs: List[IdentityInput] = []
    for path in conf_files:
        rel = path.relative_to(root).as_posix()
        identity_inputs.append((f"beam_conf/{rel}", path))
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
    }


def build_urbansim_facet(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.urbansim
    if cfg is None:
        return {}
    if hasattr(cfg, "model_dump"):
        # Store the full model config as facet (queryable), but keep identity compact.
        return cfg.model_dump()
    return {}


def build_atlas_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = settings.atlas
    if cfg is None:
        return {}
    return _only_keys(
        cfg.model_dump() if hasattr(cfg, "model_dump") else {},
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
    if cfg is None:
        return {}
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    return {}


def build_postprocessing_identity_config(settings: PilatesConfig) -> Dict[str, Any]:
    cfg = getattr(settings, "postprocessing", None)
    if cfg is None:
        return {}
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    return {}


def build_postprocessing_facet(settings: PilatesConfig) -> Dict[str, Any]:
    return build_postprocessing_identity_config(settings)
