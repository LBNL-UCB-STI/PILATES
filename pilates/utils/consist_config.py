"""
pilates/utils/consist_config.py

Helpers for Consist's config channels:
- identity config (hashed, drives cache identity)
- facet (stored + queryable, does not affect cache identity)
- hash_inputs (hash-only file/dir digests folded into identity)

These helpers centralize how PILATES maps its Pydantic settings to Consist so
`run.py` stays readable and the mapping is testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pilates.config.models import PilatesConfig

try:
    from consist.types import HasConsistFacet
except Exception:  # pragma: no cover
    HasConsistFacet = None  # type: ignore[misc,assignment]


HashInput = Tuple[str, Path]


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
    It is required for steps that use `hash_inputs` (ActivitySim/BEAM).
    """
    model_norm = (model or "").lower()

    if model_norm == "initialization":
        return {
            "config": settings.get_initialization_signature(),
            "facet": build_scenario_facet(settings),
            "facet_schema_version": "initialization_v1",
            "facet_index": True,
        }

    if model_norm in {
        "activitysim",
        "activitysim_preprocess",
        "activitysim_run",
        "activitysim_postprocess",
        "activitysim_compile",
    }:
        if workspace_path is None:
            raise ValueError("workspace_path is required for activitysim hash_inputs.")
        return {
            "config": build_activitysim_identity_config(settings),
            "facet": build_activitysim_facet(settings),
            "hash_inputs": build_activitysim_hash_inputs(settings, workspace_path),
            "facet_schema_version": {
                "activitysim_compile": "activitysim_compile_v1",
                "activitysim_preprocess": "activitysim_preprocess_v1",
                "activitysim_run": "activitysim_run_v1",
                "activitysim_postprocess": "activitysim_postprocess_v1",
            }.get(model_norm, "activitysim_v1"),
            "facet_index": True,
        }

    if model_norm == "beam":
        if workspace_path is None:
            raise ValueError("workspace_path is required for beam hash_inputs.")
        return {
            "config": build_beam_identity_config(settings),
            "facet": build_beam_facet(settings),
            "hash_inputs": build_beam_hash_inputs(settings, workspace_path),
            "facet_schema_version": "beam_v1",
            "facet_index": True,
        }

    if model_norm == "urbansim":
        return {
            "config": build_urbansim_identity_config(settings),
            "facet": build_urbansim_facet(settings),
            "facet_schema_version": "urbansim_v1",
            "facet_index": True,
        }

    if model_norm == "atlas":
        return {
            "config": build_atlas_identity_config(settings),
            "facet": build_atlas_facet(settings),
            "facet_schema_version": "atlas_v1",
            "facet_index": True,
        }

    if model_norm == "postprocessing":
        return {
            "config": build_postprocessing_identity_config(settings),
            "facet": build_postprocessing_facet(settings),
            "facet_schema_version": "postprocessing_v1",
            "facet_index": True,
        }

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


def build_activitysim_hash_inputs(
    settings: PilatesConfig, workspace_path: str
) -> List[HashInput]:
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


def build_beam_hash_inputs(
    settings: PilatesConfig, workspace_path: str
) -> List[HashInput]:
    cfg = settings.beam
    if cfg is None:
        return []

    root = Path(workspace_path) / cfg.local_mutable_data_folder
    if not root.exists():
        raise FileNotFoundError(f"BEAM mutable data dir not found: {root}")

    conf_files = sorted([p for p in root.rglob("*.conf") if p.is_file()])
    if not conf_files:
        # Fallback: hash the root itself so identity still captures "something".
        return [("beam_conf_dir", root)]

    hash_inputs: List[HashInput] = []
    for path in conf_files:
        rel = path.relative_to(root).as_posix()
        hash_inputs.append((f"beam_conf/{rel}", path))
    return hash_inputs


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
