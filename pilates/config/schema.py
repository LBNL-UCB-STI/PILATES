"""
PILATES-specific configuration schema and hash scope annotations.

This module defines which configuration fields affect computational outputs
(GLOBAL scope) vs pure metadata (METADATA scope). This drives the hierarchical
hashing for provenance and caching.
"""

from typing import Dict
from pilates.generic.config_hashing import (
    FieldAnnotation,
    ModelDependencyGraph,
    annotate_fields,
)


# =============================================================================
# FIELD ANNOTATIONS
# =============================================================================

# Define hash scopes for all configuration fields
# Format: "section.subsection.field": {"hash_scope": "global|metadata", "description": "..."}

FIELD_ANNOTATIONS = {
    # -------------------------------------------------------------------------
    # RUN-LEVEL CONFIGURATION
    # -------------------------------------------------------------------------
    # GLOBAL scope - affects all models, all years
    "run.region": {
        "hash_scope": "global",
        "description": "Geographic region - determines study area for all models",
    },
    "run.scenario": {
        "hash_scope": "global",
        "description": "Scenario name - affects model inputs and assumptions",
    },
    "run.start_year": {
        "hash_scope": "global",
        "description": "First simulation year - affects initial conditions",
    },
    "run.end_year": {
        "hash_scope": "global",  # For Phase 1, treat as global. Phase 2: iteration_dependent
        "description": "Final simulation year - affects workflow length",
    },
    "run.land_use_freq": {
        "hash_scope": "global",
        "description": "Land use model frequency - affects which years it runs",
    },
    "run.travel_model_freq": {
        "hash_scope": "global",
        "description": "Travel model frequency - affects which years it runs",
    },
    "run.vehicle_ownership_freq": {
        "hash_scope": "global",
        "description": "Vehicle ownership model frequency - affects which years it runs",
    },
    "run.supply_demand_iters": {
        "hash_scope": "global",
        "description": "Supply-demand loop iterations - affects workflow structure",
    },
    "run.models": {
        "hash_scope": "global",
        "description": "Which models are enabled - fundamentally changes workflow",
    },
    # METADATA scope - never affects computation
    "run.output_directory": {
        "hash_scope": "metadata",
        "description": "Output directory - only affects file locations",
    },
    "run.output_run_name": {
        "hash_scope": "metadata",
        "description": "Human-readable run name - only for organization",
    },
    "run.local_workspace_root": {
        "hash_scope": "metadata",
        "description": "Optional node-local workspace root - affects file locations",
    },
    "run.recovery_archive_roots": {
        "hash_scope": "metadata",
        "description": "Optional long-term recovery/archive roots for post-run promotion",
    },
    "run.enable_archive_copy": {
        "hash_scope": "metadata",
        "description": "Copy logged outputs to archive root as they are produced",
    },
    "run.restart_strict": {
        "hash_scope": "metadata",
        "description": "Fail-fast strict restart preflight mode",
    },
    "run.consist_code_identity": {
        "hash_scope": "metadata",
        "description": "Consist cache code-identity mode override for step runs",
    },
    "run.consist_hashing_strategy": {
        "hash_scope": "metadata",
        "description": "Consist artifact hashing mode override for cache identity",
    },
    # -------------------------------------------------------------------------
    # SHARED CONFIGURATION
    # -------------------------------------------------------------------------
    # All shared config is GLOBAL (affects multiple models)
    "shared.geography": {
        "hash_scope": "global",
        "description": "Geographic configuration - affects all models",
    },
    "shared.skims": {
        "hash_scope": "global",
        "description": "Skim configuration - affects ActivitySim and BEAM",
    },
    "shared.database": {
        "hash_scope": "metadata",
        "description": "Database config - affects storage, not computation",
    },
    # -------------------------------------------------------------------------
    # INFRASTRUCTURE CONFIGURATION
    # -------------------------------------------------------------------------
    # Infrastructure is METADATA (affects execution environment, not results)
    "infrastructure.container_manager": {
        "hash_scope": "metadata",
        "description": "Container manager - doesn't affect model results",
    },
    "infrastructure.singularity_images": {
        "hash_scope": "metadata",
        "description": "Container images - version tracked separately via git",
    },
    "infrastructure.docker_images": {
        "hash_scope": "metadata",
        "description": "Container images - version tracked separately via git",
    },
    "infrastructure.docker_config": {
        "hash_scope": "metadata",
        "description": "Docker settings - doesn't affect model results",
    },
    # -------------------------------------------------------------------------
    # MODEL-SPECIFIC CONFIGURATIONS
    # -------------------------------------------------------------------------
    # All model-specific fields are MODEL_CONDITIONAL by default
    # (Only matter if that model is enabled)
    # UrbanSim - all fields affect computation
    "urbansim": {
        "hash_scope": "model_conditional",
        "description": "Complete UrbanSim configuration",
    },
    # ATLAS - all fields affect computation
    "atlas": {
        "hash_scope": "model_conditional",
        "description": "Complete ATLAS configuration",
    },
    # ActivitySim - all fields affect computation
    "activitysim": {
        "hash_scope": "model_conditional",
        "description": "Complete ActivitySim configuration",
    },
    # BEAM - all fields affect computation
    "beam": {
        "hash_scope": "model_conditional",
        "description": "Complete BEAM configuration",
    },
    # -------------------------------------------------------------------------
    # POSTPROCESSING
    # -------------------------------------------------------------------------
    # Postprocessing doesn't affect upstream model outputs
    "postprocessing": {
        "hash_scope": "metadata",
        "description": "Postprocessing configuration - doesn't affect model outputs",
    },
}


def get_field_annotations() -> Dict[str, FieldAnnotation]:
    """
    Get field annotations for PILATES configuration.

    Returns:
        Dictionary mapping field path → FieldAnnotation
    """
    return annotate_fields(FIELD_ANNOTATIONS)


# =============================================================================
# MODEL DEPENDENCY GRAPH
# =============================================================================


def get_dependency_graph() -> ModelDependencyGraph:
    """
    Get the PILATES model dependency graph.

    PILATES uses a linear execution order:
    urbansim → atlas → activitysim → beam

    Returns:
        Configured ModelDependencyGraph
    """
    graph = ModelDependencyGraph()

    # Define linear execution order
    # Each model depends on all previous models
    graph.set_linear_execution_order(["urbansim", "atlas", "activitysim", "beam"])

    return graph


# =============================================================================
# HASH SCOPE HELPERS
# =============================================================================


def get_global_fields() -> list[str]:
    """
    Get list of field paths that have GLOBAL hash scope.

    Returns:
        List of field paths
    """
    return [
        field_path
        for field_path, anno in FIELD_ANNOTATIONS.items()
        if anno.get("hash_scope") == "global"
    ]


def get_metadata_fields() -> list[str]:
    """
    Get list of field paths that have METADATA hash scope.

    Returns:
        List of field paths
    """
    return [
        field_path
        for field_path, anno in FIELD_ANNOTATIONS.items()
        if anno.get("hash_scope") == "metadata"
    ]


def is_model_conditional(field_path: str) -> bool:
    """
    Check if a field is model-conditional.

    Args:
        field_path: Field path (e.g., "beam.sample")

    Returns:
        True if field is model-conditional
    """
    # Model-specific sections are model-conditional
    model_sections = ["urbansim", "atlas", "activitysim", "beam"]

    for model in model_sections:
        if field_path.startswith(f"{model}.") or field_path == model:
            return True

    return False
