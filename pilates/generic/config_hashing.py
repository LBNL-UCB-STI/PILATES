"""
Generic configuration hashing framework for provenance tracking.

This module provides a flexible system for hashing configuration data based on
scope and dependencies. It's designed to be framework-agnostic and could be
extracted to a standalone provenance library.

Key concepts:
- HashScope: Classifies config fields by how they affect computation
- ConfigHasher: Computes hashes based on scope and context
- ModelDependencyGraph: Defines relationships between models
"""

import hashlib
import json
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HashScope(Enum):
    """
    Defines when/how a configuration field affects computation.

    This classification allows for fine-grained provenance tracking and
    intelligent caching of computational results.
    """

    # Always included in hashes (affects all iterations, all models)
    GLOBAL = "global"

    # Only included if specific model is enabled
    MODEL_CONDITIONAL = "model_conditional"

    # Only included for specific iterations/years (advanced - not in Phase 1)
    ITERATION_DEPENDENT = "iteration_dependent"

    # Never included in computational hashes (pure metadata)
    METADATA = "metadata"


@dataclass
class FieldAnnotation:
    """
    Annotation for a configuration field.

    Attributes:
        hash_scope: When this field affects computation
        description: Human-readable description
        condition: Optional condition for inclusion (e.g., "year >= end_year")
    """
    hash_scope: HashScope
    description: str = ""
    condition: Optional[str] = None


class ModelDependencyGraph:
    """
    Defines execution order and dependencies between models.

    This is intentionally generic to support different workflow patterns:
    - Linear: A → B → C → D
    - Parallel: A, B → C (C depends on both A and B)
    - Iterative: A ↔ B (A and B iterate)

    For PILATES: urbansim → atlas → activitysim → beam
    """

    def __init__(self):
        """Initialize empty dependency graph."""
        self._dependencies: Dict[str, List[str]] = {}
        self._execution_order: List[str] = []

    def add_model(self, model_name: str, depends_on: Optional[List[str]] = None):
        """
        Add a model to the dependency graph.

        Args:
            model_name: Name of the model
            depends_on: List of models this model depends on (upstream)
        """
        if depends_on is None:
            depends_on = []

        self._dependencies[model_name] = depends_on

        # Update execution order (topological sort would be more robust)
        if model_name not in self._execution_order:
            self._execution_order.append(model_name)

    def set_linear_execution_order(self, models: List[str]):
        """
        Set a linear execution order (each model depends on all previous).

        Args:
            models: List of model names in execution order

        Example:
            graph.set_linear_execution_order(['urbansim', 'atlas', 'activitysim', 'beam'])
            # urbansim → atlas → activitysim → beam
        """
        self._execution_order = models.copy()

        for i, model in enumerate(models):
            # Each model depends on all previous models
            self._dependencies[model] = models[:i]

    def get_upstream_dependencies(self, model_name: str) -> List[str]:
        """
        Get all models that must run before this model.

        Args:
            model_name: Name of the model

        Returns:
            List of upstream model names in execution order
        """
        return self._dependencies.get(model_name, []).copy()

    def get_execution_order(self) -> List[str]:
        """Get the complete execution order."""
        return self._execution_order.copy()


class ConfigHasher:
    """
    Generic configuration hasher with scope-based filtering.

    This class is framework-agnostic and could be used by any workflow system
    that needs intelligent configuration hashing for provenance and caching.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        field_annotations: Dict[str, FieldAnnotation],
        dependency_graph: Optional[ModelDependencyGraph] = None
    ):
        """
        Initialize configuration hasher.

        Args:
            config: Full configuration dictionary
            field_annotations: Mapping of field paths to annotations
            dependency_graph: Optional model dependency graph
        """
        self.config = config
        self.field_annotations = field_annotations
        self.dependency_graph = dependency_graph or ModelDependencyGraph()

    def get_base_hash(self) -> str:
        """
        Get hash of all GLOBAL-scope fields.

        This represents configuration that affects all models in all iterations.

        Returns:
            SHA256 hash of global configuration
        """
        global_fields = self._extract_fields_by_scope(HashScope.GLOBAL)
        return self._hash_dict(global_fields)

    def get_model_hash(
        self,
        model_name: str,
        iteration: Optional[Any] = None,
        include_upstream: bool = True
    ) -> str:
        """
        Get hash of configuration affecting a specific model.

        Args:
            model_name: Name of model (e.g., 'beam', 'activitysim')
            iteration: Optional iteration identifier (e.g., year, step number)
            include_upstream: Whether to include upstream model configs in hash

        Returns:
            SHA256 hash of relevant configuration
        """
        relevant_config = {}

        # 1. Always include GLOBAL fields
        relevant_config['global'] = self._extract_fields_by_scope(HashScope.GLOBAL)

        # 2. Include iteration-dependent fields if applicable (Phase 2)
        if iteration is not None:
            iteration_fields = self._extract_iteration_dependent_fields(iteration)
            if iteration_fields:
                relevant_config['iteration_dependent'] = iteration_fields

        # 3. Include model-specific configuration
        model_config = self._extract_model_config(model_name)
        if model_config:
            relevant_config[model_name] = model_config

        # 4. Include upstream dependencies (if requested)
        if include_upstream:
            upstream_models = self.dependency_graph.get_upstream_dependencies(model_name)
            for upstream_model in upstream_models:
                upstream_config = self._extract_model_config(upstream_model)
                if upstream_config:
                    relevant_config[f'upstream_{upstream_model}'] = upstream_config

        return self._hash_dict(relevant_config)

    def get_hierarchical_hashes(
        self,
        enabled_models: List[str],
        iteration: Optional[Any] = None
    ) -> Dict[str, str]:
        """
        Get hierarchical hashes for all enabled models.

        This is the main method for Phase 1 implementation.

        Args:
            enabled_models: List of model names that are enabled
            iteration: Optional iteration identifier

        Returns:
            Dictionary mapping model name → config hash
        """
        hashes = {}

        # Base hash (affects everything)
        hashes['base'] = self.get_base_hash()

        # Model-specific hashes
        for model in enabled_models:
            hashes[model] = self.get_model_hash(
                model,
                iteration=iteration,
                include_upstream=True
            )

        return hashes

    def _extract_fields_by_scope(
        self,
        scope: HashScope,
        config_section: Optional[Dict] = None,
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Extract all fields matching a specific hash scope.

        Args:
            scope: Hash scope to filter by
            config_section: Config section to search (defaults to root)
            prefix: Current path prefix for nested fields

        Returns:
            Dictionary of matching fields
        """
        if config_section is None:
            config_section = self.config

        result = {}

        for key, value in config_section.items():
            field_path = f"{prefix}.{key}" if prefix else key
            annotation = self.field_annotations.get(field_path)

            if annotation and annotation.hash_scope == scope:
                result[key] = value
            elif isinstance(value, dict):
                # Recurse into nested config
                nested = self._extract_fields_by_scope(
                    scope,
                    value,
                    prefix=field_path
                )
                if nested:
                    result[key] = nested

        return result

    def _extract_iteration_dependent_fields(
        self,
        iteration: Any
    ) -> Dict[str, Any]:
        """
        Extract iteration-dependent fields (Phase 2 - placeholder for now).

        Args:
            iteration: Current iteration (e.g., year, step)

        Returns:
            Dictionary of relevant iteration-dependent fields
        """
        # Phase 2 implementation would evaluate conditions here
        # For Phase 1, return empty dict
        return {}

    def _extract_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Extract configuration section for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model-specific configuration dictionary
        """
        # Look for model config in top-level sections
        return self.config.get(model_name, {})

    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """
        Create deterministic SHA256 hash of a dictionary.

        Args:
            d: Dictionary to hash

        Returns:
            Hexadecimal hash string
        """
        # Use sort_keys=True for deterministic ordering
        json_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class ConfigHashRegistry:
    """
    Registry for managing config hashers across a workflow.

    This provides a convenient way to access hashing functionality
    without passing the hasher around everywhere.
    """

    _instance: Optional['ConfigHashRegistry'] = None

    def __init__(self):
        """Initialize registry."""
        self._hasher: Optional[ConfigHasher] = None

    @classmethod
    def get_instance(cls) -> 'ConfigHashRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_hasher(self, hasher: ConfigHasher):
        """Register a config hasher."""
        self._hasher = hasher

    def get_hasher(self) -> Optional[ConfigHasher]:
        """Get registered hasher."""
        return self._hasher

    def get_model_hash(self, model_name: str, **kwargs) -> Optional[str]:
        """Convenience method to get model hash."""
        if self._hasher:
            return self._hasher.get_model_hash(model_name, **kwargs)
        return None


# Convenience functions for common operations

def create_linear_dependency_graph(models: List[str]) -> ModelDependencyGraph:
    """
    Create a linear dependency graph where each model depends on all previous.

    Args:
        models: List of model names in execution order

    Returns:
        Configured ModelDependencyGraph
    """
    graph = ModelDependencyGraph()
    graph.set_linear_execution_order(models)
    return graph


def annotate_fields(
    annotations: Dict[str, Dict[str, Any]]
) -> Dict[str, FieldAnnotation]:
    """
    Convert simple annotation dict to FieldAnnotation objects.

    Args:
        annotations: Dict mapping field path → annotation dict

    Returns:
        Dict mapping field path → FieldAnnotation

    Example:
        annotations = {
            'run.region': {'hash_scope': 'global', 'description': 'Region name'},
            'run.output_name': {'hash_scope': 'metadata', 'description': 'Run name'}
        }
        field_annotations = annotate_fields(annotations)
    """
    result = {}

    for field_path, anno_dict in annotations.items():
        scope_str = anno_dict.get('hash_scope', 'global')

        # Convert string to HashScope enum
        if isinstance(scope_str, str):
            scope = HashScope[scope_str.upper()]
        else:
            scope = scope_str

        result[field_path] = FieldAnnotation(
            hash_scope=scope,
            description=anno_dict.get('description', ''),
            condition=anno_dict.get('condition')
        )

    return result
