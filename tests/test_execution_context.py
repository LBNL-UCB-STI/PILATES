"""
Tests for ExecutionContext protocol.

This module verifies that:
1. The ExecutionContext protocol is correctly defined
2. WorkflowState satisfies the ExecutionContext protocol
3. Simple custom objects can satisfy the protocol
4. The protocol validation works correctly
"""

import os
import tempfile

import pytest
from types import SimpleNamespace
from dataclasses import dataclass

import yaml

from pilates.config import PilatesConfig, load_config
from pilates.generic.execution_context import ExecutionContext, validate_context
from workflow_state import WorkflowState


def create_config(dummy_config_content) -> PilatesConfig:
    # # Create a dummy settings.yaml for PilatesConfig
    # dummy_config_content = {
    #     "run": {
    #         "region": "test",
    #         "scenario": "test",
    #         "start_year": 2020,
    #         "end_year": 2020,
    #         "output_directory": self.temp_dir,
    #         "output_run_name": "test_run",
    #         "models": {
    #             "land_use": None,
    #             "travel": None,
    #             "activity_demand": None,
    #             "vehicle_ownership": None,
    #         },
    #     },
    #     "shared": {
    #         "geography": {
    #             "FIPS": {"county": ["06001"]},
    #             "local_crs": "EPSG:32048",
    #         },
    #         "skims": {
    #             "zone_type": "taz",
    #             "fname": "skims.h5",
    #             "geoms_fname": "geoms.geojson",
    #             "geoms_index_col": "TAZ",
    #         },
    #         "database": {
    #             "enabled": True,
    #             "type": "duckdb",
    #             "path": self.db_path,
    #         },
    #     },
    #     "infrastructure": {
    #         "container_manager": "docker",
    #         "singularity_images": {},
    #         "docker_images": {},
    #         "docker_config": {"stdout": False, "pull_latest": False},
    #     },
    # }
    temp_dir = tempfile.mkdtemp(prefix="execution_context_test_")
    # Write the dummy config to a temporary YAML file
    dummy_config_path = os.path.join(temp_dir, "dummy_settings.yaml")
    with open(dummy_config_path, "w") as f:
        yaml.dump(dummy_config_content, f)

    # Load the config using load_config to get a PilatesConfig object
    settings = load_config(dummy_config_path)
    return settings


def test_workflow_state_satisfies_protocol():
    """Verify that WorkflowState automatically satisfies ExecutionContext protocol."""
    # Create minimal settings for WorkflowState
    dummy_config_content = {
        "run": {
            "start_year": 2020,
            "end_year": 2030,
            "scenario": "test",
            "region": "test",
            "output_run_name": "test_run",
            "state_file_loc": "/tmp/test_state.yaml",
            "output_directory": "/tmp",
            "models": {
                "land_use": None,
                "travel": None,
                "activity_demand": None,
                "vehicle_ownership": None,
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["06001"]},
                "local_crs": "EPSG:32048",
            },
            "skims": {
                "zone_type": "taz",
                "fname": "skims.h5",
                "geoms_fname": "geoms.geojson",
                "geoms_index_col": "TAZ",
            },
            "database": {
                "enabled": True,
                "type": "duckdb",
                "path": "/tmp",
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
    }

    settings = create_config(dummy_config_content)

    state = WorkflowState.from_settings(settings)

    # Should satisfy protocol
    assert isinstance(state, ExecutionContext)

    # Should have all required attributes
    assert hasattr(state, "current_year")
    assert hasattr(state, "current_major_stage")
    assert hasattr(state, "current_inner_iter")

    # Attributes should be accessible and return expected types
    assert isinstance(state.current_year, (int, type(None)))
    assert state.current_inner_iter is not None  # Should be int

    # Validation should pass
    validate_context(state)


def test_simple_namespace_satisfies_protocol():
    """Verify that simple namespace objects can satisfy the protocol."""
    context = SimpleNamespace(
        current_year=2025, current_major_stage="preprocessing", current_inner_iter=0
    )

    # Should satisfy protocol
    assert isinstance(context, ExecutionContext)

    # Validation should pass
    validate_context(context)


def test_dataclass_satisfies_protocol():
    """Verify that dataclasses can satisfy the protocol."""

    @dataclass
    class CustomContext:
        current_year: int = 2025
        current_major_stage: str = "analysis"
        current_inner_iter: int = 0

    context = CustomContext()

    # Should satisfy protocol
    assert isinstance(context, ExecutionContext)

    # Validation should pass
    validate_context(context)


def test_protocol_with_none_values():
    """Verify that None values are acceptable for optional context."""
    context = SimpleNamespace(
        current_year=None, current_major_stage=None, current_inner_iter=None
    )

    # Should still satisfy protocol
    assert isinstance(context, ExecutionContext)
    validate_context(context)


def test_protocol_rejects_incomplete_object():
    """Verify that objects missing required attributes don't satisfy the protocol."""
    # Missing current_major_stage and current_inner_iter
    incomplete = SimpleNamespace(current_year=2025)

    # @runtime_checkable Protocol actually checks for attributes at isinstance() time
    # So isinstance() should return False for incomplete objects
    assert not isinstance(incomplete, ExecutionContext)

    # And validation should also fail with TypeError
    with pytest.raises(
        TypeError, match="Context must satisfy ExecutionContext protocol"
    ):
        validate_context(incomplete)


def test_protocol_with_enum_stage():
    """Verify that Enum values work for current_major_stage."""
    from enum import Enum

    class Stage(Enum):
        PREPROCESSING = "preprocessing"
        ANALYSIS = "analysis"

    context = SimpleNamespace(
        current_year=2025, current_major_stage=Stage.PREPROCESSING, current_inner_iter=0
    )

    # Should satisfy protocol (Enum is acceptable)
    assert isinstance(context, ExecutionContext)
    validate_context(context)


def test_protocol_with_property_methods():
    """Verify that objects with property methods satisfy the protocol."""

    class PropertyContext:
        def __init__(self, year, stage, iteration):
            self._year = year
            self._stage = stage
            self._iteration = iteration

        @property
        def current_year(self):
            return self._year

        @property
        def current_major_stage(self):
            return self._stage

        @property
        def current_inner_iter(self):
            return self._iteration

    context = PropertyContext(2025, "testing", 1)

    # Should satisfy protocol
    assert isinstance(context, ExecutionContext)
    validate_context(context)

    # Values should be accessible
    assert context.current_year == 2025
    assert context.current_major_stage == "testing"
    assert context.current_inner_iter == 1
