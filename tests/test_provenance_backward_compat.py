"""
Tests for backward compatibility of provenance tracking API.

This module verifies that:
1. The old 'state=' parameter still works
2. The new 'context=' parameter works
3. Both parameters can coexist during transition
4. ExecutionContext protocol is properly used
"""

import tempfile
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from pilates.utils.provenance import OpenLineageTracker
from workflow_state import WorkflowState


def test_backward_compat_state_parameter():
    """Verify that old 'state=' parameter still works."""
    # Create minimal settings for WorkflowState
    settings = MagicMock()
    settings.run.start_year = 2020
    settings.run.end_year = 2030
    settings.run.travel_model_freq = 5
    settings.run.land_use_freq = 1
    settings.run.vehicle_ownership_freq = 1
    settings.run.models.land_use = "urbansim"
    settings.run.models.vehicle_ownership = None
    settings.run.models.activity_demand = "activitysim"
    settings.run.models.travel = "beam"
    settings.activitysim.replan_iters = 0
    settings.warm_start_skims = False
    settings.static_skims = False
    settings.provenance.context_file_path = None
    settings.state_file_loc = "/tmp/test_state_backcompat.yaml"

    state = WorkflowState.from_settings(settings)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = OpenLineageTracker(run_id="test_backcompat", output_path=tmpdir)

        # Create a test file
        test_file = os.path.join(tmpdir, "test_input.csv")
        with open(test_file, "w") as f:
            f.write("col1,col2\n1,2\n")

        # OLD API: Use state= parameter (should still work)
        record = tracker.record_input_file(
            model="test",
            file_path=test_file,
            description="Test input",
            state=state,  # Old parameter name
        )

        assert record is not None
        assert record.year == state.current_year


def test_new_context_parameter():
    """Verify that new 'context=' parameter works."""
    # Create minimal settings for WorkflowState
    settings = MagicMock()
    settings.run.start_year = 2020
    settings.run.end_year = 2030
    settings.run.travel_model_freq = 5
    settings.run.land_use_freq = 1
    settings.run.vehicle_ownership_freq = 1
    settings.run.models.land_use = "urbansim"
    settings.run.models.vehicle_ownership = None
    settings.run.models.activity_demand = "activitysim"
    settings.run.models.travel = "beam"
    settings.activitysim.replan_iters = 0
    settings.warm_start_skims = False
    settings.static_skims = False
    settings.provenance.context_file_path = None
    settings.state_file_loc = "/tmp/test_state_context.yaml"


    state = WorkflowState.from_settings(settings)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = OpenLineageTracker(run_id="test_context", output_path=tmpdir)

        # Create a test file
        test_file = os.path.join(tmpdir, "test_input.csv")
        with open(test_file, "w") as f:
            f.write("col1,col2\n1,2\n")

        # NEW API: Use context= parameter
        record = tracker.record_input_file(
            model="test",
            file_path=test_file,
            description="Test input",
            context=state,  # New parameter name
        )

        assert record is not None
        assert record.year == state.current_year


def test_custom_context_object():
    """Verify that custom context objects work with new API."""
    # Create simple custom context
    context = SimpleNamespace(
        current_year=2025, current_major_stage="preprocessing", current_inner_iter=0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = OpenLineageTracker(run_id="test_custom_context", output_path=tmpdir)

        # Create a test file
        test_file = os.path.join(tmpdir, "test_input.csv")
        with open(test_file, "w") as f:
            f.write("col1,col2\n1,2\n")

        # Use custom context object
        record = tracker.record_input_file(
            model="test", file_path=test_file, description="Test input", context=context
        )

        assert record is not None
        assert record.year == 2025


def test_no_context_works():
    """Verify that context is truly optional."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = OpenLineageTracker(run_id="test_no_context", output_path=tmpdir)

        # Create a test file
        test_file = os.path.join(tmpdir, "test_input.csv")
        with open(test_file, "w") as f:
            f.write("col1,col2\n1,2\n")

        # No context provided - should work fine
        record = tracker.record_input_file(
            model="test",
            file_path=test_file,
            description="Test input",
            # No state or context parameter!
        )

        assert record is not None
        assert record.year is None  # No context means no year


def test_context_takes_precedence():
    """Verify that context= takes precedence over state= if both provided."""
    settings = MagicMock()
    settings.run.start_year = 2020
    settings.run.end_year = 2030
    settings.run.travel_model_freq = 5
    settings.run.land_use_freq = 1
    settings.run.vehicle_ownership_freq = 1
    settings.run.models.land_use = "urbansim"
    settings.run.models.vehicle_ownership = None
    settings.run.models.activity_demand = "activitysim"
    settings.run.models.travel = "beam"
    settings.activitysim.replan_iters = 0
    settings.warm_start_skims = False
    settings.static_skims = False
    settings.provenance.context_file_path = None
    settings.state_file_loc = "/tmp/test_state_precedence.yaml"

    state = WorkflowState.from_settings(settings)

    # Custom context with different year
    context = SimpleNamespace(
        current_year=9999, current_major_stage="custom", current_inner_iter=5
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = OpenLineageTracker(run_id="test_precedence", output_path=tmpdir)

        test_file = os.path.join(tmpdir, "test_input.csv")
        with open(test_file, "w") as f:
            f.write("col1,col2\n1,2\n")

        # Provide both - context should win
        record = tracker.record_input_file(
            model="test",
            file_path=test_file,
            description="Test input",
            state=state,
            context=context,
        )

        assert record is not None
        # Should use context, not state
        assert record.year == 9999


def test_output_file_with_context():
    """Verify that record_output_file also supports context parameter."""
    context = SimpleNamespace(
        current_year=2025, current_major_stage="processing", current_inner_iter=1
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = OpenLineageTracker(run_id="test_output_context", output_path=tmpdir)

        test_file = os.path.join(tmpdir, "test_output.csv")
        with open(test_file, "w") as f:
            f.write("result1,result2\n10,20\n")

        # Use context with output file
        record = tracker.record_output_file(
            model="test",
            file_path=test_file,
            description="Test output",
            context=context,
        )

        assert record is not None
        assert record.year == 2025
