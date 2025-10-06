#!/usr/bin/env python3
"""
Tests for Phase 2 provenance tracking improvements.

Tests the two improvements added in Phase 2:
4. record_input_from_previous_output() method
6. ProvenanceQuery API
"""

import os
import tempfile
import pytest
from pathlib import Path

from pilates.utils.provenance import FileProvenanceTracker
from pilates.utils.provenance_queries import ProvenanceQuery
from workflow_state import WorkflowState


class TestPhase2Improvements:
    """Test Phase 2 provenance improvements."""

    def test_record_input_from_previous_output_basic(self, tmp_path):
        """Test basic functionality of record_input_from_previous_output."""
        print("\n🧪 Testing record_input_from_previous_output...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        # Create a file
        h5_file = tmp_path / "model_data.h5"
        h5_file.write_text("data")

        # Model 1 creates the file
        run1 = tracker.start_model_run("urbansim", 2017, 0)
        output_rec = tracker.record_output_file(
            "urbansim",
            str(h5_file),
            short_name="usim_h5",
            model_run_id=run1,
        )
        tracker.complete_model_run(run1, output_records=[output_rec])

        # Model 2 reads it using the new method
        run2 = tracker.start_model_run("atlas_preprocessor", 2017, 0)
        input_rec = tracker.record_input_from_previous_output(
            "atlas_preprocessor",
            str(h5_file),
            producing_model="urbansim",  # Optional validation
            model_run_id=run2,
        )

        # Should reuse the same FileRecord
        assert input_rec == output_rec
        print("   ✅ Reused existing FileRecord from urbansim")

        # Both models should be listed
        assert "urbansim" in input_rec.models
        assert "atlas_preprocessor" in input_rec.models
        print(f"   ✅ Both models listed: {input_rec.models}")

        tracker.complete_model_run(run2)

    def test_record_input_from_previous_output_validation_warning(self, tmp_path, caplog):
        """Test that validation warns if expected producing model doesn't match."""
        print("\n🧪 Testing producing model validation...")

        import logging
        caplog.set_level(logging.WARNING)

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        h5_file = tmp_path / "model_data.h5"
        h5_file.write_text("data")

        # UrbanSim creates the file
        run1 = tracker.start_model_run("urbansim", 2017, 0)
        output_rec = tracker.record_output_file(
            "urbansim",
            str(h5_file),
            short_name="usim_h5",
            model_run_id=run1,
        )
        tracker.complete_model_run(run1, output_records=[output_rec])

        # ATLAS expects file from a different model
        run2 = tracker.start_model_run("atlas", 2017, 0)
        input_rec = tracker.record_input_from_previous_output(
            "atlas",
            str(h5_file),
            producing_model="atlas_postprocessor",  # WRONG! It was urbansim
            model_run_id=run2,
        )

        # Should log a warning
        assert any(
            "was expected to be produced by" in record.message
            for record in caplog.records
        ), "Should warn about unexpected producing model"

        print("   ✅ Warning logged for unexpected producing model")

    def test_provenance_query_find_outputs(self, tmp_path):
        """Test ProvenanceQuery.find_outputs_of_model()."""
        print("\n🧪 Testing ProvenanceQuery.find_outputs_of_model...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        # Create some files
        for i in range(3):
            file = tmp_path / f"output{i}.csv"
            file.write_text(f"data{i}")

            run_hash = tracker.start_model_run("atlas", 2017, i)
            output_rec = tracker.record_output_file(
                "atlas",
                str(file),
                short_name=f"atlas_output{i}",
                model_run_id=run_hash,
            )
            tracker.complete_model_run(run_hash, output_records=[output_rec])

        # Query for ATLAS outputs
        query = ProvenanceQuery(tracker.run_info)
        atlas_outputs = query.find_outputs_of_model("atlas", year=2017)

        assert len(atlas_outputs) == 3
        print(f"   ✅ Found {len(atlas_outputs)} ATLAS outputs")

        # Check names
        output_names = {rec.short_name for rec in atlas_outputs}
        assert "atlas_output0" in output_names
        print(f"   ✅ Output names: {output_names}")

    def test_provenance_query_trace_lineage(self, tmp_path):
        """Test ProvenanceQuery.trace_file_lineage()."""
        print("\n🧪 Testing ProvenanceQuery.trace_file_lineage...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        # Create a lineage chain: input -> intermediate -> output
        input_file = tmp_path / "input.csv"
        intermediate_file = tmp_path / "intermediate.csv"
        output_file = tmp_path / "output.csv"

        for f in [input_file, intermediate_file, output_file]:
            f.write_text("data")

        # Step 1: Create input
        run1 = tracker.start_model_run("model1", 2017, 0)
        input_rec = tracker.record_output_file(
            "model1",
            str(input_file),
            short_name="input",
            model_run_id=run1,
        )
        tracker.complete_model_run(run1, output_records=[input_rec])

        # Step 2: Create intermediate (from input)
        run2 = tracker.start_model_run("model2", 2017, 0)
        intermediate_rec = tracker.record_output_file(
            "model2",
            str(intermediate_file),
            source_file_paths=[str(input_file)],
            short_name="intermediate",
            model_run_id=run2,
        )
        tracker.complete_model_run(run2, output_records=[intermediate_rec])

        # Step 3: Create output (from intermediate)
        run3 = tracker.start_model_run("model3", 2017, 0)
        output_rec = tracker.record_output_file(
            "model3",
            str(output_file),
            source_file_paths=[str(intermediate_file)],
            short_name="output",
            model_run_id=run3,
        )
        tracker.complete_model_run(run3, output_records=[output_rec])

        # Trace lineage of output file
        query = ProvenanceQuery(tracker.run_info)
        lineage = query.trace_file_lineage(short_name="output")

        assert lineage is not None
        assert lineage["file"].short_name == "output"
        assert len(lineage["ancestors"]) == 2  # intermediate and input
        print(f"   ✅ Found {len(lineage['ancestors'])} ancestors")

        ancestor_names = {a.short_name for a in lineage["ancestors"]}
        assert "intermediate" in ancestor_names
        assert "input" in ancestor_names
        print(f"   ✅ Ancestors: {ancestor_names}")

    def test_provenance_query_summary_statistics(self, tmp_path):
        """Test ProvenanceQuery.get_summary_statistics()."""
        print("\n🧪 Testing ProvenanceQuery.get_summary_statistics...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        # Create some files from different models
        for model_name in ["urbansim", "atlas", "activitysim"]:
            file = tmp_path / f"{model_name}_output.csv"
            file.write_text("data")

            run_hash = tracker.start_model_run(model_name, 2017, 0)
            output_rec = tracker.record_output_file(
                model_name,
                str(file),
                short_name=f"{model_name}_output",
                model_run_id=run_hash,
            )
            tracker.complete_model_run(run_hash, output_records=[output_rec])

        # Get summary
        query = ProvenanceQuery(tracker.run_info)
        summary = query.get_summary_statistics()

        assert summary["total_files"] == 3
        assert summary["total_model_runs"] == 3
        assert len(summary["models_used"]) == 3
        assert "urbansim" in summary["models_used"]

        print(f"   ✅ Total files: {summary['total_files']}")
        print(f"   ✅ Total runs: {summary['total_model_runs']}")
        print(f"   ✅ Models used: {summary['models_used']}")

    def test_provenance_query_find_broken_source_paths(self, tmp_path):
        """Test ProvenanceQuery.find_broken_source_paths()."""
        print("\n🧪 Testing ProvenanceQuery.find_broken_source_paths...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        output_file = tmp_path / "output.csv"
        output_file.write_text("data")

        run_hash = tracker.start_model_run("test_model", 2017, 0)

        # Create output with broken source reference
        output_rec = tracker.record_output_file(
            "test_model",
            str(output_file),
            source_file_paths=["/fake/nonexistent/file.csv"],
            short_name="output",
            model_run_id=run_hash,
        )
        tracker.complete_model_run(run_hash, output_records=[output_rec])

        # Find broken references
        query = ProvenanceQuery(tracker.run_info)
        broken = query.find_broken_source_paths()

        assert len(broken) > 0
        assert broken[0]["file"] == "output"
        assert "nonexistent" in broken[0]["missing_source"]

        print(f"   ✅ Found {len(broken)} broken source path(s)")
        print(f"   📋 Broken: {broken[0]['missing_source']}")


if __name__ == "__main__":
    # Run tests standalone
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
