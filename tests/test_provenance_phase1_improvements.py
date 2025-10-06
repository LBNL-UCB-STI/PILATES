#!/usr/bin/env python3
"""
Tests for Phase 1 provenance tracking improvements.

Tests the three improvements added in Phase 1:
1. record_output_file_with_inputs() helper method
2. Validation of source_file_paths at record time
3. validate_provenance_chain() method
"""

import os
import tempfile
import pytest
from pathlib import Path

from pilates.utils.provenance import FileProvenanceTracker
from workflow_state import WorkflowState


class TestPhase1Improvements:
    """Test Phase 1 provenance improvements."""

    def test_record_output_file_with_inputs_helper(self, tmp_path):
        """Test the record_output_file_with_inputs convenience method."""
        print("\n🧪 Testing record_output_file_with_inputs helper...")

        # Create tracker
        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        # Create some dummy input files
        input1 = tmp_path / "input1.csv"
        input2 = tmp_path / "input2.csv"
        output_file = tmp_path / "output.csv"

        input1.write_text("data1")
        input2.write_text("data2")
        output_file.write_text("result")

        # Start a model run
        run_hash = tracker.start_model_run(
            "test_model",
            2017,
            0,
            description="Test run"
        )

        # Record inputs using normal method
        input_rec1 = tracker.record_input_file(
            "test_model",
            str(input1),
            description="Input 1",
            short_name="input1",
            model_run_id=run_hash,
        )

        input_rec2 = tracker.record_input_file(
            "test_model",
            str(input2),
            description="Input 2",
            short_name="input2",
            model_run_id=run_hash,
        )

        # OLD WAY (commented out for comparison):
        # source_files = []
        # if input_rec1:
        #     source_files.append(str(input1))
        # if input_rec2:
        #     source_files.append(str(input2))
        #
        # output_rec = tracker.record_output_file(
        #     "test_model",
        #     str(output_file),
        #     source_file_paths=source_files,
        #     ...
        # )

        # NEW WAY (using helper):
        output_rec = tracker.record_output_file_with_inputs(
            "test_model",
            str(output_file),
            input_records=[input_rec1, input_rec2],  # Just pass the records!
            year=2017,
            description="Merged output",
            short_name="output",
            model_run_id=run_hash,
        )

        # Verify output record was created
        assert output_rec is not None
        print(f"   ✅ Output record created: {output_rec.short_name}")

        # Verify source_file_paths was set correctly
        assert output_rec.source_file_paths is not None
        assert len(output_rec.source_file_paths) == 2
        print(f"   ✅ Source file paths set: {output_rec.source_file_paths}")

        # Complete the run
        tracker.complete_model_run(run_hash, output_records=[output_rec])

        print("   ✅ Test passed: Helper method works correctly")

    def test_record_output_file_with_inputs_handles_none(self, tmp_path):
        """Test that the helper correctly filters out None input records."""
        print("\n🧪 Testing None handling in record_output_file_with_inputs...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        input1 = tmp_path / "input1.csv"
        output_file = tmp_path / "output.csv"

        input1.write_text("data1")
        output_file.write_text("result")

        run_hash = tracker.start_model_run("test_model", 2017, 0)

        input_rec1 = tracker.record_input_file(
            "test_model",
            str(input1),
            short_name="input1",
            model_run_id=run_hash,
        )

        # Pass some None values in input_records
        output_rec = tracker.record_output_file_with_inputs(
            "test_model",
            str(output_file),
            input_records=[input_rec1, None, None],  # None values should be filtered
            short_name="output",
            model_run_id=run_hash,
        )

        # Should only have 1 source file (None values filtered)
        assert len(output_rec.source_file_paths) == 1
        print(f"   ✅ None values filtered correctly: {len(output_rec.source_file_paths)} source(s)")

    def test_source_file_paths_validation_warning(self, tmp_path, caplog):
        """Test that validation warnings are logged for untracked source files."""
        print("\n🧪 Testing source_file_paths validation warnings...")

        import logging
        caplog.set_level(logging.WARNING)

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        output_file = tmp_path / "output.csv"
        output_file.write_text("result")

        run_hash = tracker.start_model_run("test_model", 2017, 0)

        # Create a source file path that was NEVER tracked
        untracked_source = tmp_path / "untracked.csv"
        untracked_source.write_text("data")

        # Record output with source that wasn't tracked as input
        output_rec = tracker.record_output_file(
            "test_model",
            str(output_file),
            source_file_paths=[str(untracked_source)],  # This wasn't tracked!
            short_name="output",
            model_run_id=run_hash,
        )

        # Should have logged a warning
        assert any(
            "not found in file_records" in record.message
            for record in caplog.records
        ), "Should log warning about untracked source file"

        print("   ✅ Warning logged for untracked source file")
        print(f"   📋 Warning message: {[r.message for r in caplog.records if 'not found' in r.message][0][:100]}...")

    def test_validate_provenance_chain_basic(self, tmp_path):
        """Test the validate_provenance_chain method with a clean chain."""
        print("\n🧪 Testing validate_provenance_chain with clean chain...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        # Create a proper chain: input -> model -> output
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        input_file.write_text("data")
        output_file.write_text("result")

        run_hash = tracker.start_model_run("test_model", 2017, 0)

        input_rec = tracker.record_input_file(
            "test_model",
            str(input_file),
            short_name="input",
            model_run_id=run_hash,
        )

        output_rec = tracker.record_output_file_with_inputs(
            "test_model",
            str(output_file),
            input_records=[input_rec],
            short_name="output",
            model_run_id=run_hash,
        )

        tracker.complete_model_run(run_hash, output_records=[output_rec])

        # Validate the chain
        issues = tracker.validate_provenance_chain()

        print(f"   📋 Warnings: {len(issues['warnings'])}")
        print(f"   📋 Errors: {len(issues['errors'])}")

        # Should have no errors
        assert len(issues['errors']) == 0, f"Should have no errors, got: {issues['errors']}"
        print("   ✅ No errors in clean provenance chain")

        # May have some warnings (e.g., about inputs not being outputs)
        if issues['warnings']:
            print(f"   ℹ️  Warnings (acceptable): {issues['warnings']}")

    def test_validate_provenance_chain_detects_missing_sources(self, tmp_path):
        """Test that validation detects outputs without source_file_paths."""
        print("\n🧪 Testing validation detects missing source_file_paths...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        output_file = tmp_path / "output.csv"
        output_file.write_text("result")

        run_hash = tracker.start_model_run("test_model", 2017, 0)

        # Record output WITHOUT source_file_paths
        output_rec = tracker.record_output_file(
            "test_model",
            str(output_file),
            short_name="output",
            model_run_id=run_hash,
            # NO source_file_paths!
        )

        tracker.complete_model_run(run_hash, output_records=[output_rec])

        # Validate
        issues = tracker.validate_provenance_chain()

        # Should have a warning about missing source_file_paths
        assert any(
            "has no source_file_paths" in warning
            for warning in issues['warnings']
        ), "Should warn about output without source_file_paths"

        print(f"   ✅ Detected missing source_file_paths")
        matching_warnings = [w for w in issues['warnings'] if "source_file_paths" in w]
        print(f"   📋 Warning: {matching_warnings[0]}")

    def test_validate_provenance_chain_detects_broken_references(self, tmp_path):
        """Test that validation detects broken source_file_paths references."""
        print("\n🧪 Testing validation detects broken source references...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        output_file = tmp_path / "output.csv"
        output_file.write_text("result")

        run_hash = tracker.start_model_run("test_model", 2017, 0)

        # Record output with source that doesn't exist in file_records
        output_rec = tracker.record_output_file(
            "test_model",
            str(output_file),
            source_file_paths=["/fake/nonexistent/file.csv"],  # Not in file_records!
            short_name="output",
            model_run_id=run_hash,
        )

        tracker.complete_model_run(run_hash, output_records=[output_rec])

        # Validate
        issues = tracker.validate_provenance_chain()

        # Should have an ERROR about broken reference
        assert any(
            "is not in file_records" in error
            for error in issues['errors']
        ), "Should error on broken source_file_paths reference"

        print(f"   ✅ Detected broken source reference")
        print(f"   📋 Error: {issues['errors'][0]}")

    def test_validate_provenance_chain_detects_orphaned_files(self, tmp_path):
        """Test that validation detects orphaned file records."""
        print("\n🧪 Testing validation detects orphaned files...")

        tracker = FileProvenanceTracker(
            run_id="test_run",
            output_path=str(tmp_path),
            folder_name="test"
        )

        orphan_file = tmp_path / "orphan.csv"
        orphan_file.write_text("orphaned data")

        # Record a file but DON'T associate it with any model run
        from pilates.generic.records import FileRecord

        orphan_record = FileRecord(
            file_path=str(orphan_file),
            short_name="orphan",
            unique_id=tracker._calculate_file_hash(str(orphan_file)),
        )

        # Manually add to file_records without linking to any run
        tracker.run_info.file_records[orphan_record.unique_id] = orphan_record

        # Validate
        issues = tracker.validate_provenance_chain()

        # Should have a warning about orphaned file
        assert any(
            "is not referenced by any model run" in warning
            for warning in issues['warnings']
        ), "Should warn about orphaned file"

        print(f"   ✅ Detected orphaned file")
        matching_warnings = [w for w in issues['warnings'] if "orphan" in w.lower()]
        print(f"   📋 Warning: {matching_warnings[0]}")


if __name__ == "__main__":
    # Run tests standalone
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
