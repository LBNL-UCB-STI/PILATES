import pytest
from types import SimpleNamespace

from pilates.generic.initialization import Initialization
from pilates.generic.records import RecordStore, Record, FileRecord, RepoRecord

# ----------------------------------------------------------------------
# Dummy objects to replace real model components
# ----------------------------------------------------------------------


class DummyPreprocessor:
    """A simple preprocessor that returns deterministic RecordStore objects."""

    def __init__(self, *_, **__):
        pass

    def copy_data_to_mutable_location(self, settings, output_dir):
        # Return two RecordStore objects with known records
        in_record = Record(unique_id="in1", short_name="input")
        out_record = Record(unique_id="out1", short_name="output")
        return RecordStore(recordList=[in_record]), RecordStore(recordList=[out_record])

    def preprocess(self, workspace, previous_records=None):
        return RecordStore()


class DummyModelFactory:
    """ModelFactory stub that returns DummyPreprocessor for any model."""

    def get_preprocessor(self, model_name, state, provenance_tracker):
        return DummyPreprocessor()


# ----------------------------------------------------------------------
# Minimal workspace stub required by Initialization.run
# ----------------------------------------------------------------------


class DummyWorkspace:
    def __init__(self):
        self.input_data = {}
        self.output_data = {}
        self.beam_mutable_dir = "/tmp/beam"
        self.usim_mutable_dir = "/tmp/usim"
        self.atlas_mutable_input_dir = "/tmp/atlas_in"
        self.atlas_output_dir = "/tmp/atlas_out"
        self.activitysim_mutable_dir = "/tmp/asim"

    def get_beam_mutable_data_dir(self):
        return self.beam_mutable_dir

    def get_usim_mutable_data_dir(self):
        return self.usim_mutable_dir

    def get_atlas_mutable_input_dir(self):
        return self.atlas_mutable_input_dir

    def get_atlas_output_dir(self):
        return self.atlas_output_dir

    def get_asim_mutable_data_dir(self):
        return self.activitysim_mutable_dir


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_initialization_runs_beam_and_urbansim(monkeypatch):
    """
    Verify that Initialization.run correctly aggregates provenance records
    for beam and urbansim models using the dummy preprocessor.
    """
    # Patch ModelFactory used inside Initialization
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", DummyModelFactory
    )

    init = Initialization("init", None, None)
    workspace = DummyWorkspace()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="beam",
                activity_demand="dummy_activity",
                vehicle_ownership="dummy_vehicle",
                land_use="urbansim",
            )
        )
    )

    # Run initialization – should not raise any exception
    init.run(settings, workspace)

    # After run, provenance records should be stored in workspace dicts
    # Two records per model (input + output)
    assert "beam" in workspace.input_data
    assert "beam" in workspace.output_data
    assert "urbansim" in workspace.input_data
    assert "urbansim" in workspace.output_data

    # Verify that the stored records are of type Record
    for model_key in ("beam", "urbansim"):
        for rec in workspace.input_data[model_key].all_records():
            assert isinstance(rec, Record)
        for rec in workspace.output_data[model_key].all_records():
            assert isinstance(rec, Record)


def test_initialization_handles_missing_models_gracefully(monkeypatch):
    """
    If a model key is missing from settings, Initialization should simply skip it
    without raising errors.
    """
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", DummyModelFactory
    )

    init = Initialization("init", None, None)
    workspace = DummyWorkspace()
    # Settings only contains a model that is not in the loop (e.g., no beam)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="none",
                land_use=None,
                activity_demand=None,
                vehicle_ownership=None,
            )
        )
    )

    # Should complete without exception
    init.run(settings, workspace)

    # No data should have been added to workspace dictionaries
    assert workspace.input_data == {}
    assert workspace.output_data == {}
