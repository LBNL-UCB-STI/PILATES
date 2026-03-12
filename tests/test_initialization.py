import os

from types import SimpleNamespace

from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.generic.records import RecordStore, FileRecord
import json

# ----------------------------------------------------------------------
# Dummy objects to replace real model components
# ----------------------------------------------------------------------


class DummyPreprocessor:
    """A simple preprocessor that returns deterministic RecordStore objects."""

    def __init__(self, *_, **__):
        pass

    def copy_data_to_mutable_location(self, settings, output_dir):
        # Return two RecordStore objects with known records
        in_record = FileRecord(
            unique_id="in1",
            short_name="input",
            file_path="/tmp/input",
        )
        out_record = FileRecord(
            unique_id="out1",
            short_name="output",
            file_path="/tmp/output",
        )
        return RecordStore(recordList=[in_record]), RecordStore(recordList=[out_record])

    def preprocess(self, workspace, previous_records=None):
        return RecordStore()


class DummyModelFactory:
    """ModelFactory stub that returns DummyPreprocessor for any model."""

    def get_preprocessor(self, model_name, state, **_kwargs):
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
        self.full_path = "/tmp"

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
    Verify that Initialization.run aggregates input/output records for
    beam and urbansim models using the dummy preprocessor.
    """
    # Disable consist for this test to avoid needing a run context
    from pilates.utils import consist_runtime as cr
    monkeypatch.setattr(cr, "consist", None)

    # Patch ModelFactory used inside Initialization
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", DummyModelFactory
    )

    init = Initialization("init", None)
    workspace = DummyWorkspace()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="beam",
                activity_demand="dummy_activity",
                vehicle_ownership="dummy_vehicle",
                land_use="urbansim",
            ),
            start_year=2020,
        ),
        shared=SimpleNamespace(
            database=SimpleNamespace(use_consist=False),
            skims=SimpleNamespace(zone_type="block_group"),
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file="/tmp/canonical_zones.geojson",
                    activitysim_index_col="TAZ",
                    zone_type="block_group",
                    canonical_id_col="zone_key",
                )
            ),
        ),
    )

    # Create a dummy canonical_zones.geojson in the mocked mutable directory
    canonical_zones_path = os.path.join(
        workspace.get_asim_mutable_data_dir(), "canonical_zones.geojson"
    )
    os.makedirs(os.path.dirname(canonical_zones_path), exist_ok=True)
    dummy_geojson_content = {"type": "FeatureCollection", "features": []}
    with open(canonical_zones_path, "w") as f:
        json.dump(dummy_geojson_content, f)

    # Update settings to point to this dummy file
    settings.shared.geography.zones.source_file = canonical_zones_path

    # Run initialization – should not raise any exception
    init.run(settings, workspace)

    # Initialization returns the copied records and does not maintain a
    # duplicate runtime cache on the workspace.
    assert workspace.input_data == {}
    assert workspace.output_data == {}


def test_initialization_runs_beam_when_only_traffic_assignment_is_set(monkeypatch):
    from pilates.utils import consist_runtime as cr

    monkeypatch.setattr(cr, "consist", None)
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", DummyModelFactory
    )

    init = Initialization("init", None)
    workspace = DummyWorkspace()
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel=None,
                traffic_assignment="beam",
                activity_demand=None,
                vehicle_ownership=None,
                land_use=None,
            ),
            start_year=2020,
        ),
        shared=SimpleNamespace(
            database=SimpleNamespace(use_consist=False),
            geography=SimpleNamespace(zones=None),
        ),
    )

    records = init.run(settings, workspace)

    assert len(records.all_records()) == 2


def test_initialization_handles_missing_models_gracefully(monkeypatch):
    """
    If a model key is missing from settings, Initialization should simply skip it
    without raising errors.
    """
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", DummyModelFactory
    )

    init = Initialization("init", None)
    workspace = DummyWorkspace()
    # Settings only contains a model that is not in the loop (e.g., no beam)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="none",
                land_use=None,
                activity_demand=None,
                vehicle_ownership=None,
            ),
            start_year=2020,
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(zone_type="block_group"),
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file="/tmp/canonical_zones.geojson",
                    activitysim_index_col="TAZ",
                    zone_type="block_group",
                    canonical_id_col="zone_key",
                )
            ),
        ),
    )

    # Create a dummy canonical_zones.geojson in the mocked mutable directory
    canonical_zones_path = os.path.join(
        workspace.get_asim_mutable_data_dir(), "canonical_zones.geojson"
    )
    os.makedirs(os.path.dirname(canonical_zones_path), exist_ok=True)
    dummy_geojson_content = {"type": "FeatureCollection", "features": []}
    with open(canonical_zones_path, "w") as f:
        json.dump(dummy_geojson_content, f)

    # Update settings to point to this dummy file
    settings.shared.geography.zones.source_file = canonical_zones_path

    # Should complete without exception
    init.run(settings, workspace)

    # No data should have been added to workspace dictionaries
    assert workspace.input_data == {}
    assert workspace.output_data == {}


def test_initialization_logs_copy_records(monkeypatch, tmp_path):
    """
    Initialization should log each copied input/output record via Consist
    using the RecordStore keys (sanitized when needed).
    """

    class DummyLoggingPreprocessor:
        def __init__(self, input_path, output_path):
            self.input_path = input_path
            self.output_path = output_path

        def copy_data_to_mutable_location(self, settings, output_dir):
            in_record = FileRecord(
                unique_id="in1",
                short_name="bad key",
                file_path=str(self.input_path),
            )
            out_record = FileRecord(
                unique_id="out1",
                short_name="output_ok",
                file_path=str(self.output_path),
            )
            return RecordStore(recordList=[in_record]), RecordStore(
                recordList=[out_record]
            )

    class DummyLoggingModelFactory:
        def __init__(self, input_path, output_path):
            self.input_path = input_path
            self.output_path = output_path

        def get_preprocessor(self, model_name, state, **_kwargs):
            return DummyLoggingPreprocessor(self.input_path, self.output_path)

    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "dest.txt"
    input_path.write_text("source")
    output_path.write_text("dest")

    factory = DummyLoggingModelFactory(input_path, output_path)
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", lambda: factory
    )

    logged_inputs = []
    logged_outputs = []

    def _log_input(path, key=None, **_kwargs):
        logged_inputs.append((path, key))

    def _log_output(path, key=None, **_kwargs):
        logged_outputs.append((path, key))

    monkeypatch.setattr("pilates.generic.initialization.cr.log_input", _log_input)
    monkeypatch.setattr("pilates.generic.initialization.cr.log_output", _log_output)

    init = Initialization("init", None)
    workspace = DummyWorkspace()
    workspace.full_path = str(tmp_path)
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="beam",
                activity_demand=None,
                vehicle_ownership=None,
                land_use="urbansim",
            ),
            start_year=2020,
        )
    )
    zones_path = tmp_path / "canonical_zones.geojson"
    zones_path.write_text("{}")
    settings.shared = SimpleNamespace(
        geography=SimpleNamespace(
            zones=SimpleNamespace(
                source_file=str(zones_path),
                activitysim_index_col="TAZ",
                zone_type="block_group",
                canonical_id_col="zone_key",
            )
        )
    )

    init.run(settings, workspace)

    assert (str(input_path), "bad_key") in logged_inputs
    assert (str(output_path), "output_ok") in logged_outputs


def test_build_bootstrap_artifact_summary_counts_records_by_model():
    workspace = DummyWorkspace()
    copied_records = RecordStore(
        recordList=[
            FileRecord(
                unique_id="in1",
                short_name="beam_in",
                file_path="/tmp/in",
                metadata={"model": "beam", "bootstrap_direction": "input"},
            ),
            FileRecord(
                unique_id="out1",
                short_name="beam_out1",
                file_path="/tmp/out1",
                metadata={"model": "beam", "bootstrap_direction": "output"},
            ),
            FileRecord(
                unique_id="out2",
                short_name="beam_out2",
                file_path="/tmp/out2",
                metadata={"model": "beam", "bootstrap_direction": "output"},
            ),
        ]
    )

    summary = build_bootstrap_artifact_summary(workspace, copied_records)

    assert summary["models"] == ["beam"]
    assert summary["input_records_by_model"] == {"beam": 1}
    assert summary["output_records_by_model"] == {"beam": 2}
    assert summary["input_records_total"] == 1
    assert summary["output_records_total"] == 2
    assert summary["copied_records_total"] == 3


def test_initialization_summary_counts_canonical_zones_under_activitysim(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", DummyModelFactory
    )

    workspace = DummyWorkspace()
    workspace.activitysim_mutable_dir = str(tmp_path / "asim")
    workspace.beam_mutable_dir = str(tmp_path / "beam")
    workspace.usim_mutable_dir = str(tmp_path / "usim")
    workspace.atlas_mutable_input_dir = str(tmp_path / "atlas_in")
    workspace.atlas_output_dir = str(tmp_path / "atlas_out")

    zones_path = tmp_path / "canonical_zones.geojson"
    zones_path.write_text("{}")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="beam",
                activity_demand="activitysim",
                vehicle_ownership=None,
                land_use="urbansim",
            ),
            start_year=2020,
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(zone_type="block_group"),
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file=str(zones_path),
                    activitysim_index_col="TAZ",
                    zone_type="block_group",
                    canonical_id_col="zone_key",
                )
            ),
        ),
    )

    init = Initialization("init", None)
    copied_records = init.run(settings, workspace)
    summary = build_bootstrap_artifact_summary(workspace, copied_records)

    assert summary["input_records_by_model"]["activitysim"] >= 1
    assert summary["output_records_by_model"]["activitysim"] >= 1
    assert "activitysim" in summary["models"]


def test_initialization_copies_urbansim_bootstrap_inputs_once_when_urbansim_and_activitysim_enabled(
    monkeypatch, tmp_path
):
    class _CountingPreprocessor:
        def __init__(self, model_name, calls):
            self.model_name = model_name
            self.calls = calls

        def copy_data_to_mutable_location(self, settings, output_dir):
            self.calls.append((self.model_name, output_dir))
            record = FileRecord(
                unique_id=f"{self.model_name}-out",
                short_name=f"{self.model_name}_output",
                file_path=str(tmp_path / f"{self.model_name}.txt"),
            )
            return RecordStore(), RecordStore(recordList=[record])

    class _CountingFactory:
        def __init__(self):
            self.calls = []

        def get_preprocessor(self, model_name, state, **_kwargs):
            return _CountingPreprocessor(model_name, self.calls)

    factory = _CountingFactory()
    monkeypatch.setattr(
        "pilates.generic.initialization.ModelFactory", lambda: factory
    )

    workspace = DummyWorkspace()
    workspace.activitysim_mutable_dir = str(tmp_path / "asim")
    workspace.usim_mutable_dir = str(tmp_path / "usim")

    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(
                travel="none",
                activity_demand="activitysim",
                vehicle_ownership=None,
                land_use="urbansim",
            ),
            start_year=2020,
        ),
        shared=SimpleNamespace(
            geography=SimpleNamespace(zones=None),
        ),
    )

    init = Initialization("init", None)
    init.run(settings, workspace)

    urbansim_calls = [call for call in factory.calls if call[0] == "urbansim"]
    activitysim_calls = [call for call in factory.calls if call[0] == "activitysim"]

    assert len(urbansim_calls) == 1
    assert len(activitysim_calls) == 1
