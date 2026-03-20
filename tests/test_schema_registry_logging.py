import types

import pandas as pd
import pyarrow as pa

from pilates.utils import consist_runtime as cr


def _install_consist_stub(monkeypatch, calls):
    def _log_input(path, key=None, enabled=None, **meta):
        calls.append(("input", key, meta))
        return {"path": path, "key": key}

    def _log_output(path, key=None, enabled=None, **meta):
        calls.append(("output", key, meta))
        return {"path": path, "key": key}

    stub = types.SimpleNamespace(log_input=_log_input, log_output=_log_output)
    monkeypatch.setattr(cr, "consist", stub)


def test_log_input_attaches_schema_for_known_key(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_input("/tmp/households.csv", key="households_asim_in", enabled=True)

    assert calls
    _, key, meta = calls[0]
    assert key == "households_asim_in"
    assert meta["schema"].__name__ == "HouseholdsAsimIn"
    assert meta["declared_schema_class"] == "HouseholdsAsimIn"
    assert meta["declared_schema_table"] == "HouseholdsAsimIn"


def test_log_output_does_not_override_explicit_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    class _ExplicitSchema:
        pass

    cr.log_output(
        "/tmp/households.csv",
        key="households_asim_in",
        enabled=True,
        schema=_ExplicitSchema,
    )

    assert calls
    _, _, meta = calls[0]
    assert meta["schema"] is _ExplicitSchema


def test_log_input_unknown_key_has_no_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_input("/tmp/unknown.csv", key="not_a_known_key", enabled=True)

    assert calls
    _, _, meta = calls[0]
    assert "schema" not in meta


def test_log_input_alias_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_input("/tmp/households.csv", key="asim_households_in", enabled=True)

    assert calls
    _, key, meta = calls[0]
    assert key == "asim_households_in"
    assert meta["schema"].__name__ == "HouseholdsAsimIn"


def test_log_output_linkstats_key_attaches_beam_linkstats_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_output("/tmp/linkstats.parquet", key="linkstats", enabled=True)

    assert calls
    _, key, meta = calls[0]
    assert key == "linkstats"
    assert meta["schema"].__name__ == "BeamLinkstats"


def test_log_output_phys_sim_linkstats_key_attaches_beam_linkstats_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter9__beam_sub_iter1"
    cr.log_output("/tmp/linkstats.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamLinkstats"


def test_log_output_split_events_key_attaches_curated_split_event_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "events_parquet_2018_0_type_PathTraversal"
    cr.log_output("/tmp/events.PathTraversal.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamEventsPathTraversal"


def test_log_output_path_traversal_links_key_attaches_links_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "path_traversal_links_2018_0"
    cr.log_output("/tmp/events.PathTraversal.links.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamPathTraversalLinks"


def test_log_output_split_events_lowercase_type_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "events_parquet_2018_0_type_actstart"
    cr.log_output("/tmp/events.actstart.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamEventsActStart"


def test_log_output_atlas_householdv_year_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "householdv_2023"
    cr.log_output("/tmp/householdv_2023.csv", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "HouseholdVAtlasOut"


def test_log_output_atlas_vehicles_year_csv_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "vehicles_2023.csv"
    cr.log_output("/tmp/vehicles_2023.csv", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "VehiclesAtlasOut"


def test_log_output_vehicles_beam_in_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "vehicles_beam_in"
    cr.log_output("/tmp/vehicles.csv.gz", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "VehiclesBeamIn"


def test_log_output_households_beam_in_key_attaches_beam_input_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "households_beam_in"
    cr.log_output("/tmp/households.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "HouseholdsBeamIn"


def test_log_output_persons_beam_in_key_attaches_beam_input_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "persons_beam_in"
    cr.log_output("/tmp/persons.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "PersonsBeamIn"


def test_log_output_households_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "households_asim_out"
    cr.log_output("/tmp/households.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "HouseholdsAsimOut"


def test_log_output_persons_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "persons_asim_out"
    cr.log_output("/tmp/persons.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "PersonsAsimOut"


def test_log_output_tours_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "tours_asim_out"
    cr.log_output("/tmp/tours.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "ToursAsimOut"


def test_log_output_accessibility_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "accessibility_asim_out"
    cr.log_output("/tmp/accessibility.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "AccessibilityAsimOut"


def test_log_output_land_use_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "land_use_asim_out"
    cr.log_output("/tmp/land_use.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "LandUseAsimOut"


def test_log_output_joint_tour_participants_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "joint_tour_participants_asim_out"
    cr.log_output("/tmp/joint_tour_participants.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "JointTourParticipantsAsimOut"


def test_log_h5_table_attaches_declared_schema_metadata(monkeypatch):
    calls = []

    class _TrackerStub:
        def log_h5_table(self, path, key=None, table_path=None, direction="input", **meta):
            calls.append((path, key, table_path, direction, meta))
            return types.SimpleNamespace(meta={"table_path": table_path})

    monkeypatch.setattr(cr, "current_tracker", lambda: _TrackerStub())

    artifact = cr.log_h5_table(
        "/tmp/data.h5",
        key="tours_asim_out",
        table_path="/2030/tours",
        direction="output",
        enabled=True,
    )

    assert artifact is not None
    assert calls
    _, key, table_path, direction, meta = calls[0]
    assert key == "tours_asim_out"
    assert table_path == "/2030/tours"
    assert direction == "output"
    assert meta["schema"].__name__ == "ToursAsimOut"
    assert meta["declared_schema_class"] == "ToursAsimOut"
    assert meta["declared_schema_table"] == "ToursAsimOut"


def test_log_h5_table_activitysim_postprocess_households_key_attaches_schema(monkeypatch):
    calls = []

    class _TrackerStub:
        def log_h5_table(self, path, key=None, table_path=None, direction="input", **meta):
            calls.append((path, key, table_path, direction, meta))
            return types.SimpleNamespace(meta={"table_path": table_path})

    monkeypatch.setattr(cr, "current_tracker", lambda: _TrackerStub())

    artifact = cr.log_h5_table(
        "/tmp/data.h5",
        key="activitysim_postprocess_usim_households_table_updated",
        table_path="/households",
        direction="output",
        enabled=True,
    )

    assert artifact is not None
    assert calls
    _, key, table_path, direction, meta = calls[0]
    assert key == "activitysim_postprocess_usim_households_table_updated"
    assert table_path == "/households"
    assert direction == "output"
    assert meta["schema"].__name__ == "ActivitysimPostprocessUsimHouseholdsUpdated"


def test_log_h5_table_activitysim_postprocess_persons_key_attaches_schema(monkeypatch):
    calls = []

    class _TrackerStub:
        def log_h5_table(self, path, key=None, table_path=None, direction="input", **meta):
            calls.append((path, key, table_path, direction, meta))
            return types.SimpleNamespace(meta={"table_path": table_path})

    monkeypatch.setattr(cr, "current_tracker", lambda: _TrackerStub())

    artifact = cr.log_h5_table(
        "/tmp/data.h5",
        key="activitysim_postprocess_usim_persons_table_updated",
        table_path="/persons",
        direction="output",
        enabled=True,
    )

    assert artifact is not None
    assert calls
    _, key, table_path, direction, meta = calls[0]
    assert key == "activitysim_postprocess_usim_persons_table_updated"
    assert table_path == "/persons"
    assert direction == "output"
    assert meta["schema"].__name__ == "ActivitysimPostprocessUsimPersonsUpdated"


def test_log_output_beam_plans_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "beam_plans_asim_out"
    cr.log_output("/tmp/beam_plans.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamPlansAsimOut"


def test_log_output_beam_plans_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "beam_plans_out"
    cr.log_output("/tmp/beam_plans.csv.gz", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamPlansOut"


def test_log_output_events_parquet_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "events_parquet_2023_0"
    cr.log_output("/tmp/events.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamEventsParquet"


def test_log_output_final_vehicles_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "final_vehicles_2023_0"
    cr.log_output("/tmp/final_vehicles.csv.gz", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamFinalVehicles"


def test_log_output_route_history_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "route_history_2023_0"
    cr.log_output("/tmp/route_history.csv.gz", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamRouteHistory"


def test_log_output_trips_asim_out_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "trips_asim_out"
    cr.log_output("/tmp/trips.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "TripsAsimOut"


def test_log_output_warns_on_schema_column_mismatch_without_failing(
    monkeypatch, tmp_path, caplog
):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    csv_path = tmp_path / "householdv_2023.csv"
    csv_path.write_text("household_id,unexpected_column\n1,foo\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        cr.log_output(str(csv_path), key="householdv_2023", enabled=True)

    assert calls
    _, _, meta = calls[0]
    assert meta["schema"].__name__ == "HouseholdVAtlasOut"
    assert "[SCHEMA WARNING]" in caplog.text
    assert "missing_columns" in caplog.text


def test_arrow_type_family_treats_dictionary_encoded_strings_as_string():
    dictionary_type = pa.dictionary(pa.int32(), pa.string())

    assert cr._arrow_type_family(dictionary_type, pa_module=pa) == "string"


def test_integer_expected_accepts_float_observed_for_schema_checker():
    assert cr._type_families_compatible("integer", "float") is True


def test_observed_columns_and_families_ignore_pandas_index_parquet_column(tmp_path):
    parquet_path = tmp_path / "indexed.parquet"
    pd.DataFrame({"tour_mode": ["drive", "walk"]}).to_parquet(parquet_path, index=True)

    names, families = cr._observed_columns_and_families(str(parquet_path))

    assert "__index_level_0__" not in names
    assert "__index_level_0__" not in families
    assert "tour_mode" in names


def test_log_output_retries_without_schema_when_schema_logging_fails(
    monkeypatch, caplog
):
    calls = []

    def _log_output(path, key=None, enabled=None, **meta):
        calls.append((enabled, key, dict(meta)))
        if enabled and "schema" in meta:
            raise ValueError("schema registration failed")
        return {"path": path, "key": key}

    monkeypatch.setattr(cr, "consist", types.SimpleNamespace(log_output=_log_output))

    with caplog.at_level("WARNING"):
        result = cr.log_output("/tmp/households.parquet", key="households_asim_out")

    assert result is not None
    assert len(calls) == 2
    assert calls[0][0] is True
    assert "schema" in calls[0][2]
    assert calls[0][2]["declared_schema_class"] == "HouseholdsAsimOut"
    assert calls[1][0] is True
    assert "schema" not in calls[1][2]
    assert "declared_schema_class" not in calls[1][2]
    assert "retrying without schema metadata" in caplog.text


def test_log_output_raises_schema_logging_failure_when_warn_only_disabled(monkeypatch):
    calls = []

    def _log_output(path, key=None, enabled=None, **meta):
        calls.append((enabled, key, dict(meta)))
        if enabled:
            raise ValueError("schema registration failed")
        return {"path": path, "key": key}

    monkeypatch.setattr(cr, "consist", types.SimpleNamespace(log_output=_log_output))
    monkeypatch.setenv("PILATES_SCHEMA_WARN_ONLY", "0")

    try:
        cr.log_output("/tmp/households.parquet", key="households_asim_out")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    assert len(calls) == 1


def test_log_output_warns_when_schema_fk_target_not_registered(monkeypatch, caplog):
    from sqlalchemy import BigInteger, Column, ForeignKey
    from sqlmodel import Field, SQLModel

    calls = []
    _install_consist_stub(monkeypatch, calls)

    class _BadFkSchema(SQLModel, table=True):
        __tablename__ = "BadFkSchema"
        __table_args__ = {"extend_existing": True}
        __abstract__ = True

        bad_ref: int | None = Field(
            default=None,
            sa_column=Column(
                "bad_ref",
                BigInteger,
                ForeignKey("MissingTable.missing_id"),
                nullable=True,
            ),
        )

    monkeypatch.setattr(cr, "_schema_for_key", lambda _key: _BadFkSchema)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/does_not_matter.parquet", key="bad_fk_key", enabled=True)

    assert calls
    assert "target table is not registered in schema registry" in caplog.text


def test_log_output_trips_asim_out_fk_targets_are_registered(monkeypatch, caplog):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/trips.parquet", key="trips_asim_out", enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == "trips_asim_out"
    assert meta["schema"].__name__ == "TripsAsimOut"
    assert "target table is not registered in schema registry" not in caplog.text


def test_log_output_beam_plans_out_fk_targets_are_registered(monkeypatch, caplog):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/beam_plans.csv.gz", key="beam_plans_out", enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == "beam_plans_out"
    assert meta["schema"].__name__ == "BeamPlansOut"
    assert "target table is not registered in schema registry" not in caplog.text


def test_log_output_accessibility_asim_out_fk_targets_are_registered(monkeypatch, caplog):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/accessibility.parquet", key="accessibility_asim_out", enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == "accessibility_asim_out"
    assert meta["schema"].__name__ == "AccessibilityAsimOut"
    assert "target table is not registered in schema registry" not in caplog.text


def test_log_output_events_parquet_fk_targets_are_registered(monkeypatch, caplog):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/events.parquet", key="events_parquet_2023_0", enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == "events_parquet_2023_0"
    assert meta["schema"].__name__ == "BeamEventsParquet"
    assert "target table is not registered in schema registry" not in caplog.text


def test_log_output_final_vehicles_fk_targets_are_registered(monkeypatch, caplog):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/final_vehicles.csv.gz", key="final_vehicles_2023_0", enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == "final_vehicles_2023_0"
    assert meta["schema"].__name__ == "BeamFinalVehicles"
    assert "target table is not registered in schema registry" not in caplog.text


def test_log_output_route_history_fk_targets_are_registered(monkeypatch, caplog):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    with caplog.at_level("WARNING"):
        cr.log_output("/tmp/route_history.csv.gz", key="route_history_2023_0", enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == "route_history_2023_0"
    assert meta["schema"].__name__ == "BeamRouteHistory"
    assert "target table is not registered in schema registry" not in caplog.text


def test_log_h5_table_activitysim_postprocess_persons_fk_targets_are_registered(
    monkeypatch, caplog
):
    calls = []

    class _TrackerStub:
        def log_h5_table(self, path, key=None, table_path=None, direction="input", **meta):
            calls.append((path, key, table_path, direction, meta))
            return types.SimpleNamespace(meta={"table_path": table_path})

    monkeypatch.setattr(cr, "current_tracker", lambda: _TrackerStub())

    with caplog.at_level("WARNING"):
        cr.log_h5_table(
            "/tmp/data.h5",
            key="activitysim_postprocess_usim_persons_table_updated",
            table_path="/persons",
            direction="output",
            enabled=True,
        )

    assert calls
    _, key, table_path, direction, meta = calls[0]
    assert key == "activitysim_postprocess_usim_persons_table_updated"
    assert table_path == "/persons"
    assert direction == "output"
    assert meta["schema"].__name__ == "ActivitysimPostprocessUsimPersonsUpdated"
    assert "target table is not registered in schema registry" not in caplog.text
