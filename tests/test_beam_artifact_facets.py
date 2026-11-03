from pilates.workflows.steps.shared import _activitysim_output_facet_meta


def test_activitysim_output_facet_meta():
    meta = _activitysim_output_facet_meta(
        "persons_asim_out",
        year=2030,
        iteration=3,
    )
    assert meta["facet"]["artifact_family"] == "persons"
    assert meta["facet"]["year"] == 2030
    assert meta["facet"]["iteration"] == 3
    assert meta["facet_schema_version"] == "v1"
    assert meta["facet_index"] is True

    archived_meta = _activitysim_output_facet_meta(
        "asim_input_households_csv_archived",
        year=2030,
        iteration=3,
    )
    assert archived_meta["facet"]["artifact_family"] == "asim_input_archived"
    assert archived_meta["facet"]["source_role"] == "households_asim_in"
    assert archived_meta["facet"]["snapshot_role"] == "asim_input_households_csv"
    assert archived_meta["facet"]["snapshot_reason"] == "exact_rewind"
    assert archived_meta["facet"]["storage_event"] == "snapshot_copy"
