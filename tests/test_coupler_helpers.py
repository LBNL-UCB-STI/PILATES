from types import SimpleNamespace

from pilates.generic.records import FileRecord, RecordStore

from pilates.utils import coupler_helpers


class CouplerWithSetFromArtifact:
    def __init__(self) -> None:
        self.calls = []

    def set_from_artifact(self, key, value) -> None:
        self.calls.append(("set_from_artifact", key, value))

    def set(self, key, value) -> None:
        self.calls.append(("set", key, value))


class CouplerWithSetOnly:
    def __init__(self) -> None:
        self.calls = []

    def set(self, key, value) -> None:
        self.calls.append(("set", key, value))


def test_set_coupler_from_artifact_prefers_set_from_artifact() -> None:
    coupler = CouplerWithSetFromArtifact()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "usim_datastore_h5", "artifact", fallback="fallback"
    )
    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", "artifact")]


def test_set_coupler_from_artifact_falls_back_to_set() -> None:
    coupler = CouplerWithSetOnly()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "usim_datastore_h5", None, fallback="/tmp/path.h5"
    )
    assert coupler.calls == [("set", "usim_datastore_h5", "/tmp/path.h5")]


def test_update_coupler_from_beam_outputs_sets_latest_records(
    monkeypatch, tmp_path
) -> None:
    log_calls = []

    def fake_log_output(path, key, description):
        log_calls.append((key, path, description))
        return f"artifact:{key}:{path}"

    monkeypatch.setattr(coupler_helpers.cr, "log_output", fake_log_output)

    linkstats_iter1 = tmp_path / "linkstats_2030_1.csv.gz"
    linkstats_iter2 = tmp_path / "linkstats_2030_2.csv.gz"
    linkstats_sub = tmp_path / "linkstats_2030_2_sub1.csv.gz"
    plans_iter1 = tmp_path / "plans_2030_1.csv.gz"
    plans_iter2 = tmp_path / "plans_2030_2.csv.gz"

    for path in (
        linkstats_iter1,
        linkstats_iter2,
        linkstats_sub,
        plans_iter1,
        plans_iter2,
    ):
        path.write_text("data")

    records = [
        FileRecord(
            file_path=str(linkstats_iter1),
            short_name="linkstats_2030_1",
        ),
        FileRecord(
            file_path=str(linkstats_iter2),
            short_name="linkstats_2030_2",
        ),
        FileRecord(
            file_path=str(linkstats_sub),
            short_name="linkstats_2030_2_sub1",
        ),
        FileRecord(
            file_path=str(plans_iter1),
            short_name="beam_plans_out_2030_1",
        ),
        FileRecord(
            file_path=str(plans_iter2),
            short_name="beam_plans_out_2030_2",
        ),
    ]
    output_store = RecordStore(recordList=records)

    coupler = CouplerWithSetFromArtifact()
    workspace = SimpleNamespace(full_path=str(tmp_path))

    coupler_helpers.update_coupler_from_beam_outputs(output_store, coupler, workspace)

    assert (
        "linkstats",
        str(linkstats_iter2),
        "BEAM linkstats output for downstream runs",
    ) in log_calls
    assert (
        "beam_plans_out",
        str(plans_iter2),
        "BEAM plans output for downstream runs",
    ) in log_calls
    assert (
        "set_from_artifact",
        "linkstats",
        f"artifact:linkstats:{linkstats_iter2}",
    ) in coupler.calls
    assert (
        "set_from_artifact",
        "beam_plans_out",
        f"artifact:beam_plans_out:{plans_iter2}",
    ) in coupler.calls
