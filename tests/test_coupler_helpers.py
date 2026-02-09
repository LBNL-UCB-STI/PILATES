from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict
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


def test_set_coupler_from_artifact_resolves_alias_key() -> None:
    coupler = CouplerWithSetOnly()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "asim_households_in", None, fallback="/tmp/households.csv"
    )
    assert coupler.calls == [("set", "households_asim_in", "/tmp/households.csv")]


@dataclass
class _AliasOutput:
    record_keys: ClassVar[Dict[str, str]] = {"households": "households_asim_in"}
    households: Path


def test_record_store_to_outputs_resolves_alias_mapping(tmp_path) -> None:
    households_path = tmp_path / "households.csv"
    households_path.write_text("x")
    workspace = SimpleNamespace(full_path=str(tmp_path))
    store = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(households_path),
                short_name="asim_households_in",
            )
        ]
    )

    outputs = coupler_helpers.record_store_to_outputs(store, _AliasOutput, workspace)
    assert outputs.households == households_path


def test_update_coupler_from_beam_outputs_sets_latest_records(
    monkeypatch, tmp_path
) -> None:
    log_calls = []

    def fake_log_output(path, key, description, **_meta):
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


def test_update_coupler_from_beam_outputs_profiles_linkstats_family(
    monkeypatch, tmp_path
) -> None:
    log_calls = []

    def fake_log_output(path, key, description, **meta):
        log_calls.append((key, path, description, meta))
        return f"artifact:{key}:{path}"

    monkeypatch.setattr(coupler_helpers.cr, "log_output", fake_log_output)
    monkeypatch.setattr(coupler_helpers.cr, "current_run", lambda: object())

    linkstats_csv = tmp_path / "linkstats_2018_0.csv.gz"
    linkstats_parquet = tmp_path / "linkstats_parquet_2018_0.parquet"
    linkstats_parquet_sub = tmp_path / "linkstats_parquet_2018_0_sub1.parquet"
    phys_sim = tmp_path / "0.linkstats_unmodified_physSimIter3.parquet"

    for path in (linkstats_csv, linkstats_parquet, linkstats_parquet_sub, phys_sim):
        path.write_text("data")

    output_store = RecordStore(
        recordList=[
            FileRecord(file_path=str(linkstats_csv), short_name="linkstats_2018_0"),
            FileRecord(
                file_path=str(linkstats_parquet), short_name="linkstats_parquet_2018_0"
            ),
            FileRecord(
                file_path=str(linkstats_parquet_sub),
                short_name="linkstats_parquet_2018_0_sub1",
            ),
            FileRecord(
                file_path=str(phys_sim),
                short_name=(
                    "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3"
                    "__beam_sub_iter1"
                ),
            ),
        ]
    )

    coupler = CouplerWithSetFromArtifact()
    workspace = SimpleNamespace(full_path=str(tmp_path))

    coupler_helpers.update_coupler_from_beam_outputs(output_store, coupler, workspace)

    by_key = {key: meta for key, _path, _desc, meta in log_calls}
    assert by_key["linkstats"]["profile_file_schema"] is True
    assert by_key["linkstats_warmstart"]["profile_file_schema"] is True
    assert by_key["linkstats"]["facet_schema_version"] == "v1"
    assert by_key["linkstats"]["facet"]["artifact_family"] == "linkstats"
    assert by_key["linkstats"]["facet"]["year"] == 2018
    assert by_key["linkstats"]["facet"]["iteration"] == 0
    assert by_key["linkstats_parquet_2018_0"]["profile_file_schema"] is True
    assert by_key["linkstats_parquet_2018_0_sub1"]["profile_file_schema"] is True
    assert by_key["linkstats_parquet_2018_0"]["facet_schema_version"] == "v1"
    assert (
        by_key["linkstats_parquet_2018_0"]["facet"]["artifact_family"]
        == "linkstats_parquet"
    )
    assert (
        by_key[
            "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3__beam_sub_iter1"
        ][
            "profile_file_schema"
        ]
        is True
    )
    assert (
        by_key[
            "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3__beam_sub_iter1"
        ]["facet_schema_version"]
        == "v1"
    )
    assert not any(
        c[1] == "linkstats_parquet_2018_0_sub1"
        for c in coupler.calls
        if c[0] == "set_from_artifact"
    )
    assert not any(
        c[1]
        == "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter3__beam_sub_iter1"
        for c in coupler.calls
        if c[0] == "set_from_artifact"
    )
