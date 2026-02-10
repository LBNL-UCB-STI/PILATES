from __future__ import annotations

from pilates.utils.consist_analysis import (
    build_archive_mounts,
    find_linkstats_artifacts,
    get_duckdb_health,
    parse_linkstats_facets_from_key,
    print_duckdb_health,
    summarize_linkstats_artifacts,
    summarize_linkstats_deltas,
    summarize_linkstats_traveltime_deltas,
    summarize_linkstats_traveltime_deltas_hourly_weighted,
)


def test_build_archive_mounts_sets_workspace_to_archive_dir(tmp_path):
    archive_run_dir = tmp_path / "archive-run"
    project_root = tmp_path / "project"
    output_root = tmp_path / "output"
    archive_run_dir.mkdir()
    project_root.mkdir()
    output_root.mkdir()

    mounts = build_archive_mounts(
        archive_run_dir=archive_run_dir,
        project_root=project_root,
        output_root=output_root,
    )

    assert mounts["workspace"] == str(archive_run_dir.resolve())
    assert mounts["inputs"] == str(project_root.resolve())
    assert mounts["scratch"] == str(output_root.resolve())


def test_get_duckdb_health_reports_missing_file(tmp_path):
    missing = tmp_path / "missing.duckdb"
    info = get_duckdb_health(db_path=missing, probe_open=False)
    assert info["db_path"] == str(missing.resolve())
    assert info["db_exists"] is False
    assert info["wal_exists"] is False
    assert info["duckdb_open_seconds"] is None
    assert info["duckdb_open_error"] is None


def test_print_duckdb_health_without_open_probe(tmp_path, capsys):
    db_file = tmp_path / "test.duckdb"
    db_file.write_bytes(b"")
    info = print_duckdb_health(db_path=db_file, probe_open=False)
    output = capsys.readouterr().out
    assert "DuckDB health:" in output
    assert "DB exists: True" in output
    assert info["db_exists"] is True
    assert info["duckdb_open_seconds"] is None


def test_parse_linkstats_facets_parses_phys_sim_and_sub_iteration():
    parsed = parse_linkstats_facets_from_key(
        "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter9__beam_sub_iter1"
    )
    assert parsed == {
        "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
        "year": 2018,
        "iteration": 0,
        "phys_sim_iteration": 9,
        "beam_sub_iteration": 1,
    }


def test_parse_linkstats_facets_parses_linkstats_parquet_iteration_key():
    parsed = parse_linkstats_facets_from_key("linkstats_parquet_2018_0_sub1")
    assert parsed == {
        "artifact_family": "linkstats_parquet",
        "year": 2018,
        "iteration": 0,
        "beam_sub_iteration": 1,
    }


def test_find_linkstats_artifacts_falls_back_to_key_prefix(monkeypatch):
    class _FakeArtifact:
        id = "artifact-1"
        key = "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter1"
        container_uri = "workspace://beam/beam_output/example.parquet"
        run_id = "run-1"
        meta = {}

    class _FakeTracker:
        def find_artifacts_by_params(self, **kwargs):
            assert kwargs.get("params")
            return []

    monkeypatch.setattr(
        "pilates.utils.consist_analysis._find_artifacts_by_key_prefix_sqlmodel",
        lambda *args, **kwargs: [_FakeArtifact()],
    )

    frame = find_linkstats_artifacts(
        _FakeTracker(),
        year=2018,
        iteration=0,
        artifact_family="linkstats_unmodified_phys_sim_iter_parquet",
    )
    assert len(frame) == 1
    assert frame.iloc[0]["key"] == _FakeArtifact.key
    assert int(frame.iloc[0]["phys_sim_iteration"]) == 1
    assert frame.iloc[0]["facet_source"] == "key_parse"


def test_find_linkstats_artifacts_prefers_artifact_kv_facets(tmp_path):
    class _FakeKV:
        def __init__(
            self,
            artifact_id,
            key_path,
            value_type,
            value_num=None,
            value_str=None,
            value_bool=None,
        ):
            self.artifact_id = artifact_id
            self.key_path = key_path
            self.value_type = value_type
            self.value_num = value_num
            self.value_str = value_str
            self.value_bool = value_bool

    class _FakeSession:
        def __init__(self, rows):
            self._rows = rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def exec(self, _statement):
            class _Result:
                def __init__(self, rows):
                    self._rows = rows

                def all(self):
                    return self._rows

            return _Result(self._rows)

    class _FakeDB:
        def __init__(self, rows):
            self._rows = rows

        def session_scope(self):
            return _FakeSession(self._rows)

    class _FakeArtifact:
        id = "artifact-1"
        key = "not_parseable_key_name"
        container_uri = "workspace://beam/beam_output/example.parquet"
        run_id = "run-1"
        meta = {}

    class _FakeTracker:
        def __init__(self, rows):
            self.db = _FakeDB(rows)

        def find_artifacts_by_params(self, **kwargs):
            assert kwargs.get("params")
            return [_FakeArtifact()]

    rows = [
        _FakeKV(
            "artifact-1",
            "artifact_family",
            "str",
            value_str="linkstats_unmodified_phys_sim_iter_parquet",
        ),
        _FakeKV("artifact-1", "year", "int", value_num=2018),
        _FakeKV("artifact-1", "iteration", "int", value_num=0),
        _FakeKV("artifact-1", "phys_sim_iteration", "int", value_num=3),
    ]
    tracker = _FakeTracker(rows)
    frame = find_linkstats_artifacts(
        tracker,
        year=2018,
        iteration=0,
        artifact_family="linkstats_unmodified_phys_sim_iter_parquet",
    )

    assert len(frame) == 1
    assert frame.iloc[0]["key"] == _FakeArtifact.key
    assert int(frame.iloc[0]["phys_sim_iteration"]) == 3
    assert frame.iloc[0]["facet_source"] == "artifact_kv"


def test_find_linkstats_artifacts_batches_kv_queries_for_multiple_artifacts():
    class _FakeKV:
        def __init__(
            self,
            artifact_id,
            key_path,
            value_type,
            value_num=None,
            value_str=None,
            value_bool=None,
        ):
            self.artifact_id = artifact_id
            self.key_path = key_path
            self.value_type = value_type
            self.value_num = value_num
            self.value_str = value_str
            self.value_bool = value_bool

    class _FakeSession:
        def __init__(self, rows, call_counter):
            self._rows = rows
            self._call_counter = call_counter

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def exec(self, _statement):
            self._call_counter["count"] += 1

            class _Result:
                def __init__(self, rows):
                    self._rows = rows

                def all(self):
                    return self._rows

            return _Result(self._rows)

    class _FakeDB:
        def __init__(self, rows):
            self._rows = rows
            self.call_counter = {"count": 0}

        def session_scope(self):
            return _FakeSession(self._rows, self.call_counter)

    class _FakeArtifact:
        def __init__(self, artifact_id, phys_sim_iteration):
            self.id = artifact_id
            self.key = f"not_parseable_key_{artifact_id}"
            self.container_uri = f"workspace://beam/beam_output/{artifact_id}.parquet"
            self.run_id = "run-1"
            self.meta = {}
            self.phys_sim_iteration = phys_sim_iteration

    class _FakeTracker:
        def __init__(self, rows):
            self.db = _FakeDB(rows)
            self._artifacts = [
                _FakeArtifact("artifact-1", 1),
                _FakeArtifact("artifact-2", 2),
            ]

        def find_artifacts_by_params(self, **kwargs):
            assert kwargs.get("params")
            return self._artifacts

    rows = [
        _FakeKV(
            "artifact-1",
            "artifact_family",
            "str",
            value_str="linkstats_unmodified_phys_sim_iter_parquet",
        ),
        _FakeKV("artifact-1", "year", "int", value_num=2018),
        _FakeKV("artifact-1", "iteration", "int", value_num=0),
        _FakeKV("artifact-1", "phys_sim_iteration", "int", value_num=1),
        _FakeKV(
            "artifact-2",
            "artifact_family",
            "str",
            value_str="linkstats_unmodified_phys_sim_iter_parquet",
        ),
        _FakeKV("artifact-2", "year", "int", value_num=2018),
        _FakeKV("artifact-2", "iteration", "int", value_num=0),
        _FakeKV("artifact-2", "phys_sim_iteration", "int", value_num=2),
    ]
    tracker = _FakeTracker(rows)

    frame = find_linkstats_artifacts(
        tracker,
        year=2018,
        iteration=0,
        artifact_family="linkstats_unmodified_phys_sim_iter_parquet",
    )

    assert len(frame) == 2
    assert tracker.db.call_counter["count"] == 1
    assert sorted(frame["phys_sim_iteration"].astype(int).tolist()) == [1, 2]


def test_summarize_linkstats_artifacts_uses_grouped_view_helpers(monkeypatch):
    class _FakeTracker:
        pass

    tracker = _FakeTracker()
    calls = []

    def _fake_create_grouped_view(*, tracker, artifacts_df, **kwargs):
        calls.append(("create_grouped", len(artifacts_df)))
        return "v_linkstats_grouped_test"

    def _fake_summarize_grouped(*, tracker, view_name, **kwargs):
        calls.append(("summarize_grouped", view_name, kwargs.get("traveltime_weighting")))
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "artifact_id": "a1",
                    "row_count": 10,
                    "distinct_links": 5,
                    "volume_sum": 100.0,
                    "traveltime_mean": 8.0,
                    "traveltime_p95": 12.0,
                }
            ]
        )

    monkeypatch.setattr(
        "pilates.utils.consist_analysis._create_linkstats_grouped_view",
        _fake_create_grouped_view,
    )
    monkeypatch.setattr(
        "pilates.utils.consist_analysis._summarize_linkstats_grouped_view",
        _fake_summarize_grouped,
    )

    import pandas as pd

    artifacts_df = pd.DataFrame(
        [
            {
                "artifact_id": "a1",
                "key": "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter1",
                "year": 2018,
                "iteration": 0,
                "phys_sim_iteration": 1,
                "beam_sub_iteration": None,
            }
        ]
    )

    summary_df = summarize_linkstats_artifacts(artifacts_df, tracker=tracker)
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["view_name"] == "v_linkstats_grouped_test"
    assert summary_df.iloc[0]["row_count"] == 10
    assert calls[0][0] == "create_grouped"
    assert calls[1] == ("summarize_grouped", "v_linkstats_grouped_test", "unweighted")


def test_summarize_linkstats_artifacts_passes_volume_weighted_option(monkeypatch):
    class _FakeTracker:
        pass

    tracker = _FakeTracker()
    calls = []

    def _fake_create_grouped_view(*, tracker, artifacts_df, **kwargs):
        return "v_linkstats_grouped_test"

    def _fake_summarize_grouped(*, tracker, view_name, **kwargs):
        calls.append(kwargs.get("traveltime_weighting"))
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "artifact_id": "a1",
                    "row_count": 1,
                    "distinct_links": 1,
                    "volume_sum": 1.0,
                    "traveltime_mean": 1.0,
                    "traveltime_p95": 1.0,
                }
            ]
        )

    monkeypatch.setattr(
        "pilates.utils.consist_analysis._create_linkstats_grouped_view",
        _fake_create_grouped_view,
    )
    monkeypatch.setattr(
        "pilates.utils.consist_analysis._summarize_linkstats_grouped_view",
        _fake_summarize_grouped,
    )

    import pandas as pd

    artifacts_df = pd.DataFrame(
        [{"artifact_id": "a1", "key": "k1", "year": 2018, "iteration": 0, "phys_sim_iteration": 1}]
    )
    _ = summarize_linkstats_artifacts(
        artifacts_df,
        tracker=tracker,
        traveltime_weighting="volume_weighted",
    )
    assert calls == ["volume_weighted"]


def test_summarize_linkstats_traveltime_deltas_uses_metric_delta_helper(monkeypatch):
    calls = []

    def _fake_delta(*, tracker, view_name, summary_df, metric_column):
        calls.append((view_name, len(summary_df), metric_column))
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "year": 2018,
                    "iteration": 0,
                    "beam_sub_iteration": 0,
                    "phys_sim_iteration_prev": 1,
                    "phys_sim_iteration_curr": 2,
                    "artifact_id_prev": "a1",
                    "artifact_id_curr": "a2",
                    "key_prev": "k1",
                    "key_curr": "k2",
                    "group_count": 2,
                    "traveltime_delta_mean": 0.5,
                    "traveltime_delta_abs_mean": 0.5,
                }
            ]
        )

    monkeypatch.setattr(
        "pilates.utils.consist_analysis._summarize_linkstats_metric_deltas_from_summary",
        _fake_delta,
    )

    import pandas as pd

    summary_df = pd.DataFrame(
        [
            {
                "year": 2018,
                "iteration": 0,
                "beam_sub_iteration": 0,
                "phys_sim_iteration": 1,
                "artifact_id": "a1",
                "key": "k1",
                "view_name": "v1",
            },
            {
                "year": 2018,
                "iteration": 0,
                "beam_sub_iteration": 0,
                "phys_sim_iteration": 2,
                "artifact_id": "a2",
                "key": "k2",
                "view_name": "v1",
            },
        ]
    )

    delta_df = summarize_linkstats_traveltime_deltas(summary_df, tracker=object())
    assert len(delta_df) == 1
    assert delta_df.iloc[0]["key_prev"] == "k1"
    assert delta_df.iloc[0]["key_curr"] == "k2"
    assert delta_df.iloc[0]["traveltime_delta_mean"] == 0.5
    assert calls == [("v1", 2, "traveltime")]


def test_summarize_linkstats_deltas_merges_metric_queries(monkeypatch):
    import pandas as pd

    travel_df = pd.DataFrame(
        [
            {
                "year": 2018,
                "iteration": 0,
                "beam_sub_iteration": 0,
                "phys_sim_iteration_prev": 1,
                "phys_sim_iteration_curr": 2,
                "artifact_id_prev": "a1",
                "artifact_id_curr": "a2",
                "key_prev": "k1",
                "key_curr": "k2",
                "group_count": 2,
                "traveltime_delta_mean": 0.5,
                "traveltime_delta_abs_mean": 0.5,
                "view_prev": "v1",
                "view_curr": "v1",
            }
        ]
    )
    volume_df = pd.DataFrame(
        [
            {
                "year": 2018,
                "iteration": 0,
                "beam_sub_iteration": 0,
                "phys_sim_iteration_prev": 1,
                "phys_sim_iteration_curr": 2,
                "artifact_id_prev": "a1",
                "artifact_id_curr": "a2",
                "key_prev": "k1",
                "key_curr": "k2",
                "group_count": 2,
                "volume_delta_mean": 1.5,
                "volume_delta_abs_mean": 1.5,
                "view_prev": "v1",
                "view_curr": "v1",
            }
        ]
    )

    monkeypatch.setattr(
        "pilates.utils.consist_analysis.summarize_linkstats_traveltime_deltas",
        lambda *args, **kwargs: travel_df,
    )
    monkeypatch.setattr(
        "pilates.utils.consist_analysis.summarize_linkstats_volume_deltas",
        lambda *args, **kwargs: volume_df,
    )

    merged = summarize_linkstats_deltas(pd.DataFrame([{"x": 1}]), tracker=object())
    assert len(merged) == 1
    assert merged.iloc[0]["traveltime_delta_mean"] == 0.5
    assert merged.iloc[0]["volume_delta_mean"] == 1.5


def test_summarize_linkstats_traveltime_deltas_hourly_weighted_uses_helper(monkeypatch):
    calls = []

    def _fake_hourly(*, tracker, view_name, summary_df, exclude_zero_volume):
        calls.append((view_name, len(summary_df), exclude_zero_volume))
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "year": 2018,
                    "iteration": 0,
                    "beam_sub_iteration": 0,
                    "phys_sim_iteration_prev": 1,
                    "phys_sim_iteration_curr": 2,
                    "artifact_id_prev": "a1",
                    "artifact_id_curr": "a2",
                    "key_prev": "k1",
                    "key_curr": "k2",
                    "hour_count": 24,
                    "prev_volume_total": 100.0,
                    "curr_volume_total": 120.0,
                    "traveltime_delta_mean": 0.2,
                    "traveltime_delta_abs_mean": 0.3,
                }
            ]
        )

    monkeypatch.setattr(
        "pilates.utils.consist_analysis._summarize_linkstats_traveltime_deltas_hourly_weighted_from_summary",
        _fake_hourly,
    )

    import pandas as pd

    summary_df = pd.DataFrame(
        [
            {
                "year": 2018,
                "iteration": 0,
                "beam_sub_iteration": 0,
                "phys_sim_iteration": 1,
                "artifact_id": "a1",
                "key": "k1",
                "view_name": "v1",
            },
            {
                "year": 2018,
                "iteration": 0,
                "beam_sub_iteration": 0,
                "phys_sim_iteration": 2,
                "artifact_id": "a2",
                "key": "k2",
                "view_name": "v1",
            },
        ]
    )

    delta_df = summarize_linkstats_traveltime_deltas_hourly_weighted(
        summary_df, tracker=object()
    )
    assert len(delta_df) == 1
    assert delta_df.iloc[0]["hour_count"] == 24
    assert delta_df.iloc[0]["traveltime_delta_mean"] == 0.2
    assert calls == [("v1", 2, True)]
