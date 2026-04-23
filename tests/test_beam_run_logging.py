from types import SimpleNamespace

import pytest

pytest.importorskip("openmatrix")

from pilates.beam.outputs import BeamPreprocessOutputs, BeamRunOutputs
from pilates.workflows.steps import beam as steps_beam


def test_beam_preprocess_fails_early_when_config_file_missing(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps_beam.make_beam_preprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]

    try:
        input_logger(
            settings=SimpleNamespace(
                run=SimpleNamespace(region="seattle"),
                beam=SimpleNamespace(config="seattle-pilates.conf"),
            ),
            state=SimpleNamespace(),
            workspace=SimpleNamespace(
                get_beam_mutable_data_dir=lambda: str(tmp_path / "beam" / "input"),
            ),
            holder=SimpleNamespace(),
        )
    except FileNotFoundError as exc:
        assert "BEAM primary config file is missing" in str(exc)
        assert "seattle-pilates.conf" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing BEAM config")


def test_beam_run_logs_config_file_input(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps_beam.make_beam_run_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    calls = []

    monkeypatch.setattr(
        steps_beam,
        "log_input_only",
        lambda *, key, path, description, **meta: calls.append((key, path, description)),
    )
    monkeypatch.setattr(steps_beam.cr, "current_tracker", lambda: None)

    beam_root = tmp_path / "beam" / "input" / "seattle"
    beam_root.mkdir(parents=True)
    config_path = beam_root / "seattle-pilates.conf"
    config_path.write_text("beam.test = 1\n", encoding="utf-8")

    upstream = SimpleNamespace(_iter_record_items=lambda: [])
    input_logger(
        settings=SimpleNamespace(
            run=SimpleNamespace(region="seattle"),
            beam=SimpleNamespace(config="seattle-pilates.conf"),
        ),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam" / "input"),
            get_beam_output_dir=lambda: str(tmp_path / "beam" / "output"),
        ),
        holder=SimpleNamespace(beam_preprocess=upstream),
    )

    assert calls == [
        (
            "beam_config_file",
            str(config_path),
            "BEAM config file consumed by the BEAM run",
        )
    ]


def test_beam_run_prefers_coupler_published_plan_outputs_over_disk_scan(
    monkeypatch, tmp_path
):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "build_standard_step",
        _fake_build_standard_step,
    )

    output_plans = tmp_path / "published-output-plans.xml.gz"
    output_plans.write_text("plans", encoding="utf-8")
    output_experienced = tmp_path / "published-output-experienced.xml.gz"
    output_experienced.write_text("experienced", encoding="utf-8")

    class CouplerStub:
        def get(self, key, default=None):
            mapping = {
                "beam_output_plans_xml": str(output_plans),
                "beam_output_experienced_plans_xml": str(output_experienced),
            }
            return mapping.get(key, default)

    steps_beam.make_beam_run_step(
        coupler=CouplerStub(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    calls = []

    monkeypatch.setattr(
        steps_beam,
        "log_input_only",
        lambda *, key, path, description, **meta: calls.append((key, path, description)),
    )
    monkeypatch.setattr(steps_beam.cr, "current_tracker", lambda: None)
    monkeypatch.setattr(
        steps_beam,
        "find_last_run_output_plans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disk scan should not run when coupler outputs are available")
        ),
    )

    beam_root = tmp_path / "beam" / "input" / "seattle"
    beam_root.mkdir(parents=True)
    config_path = beam_root / "seattle-pilates.conf"
    config_path.write_text("beam.test = 1\n", encoding="utf-8")

    upstream = SimpleNamespace(_iter_record_items=lambda: [])
    input_logger(
        settings=SimpleNamespace(
            run=SimpleNamespace(region="seattle"),
            beam=SimpleNamespace(config="seattle-pilates.conf"),
        ),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(
            full_path=str(tmp_path),
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam" / "input"),
            get_beam_output_dir=lambda: str(tmp_path / "beam" / "output"),
        ),
        holder=SimpleNamespace(beam_preprocess=upstream),
    )

    assert ("beam_output_plans_xml", str(output_plans), "BEAM warm-start plans (selected by BEAM from previous outputs)") in calls
    assert (
        "beam_output_experienced_plans_xml",
        str(output_experienced),
        "BEAM warm-start experienced plans (selected by BEAM from previous outputs)",
    ) in calls


def test_beam_run_fails_clearly_when_config_file_missing(monkeypatch, tmp_path):
    captured = {}

    def _fake_build_standard_step(*, spec, **_kwargs):
        captured["input_logger"] = spec.input_logger
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "build_standard_step",
        _fake_build_standard_step,
    )

    steps_beam.make_beam_run_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    input_logger = captured["input_logger"]
    monkeypatch.setattr(steps_beam.cr, "current_tracker", lambda: None)

    upstream = SimpleNamespace(_iter_record_items=lambda: [])
    try:
        input_logger(
            settings=SimpleNamespace(
                run=SimpleNamespace(region="seattle"),
                beam=SimpleNamespace(config="seattle-pilates.conf"),
            ),
            state=SimpleNamespace(),
            workspace=SimpleNamespace(
                get_beam_mutable_data_dir=lambda: str(tmp_path / "beam" / "input"),
                get_beam_output_dir=lambda: str(tmp_path / "beam" / "output"),
            ),
            holder=SimpleNamespace(beam_preprocess=upstream),
        )
    except FileNotFoundError as exc:
        assert "BEAM primary config file is missing" in str(exc)
        assert "seattle-pilates.conf" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing BEAM config")


def test_beam_run_archives_exact_runner_inputs(monkeypatch, tmp_path):
    coupler_values = {}
    step_fn = steps_beam.make_beam_run_step(
        coupler=SimpleNamespace(
            get=lambda key, default=None: coupler_values.get(key, default),
            set=lambda *args, **kwargs: None,
            update=lambda *args, **kwargs: None,
            set_from_artifact=lambda *args, **kwargs: None,
        ),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.pilates_output_replayer
    calls = []

    monkeypatch.setattr(
        steps_beam,
        "log_output_only",
        lambda *, key, path, description, **meta: calls.append(
            (key, path, description, meta)
        ),
    )

    beam_input_root = tmp_path / "beam" / "input"
    beam_output_root = tmp_path / "beam" / "output"
    scenario_dir = beam_input_root / "seattle" / "scenario"
    scenario_dir.mkdir(parents=True)
    beam_output_root.mkdir(parents=True)

    config_path = beam_input_root / "seattle" / "seattle-pilates.conf"
    config_path.write_text(
        'include "extra.conf"\n'
        'beam.test = 1\n'
        'beam.agentsim.agents.vehicles.travelTime = "scenario/network.csv"\n',
        encoding="utf-8",
    )
    (beam_input_root / "seattle" / "extra.conf").write_text(
        "beam.extra = 2\n",
        encoding="utf-8",
    )
    (scenario_dir / "network.csv").write_text("network", encoding="utf-8")

    plans_path = scenario_dir / "plans.csv.gz"
    plans_path.write_text("plans", encoding="utf-8")
    households_path = scenario_dir / "households.csv.gz"
    households_path.write_text("households", encoding="utf-8")
    persons_path = scenario_dir / "persons.csv.gz"
    persons_path.write_text("persons", encoding="utf-8")
    vehicles_path = scenario_dir / "vehicles.csv.gz"
    vehicles_path.write_text("vehicles", encoding="utf-8")
    linkstats_path = scenario_dir / "linkstats.csv.gz"
    linkstats_path.write_text("linkstats", encoding="utf-8")

    warmstart_plans = tmp_path / "previous" / "plans.csv.gz"
    warmstart_plans.parent.mkdir(parents=True)
    warmstart_plans.write_text("warm-plans", encoding="utf-8")
    warmstart_experienced = tmp_path / "previous" / "output_experienced_plans.xml.gz"
    warmstart_experienced.write_text("warm-exp", encoding="utf-8")
    coupler_values["beam_plans_out"] = str(warmstart_plans)
    coupler_values["beam_output_experienced_plans_xml"] = str(warmstart_experienced)

    outputs = BeamRunOutputs(
        beam_output_dir=beam_output_root,
        raw_outputs={},
    )
    holder = SimpleNamespace(
        beam_preprocess=BeamPreprocessOutputs(
            beam_mutable_data_dir=beam_input_root,
            prepared_inputs={
                "plans_beam_in": plans_path,
                "households_beam_in": households_path,
                "persons_beam_in": persons_path,
                "vehicles_beam_in": vehicles_path,
                "linkstats_warmstart": linkstats_path,
            },
        )
    )
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(beam_input_root),
        get_beam_output_dir=lambda: str(beam_output_root),
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(region="seattle"),
        beam=SimpleNamespace(config="seattle-pilates.conf"),
    )
    state = SimpleNamespace(year=2030, iteration=2)

    output_logger(
        outputs,
        settings=settings,
        state=state,
        workspace=workspace,
        holder=holder,
    )

    archive_dir = beam_output_root / "inputs-year-2030-iteration-2"
    assert (archive_dir / "beam_input_plans_archived.csv.gz").read_text(
        encoding="utf-8"
    ) == "plans"
    assert (archive_dir / "beam_input_households_archived.csv.gz").read_text(
        encoding="utf-8"
    ) == "households"
    assert (archive_dir / "beam_input_persons_archived.csv.gz").read_text(
        encoding="utf-8"
    ) == "persons"
    assert (archive_dir / "beam_input_config_archived.conf").read_text(
        encoding="utf-8"
    ) == (
        'include "extra.conf"\n'
        'beam.test = 1\n'
        'beam.agentsim.agents.vehicles.travelTime = "scenario/network.csv"\n'
    )
    assert (
        archive_dir / "beam_input_config_references_archived" / "extra.conf"
    ).read_text(encoding="utf-8") == "beam.extra = 2\n"
    assert (
        archive_dir
        / "beam_input_config_references_archived"
        / "scenario"
        / "network.csv"
    ).read_text(encoding="utf-8") == "network"
    assert (archive_dir / "beam_input_vehicles_archived.csv.gz").read_text(
        encoding="utf-8"
    ) == "vehicles"
    assert (archive_dir / "beam_input_linkstats_warmstart_archived.csv.gz").read_text(
        encoding="utf-8"
    ) == "linkstats"
    assert (archive_dir / "beam_input_plans_warmstart_archived.csv.gz").read_text(
        encoding="utf-8"
    ) == "warm-plans"
    assert (
        archive_dir / "beam_input_experienced_plans_warmstart_archived.xml.gz"
    ).read_text(encoding="utf-8") == "warm-exp"

    keys = [key for key, _path, _description, _meta in calls]
    assert keys == [
        "beam_input_config_references_archived",
        "beam_input_config_archived",
        "beam_input_plans_archived",
        "beam_input_households_archived",
        "beam_input_persons_archived",
        "beam_input_vehicles_archived",
        "beam_input_linkstats_warmstart_archived",
        "beam_input_plans_warmstart_archived",
        "beam_input_experienced_plans_warmstart_archived",
    ]
    snapshot_meta = {key: meta["facet"] for key, _path, _description, meta in calls}
    assert snapshot_meta["beam_input_plans_archived"] == {
        "artifact_family": "beam_input_archived",
        "input_name": "plans",
        "year": 2030,
        "iteration": 2,
    }
    assert snapshot_meta["beam_input_plans_warmstart_archived"] == {
        "artifact_family": "beam_input_archived",
        "input_name": "plans_warmstart",
        "year": 2030,
        "iteration": 2,
    }


def test_beam_run_archives_atlas_vehicles_input_when_present(monkeypatch, tmp_path):
    step_fn = steps_beam.make_beam_run_step(
        coupler=SimpleNamespace(
            get=lambda *args, **kwargs: None,
            set=lambda *args, **kwargs: None,
            update=lambda *args, **kwargs: None,
            set_from_artifact=lambda *args, **kwargs: None,
        ),
        outputs_holder=SimpleNamespace(),
    )
    output_logger = step_fn.pilates_output_replayer
    calls = []

    monkeypatch.setattr(
        steps_beam,
        "log_output_only",
        lambda *, key, path, description, **meta: calls.append((key, path, meta)),
    )

    beam_input_root = tmp_path / "beam" / "input"
    beam_output_root = tmp_path / "beam" / "output"
    scenario_dir = beam_input_root / "seattle" / "scenario"
    scenario_dir.mkdir(parents=True)
    beam_output_root.mkdir(parents=True)

    (beam_input_root / "seattle" / "seattle-pilates.conf").write_text(
        "beam.test = 1\n",
        encoding="utf-8",
    )
    (scenario_dir / "plans.csv.gz").write_text("plans", encoding="utf-8")
    (scenario_dir / "households.csv.gz").write_text("households", encoding="utf-8")
    (scenario_dir / "persons.csv.gz").write_text("persons", encoding="utf-8")
    atlas_vehicles_path = scenario_dir / "vehicles.csv.gz"
    atlas_vehicles_path.write_text("atlas-vehicles", encoding="utf-8")

    output_logger(
        BeamRunOutputs(beam_output_dir=beam_output_root, raw_outputs={}),
        settings=SimpleNamespace(
            run=SimpleNamespace(region="seattle"),
            beam=SimpleNamespace(config="seattle-pilates.conf"),
            atlas=SimpleNamespace(),
        ),
        state=SimpleNamespace(year=2035, iteration=1),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(beam_input_root),
            get_beam_output_dir=lambda: str(beam_output_root),
        ),
        holder=SimpleNamespace(
            beam_preprocess=BeamPreprocessOutputs(
                beam_mutable_data_dir=beam_input_root,
                prepared_inputs={
                    "plans_beam_in": scenario_dir / "plans.csv.gz",
                    "households_beam_in": scenario_dir / "households.csv.gz",
                    "persons_beam_in": scenario_dir / "persons.csv.gz",
                    "vehicles_beam_in": atlas_vehicles_path,
                },
            )
        ),
    )

    archived_vehicles = (
        beam_output_root
        / "inputs-year-2035-iteration-1"
        / "beam_input_vehicles_archived.csv.gz"
    )
    assert archived_vehicles.exists()
    assert archived_vehicles.read_text(encoding="utf-8") == "atlas-vehicles"
    vehicles_calls = [call for call in calls if call[0] == "beam_input_vehicles_archived"]
    assert len(vehicles_calls) == 1
    assert vehicles_calls[0][2]["facet"] == {
        "artifact_family": "beam_input_archived",
        "input_name": "vehicles",
        "year": 2035,
        "iteration": 1,
    }
