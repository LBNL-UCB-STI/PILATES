from types import SimpleNamespace

from pilates.workflows import steps
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

    steps.make_beam_preprocess_step(
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

    steps.make_beam_run_step(
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

    steps.make_beam_run_step(
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

    steps.make_beam_run_step(
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
