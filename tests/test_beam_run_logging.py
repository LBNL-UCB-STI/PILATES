from types import SimpleNamespace

from pilates.workflows import steps
from pilates.workflows.steps import beam as steps_beam


def test_beam_preprocess_fails_early_when_config_file_missing(monkeypatch, tmp_path):
    captured = {}

    def _fake_make_beam_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "_make_beam_step_function",
        _fake_make_beam_step_function,
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

    def _fake_make_beam_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "_make_beam_step_function",
        _fake_make_beam_step_function,
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


def test_beam_run_fails_clearly_when_config_file_missing(monkeypatch, tmp_path):
    captured = {}

    def _fake_make_beam_step_function(**kwargs):
        captured["input_logger"] = kwargs["input_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps_beam,
        "_make_beam_step_function",
        _fake_make_beam_step_function,
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
