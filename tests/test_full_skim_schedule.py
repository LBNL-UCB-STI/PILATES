from types import SimpleNamespace

from pilates.workflows.stages.supply_demand import _should_run_full_skim


def _make_settings(run_schedule: str, total_iters: int = 3):
    run_cfg = SimpleNamespace(supply_demand_iters=total_iters)
    full_skim_cfg = SimpleNamespace(run_schedule=run_schedule)
    beam_cfg = SimpleNamespace(full_skim=full_skim_cfg)
    return SimpleNamespace(run=run_cfg, beam=beam_cfg)


def test_full_skim_schedule_disabled():
    settings = _make_settings("disabled")
    assert _should_run_full_skim(settings, 0) is False
    assert _should_run_full_skim(settings, 2) is False


def test_full_skim_schedule_after_each_iteration():
    settings = _make_settings("after_each_iteration", total_iters=2)
    assert _should_run_full_skim(settings, 0) is True
    assert _should_run_full_skim(settings, 1) is True


def test_full_skim_schedule_after_final_iteration():
    settings = _make_settings("after_final_iteration", total_iters=3)
    assert _should_run_full_skim(settings, 0) is False
    assert _should_run_full_skim(settings, 1) is False
    assert _should_run_full_skim(settings, 2) is True


def test_full_skim_schedule_standalone_not_in_loop():
    settings = _make_settings("standalone", total_iters=2)
    assert _should_run_full_skim(settings, 0) is False
    assert _should_run_full_skim(settings, 1) is False
