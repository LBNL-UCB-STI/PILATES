from types import SimpleNamespace

from pilates.workflows.step_runner import (
    StepConfig,
    build_step_config,
    common_runtime_kwargs,
    run_step,
)


class DummyScenario:
    def __init__(self) -> None:
        self.kwargs = None

    def run(self, **kwargs):
        self.kwargs = kwargs
        return {"status": "ok"}


def test_common_runtime_kwargs_includes_basics() -> None:
    runtime_kwargs = common_runtime_kwargs(
        settings="settings",
        state="state",
        workspace="workspace",
        extra_key=123,
    )
    assert runtime_kwargs == {
        "settings": "settings",
        "state": "state",
        "workspace": "workspace",
        "extra_key": 123,
    }


def test_build_step_config_defaults_from_state() -> None:
    state = SimpleNamespace(year=2025, iteration=2)
    config = build_step_config(fn=lambda: None, name="step", model="model", state=state)
    assert config.year == 2025
    assert config.iteration == 2


def test_build_step_config_overrides_year_iteration() -> None:
    state = SimpleNamespace(year=2025, iteration=2)
    config = build_step_config(
        fn=lambda: None,
        name="step",
        model="model",
        state=state,
        year=2030,
        iteration=5,
    )
    assert config.year == 2030
    assert config.iteration == 5


def test_run_step_unpacks_config() -> None:
    scenario = DummyScenario()
    config = StepConfig(
        fn=lambda: None,
        name="step",
        model="model",
        year=2025,
        iteration=1,
        inputs={"input": "path"},
        outputs=["out"],
        output_paths={"out": "path"},
        required_outputs=["out"],
        output_missing="error",
        output_mismatch="error",
        runtime_kwargs={"settings": "settings"},
        cache_mode="overwrite",
        cache_hydration="none",
        load_inputs=True,
        consist_kwargs={"enabled": True},
    )
    result = run_step(scenario, config)

    assert result == {"status": "ok"}
    assert scenario.kwargs == {
        "fn": config.fn,
        "name": "step",
        "model": "model",
        "year": 2025,
        "iteration": 1,
        "inputs": {"input": "path"},
        "outputs": ["out"],
        "output_paths": {"out": "path"},
        "required_outputs": ["out"],
        "output_missing": "error",
        "output_mismatch": "error",
        "runtime_kwargs": {"settings": "settings"},
        "cache_mode": "overwrite",
        "cache_hydration": "none",
        "load_inputs": True,
        "enabled": True,
    }
