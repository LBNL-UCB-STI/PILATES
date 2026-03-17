from __future__ import annotations

from types import SimpleNamespace

from consist import define_step

from pilates.utils.coupler_helpers import log_and_set_output, log_output_only
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import StepOutputsHolder
from pilates.workflows import catalog


class _FakeScenario:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=False, run=SimpleNamespace(id="step-run"))


class _FakeCoupler:
    def __init__(self) -> None:
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)


def test_catalog_declared_key_matching_covers_dynamic_families():
    assert catalog.workflow_step_key_is_declared(
        "beam_run",
        "linkstats_2018_0",
        direction="output",
    )
    assert catalog.workflow_step_key_is_declared(
        "atlas_run",
        "atlas_static_input_example",
        direction="input",
    )
    assert not catalog.workflow_step_key_is_declared(
        "beam_postprocess",
        "not_declared",
        direction="output",
    )


def test_catalog_declared_key_matching_accepts_supported_aliases():
    assert catalog.workflow_step_key_is_declared(
        "activitysim_preprocess",
        "usim_datastore_current_h5",
        direction="input",
    )
    assert catalog.workflow_step_key_is_declared(
        "activitysim_preprocess",
        "usim_datastore_h5",
        direction="input",
    )
    assert catalog.workflow_step_key_is_declared(
        "activitysim_run",
        "asim_households_in",
        direction="input",
    )


def test_run_workflow_warns_for_undeclared_input_keys(caplog):
    scenario = _FakeScenario()
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = SimpleNamespace()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    coupler = _FakeCoupler()

    @define_step(model="activitysim_compile")
    def _dummy_step(settings, state, workspace, **kwargs):
        return None

    step = StepRef(
        name="activitysim_compile",
        step_func=_dummy_step,
        input_keys=["undeclared_input_key"],
        inputs={"undeclared_input_key": "/tmp/input.csv"},
    )

    with caplog.at_level("WARNING"):
        run_workflow(
            stage_name="unit",
            steps=[step],
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            name_suffix="unit",
        )

    assert any(
        "undeclared input key 'undeclared_input_key'" in record.message
        for record in caplog.records
    )
    assert scenario.calls[0]["inputs"] == {"undeclared_input_key": "/tmp/input.csv"}
    assert scenario.calls[0]["input_keys"] == ["undeclared_input_key"]


def test_log_and_set_output_warns_for_undeclared_output_keys(caplog, tmp_path):
    coupler = _FakeCoupler()
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("artifact", encoding="utf-8")

    with caplog.at_level("WARNING"):
        log_and_set_output(
            key="not_declared_output",
            path=str(artifact_path),
            description="test artifact",
            coupler=coupler,
            step_name="activitysim_preprocess",
        )

    assert any(
        "published undeclared output key 'not_declared_output'" in record.message
        for record in caplog.records
    )
    assert coupler.data["not_declared_output"] == str(artifact_path)


def test_log_and_set_output_does_not_warn_for_declared_dynamic_output(caplog, tmp_path):
    coupler = _FakeCoupler()
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("artifact", encoding="utf-8")

    with caplog.at_level("WARNING"):
        log_and_set_output(
            key="linkstats_2018_0",
            path=str(artifact_path),
            description="test artifact",
            coupler=coupler,
            step_name="beam_run",
        )

    assert not any(
        "published undeclared output key" in record.message for record in caplog.records
    )


def test_log_output_only_uses_same_declared_output_validation(caplog, tmp_path):
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("artifact", encoding="utf-8")

    with caplog.at_level("WARNING"):
        log_output_only(
            key="not_declared_output",
            path=str(artifact_path),
            description="test artifact",
            step_name="beam_postprocess",
        )

    assert any(
        "published undeclared output key 'not_declared_output'" in record.message
        for record in caplog.records
    )
