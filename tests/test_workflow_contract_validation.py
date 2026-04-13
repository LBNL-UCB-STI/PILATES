from __future__ import annotations

from types import SimpleNamespace

from consist import define_step
from pilates.workflows.binding import BindingPlan

from pilates.utils.coupler_helpers import (
    artifact_to_path,
    log_and_set_output,
    log_output_only,
)
from pilates.workflows.input_resolution import resolve_step_inputs
from pilates.workflows.coupler_namespace import (
    canonical_artifact_key_from_raw_key,
    resolve_coupler_value,
)
from pilates.workflows.orchestration import StepRef, run_workflow
from pilates.workflows.steps import (
    StepOutputsHolder,
    urbansim_run_output_paths,
)
from pilates.workflows import catalog
from pilates.workflows.steps.urbansim_atlas import make_urbansim_run_step
from pilates.urbansim.runner import UrbansimRunner


class _FakeScenario:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        binding = kwargs.get("binding")
        if binding is not None:
            kwargs = dict(kwargs)
            kwargs["inputs"] = binding.inputs or {}
            kwargs["input_keys"] = list(binding.input_keys or [])
            kwargs["optional_input_keys"] = list(binding.optional_input_keys or [])
        self.calls.append(kwargs)
        return SimpleNamespace(cache_hit=False, run=SimpleNamespace(id="step-run"))


class _FakeCoupler:
    def __init__(self) -> None:
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def view(self, namespace):
        parent = self

        class _View:
            def get(self, key, default=None):
                return parent.data.get(f"{namespace}/{key}", default)

        return _View()


class _FakeWorkspace:
    def __init__(self, full_path: str) -> None:
        self.full_path = full_path

    def get_usim_mutable_data_dir(self) -> str:
        return f"{self.full_path}/usim"


def test_catalog_declared_key_matching_covers_dynamic_families():
    assert catalog.workflow_step_key_is_declared(
        "beam_run",
        "linkstats_2018_0",
        direction="output",
    )
    assert catalog.workflow_step_key_is_declared(
        "atlas_run",
        "cpi",
        direction="input",
    )
    assert catalog.workflow_step_key_is_declared(
        "atlas_run",
        "adopt/zev_mandate/new_vehicles_biannual_values_2023",
        direction="input",
    )
    assert catalog.workflow_step_key_is_declared(
        "atlas_run",
        "vehicle_type_mapping_evMandForced2",
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
        "usim_population_source_h5",
        direction="input",
    )
    assert not catalog.workflow_step_key_is_declared(
        "activitysim_preprocess",
        "usim_datastore_current_h5",
        direction="input",
    )
    assert not catalog.workflow_step_key_is_declared(
        "activitysim_preprocess",
        "usim_datastore_h5",
        direction="input",
    )
    assert catalog.workflow_step_key_is_declared(
        "activitysim_run",
        "asim_households_in",
        direction="input",
    )
    assert catalog.workflow_step_key_is_declared(
        "urbansim_postprocess",
        "urbansim/usim_datastore_h5",
        direction="input",
    )


def test_canonical_artifact_key_normalizes_namespaced_and_alias_keys():
    assert (
        canonical_artifact_key_from_raw_key("urbansim/usim_datastore_h5")
        == "usim_datastore_h5"
    )
    assert (
        canonical_artifact_key_from_raw_key("usim_datastore_current_h5")
        == "usim_datastore_h5"
    )
    assert (
        canonical_artifact_key_from_raw_key(
            "adopt/baseline/new_vehicles_biannual_values_2023"
        )
        == "adopt/baseline/new_vehicles_biannual_values_2023"
    )


def test_resolve_coupler_value_preserves_canonical_key_and_storage_key():
    coupler = _FakeCoupler()
    coupler.data["urbansim/usim_datastore_h5"] = "/tmp/model_data_2023.h5"

    resolved = resolve_coupler_value(coupler, "usim_datastore_h5")

    assert resolved.value == "/tmp/model_data_2023.h5"
    assert resolved.canonical_key == "usim_datastore_h5"
    assert resolved.storage_key == "urbansim/usim_datastore_h5"
    assert resolved.source == "coupler"


def test_resolve_step_inputs_preserves_canonical_key_when_coupler_uses_namespace():
    coupler = _FakeCoupler()
    coupler.data["urbansim/usim_datastore_h5"] = "/tmp/model_data_2023.h5"

    resolved = resolve_step_inputs(
        keys=["usim_datastore_h5"],
        coupler=coupler,
    )

    assert resolved.input_keys == ["usim_datastore_h5"]
    assert resolved.coupler_key_by_key["usim_datastore_h5"] == "urbansim/usim_datastore_h5"
    assert resolved.source_by_key["usim_datastore_h5"] == "coupler"


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


def test_run_workflow_uses_step_output_path_provider():
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

    def _output_paths_provider(*, settings, state, workspace):
        return {"zarr_skims": "/tmp/workspace/activitysim/cache/skims.zarr"}

    step = StepRef(
        name="activitysim_compile",
        step_func=_dummy_step,
        output_paths_provider=_output_paths_provider,
    )

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

    assert scenario.calls[0]["output_paths"] == {
        "zarr_skims": "/tmp/workspace/activitysim/cache/skims.zarr"
    }


def test_run_workflow_uses_decorator_output_paths_callable():
    scenario = _FakeScenario()
    outputs_holder = StepOutputsHolder()
    outputs_holder.activitysim_preprocess = SimpleNamespace()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    coupler = _FakeCoupler()

    @define_step(
        model="activitysim_compile",
        output_paths=lambda *, settings, state, workspace: {
            "zarr_skims": f"{workspace.full_path}/activitysim/cache/skims.zarr"
        },
    )
    def _dummy_step(settings, state, workspace, **kwargs):
        return None

    step = StepRef(name="activitysim_compile", step_func=_dummy_step)

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

    assert scenario.calls[0]["output_paths"] == {
        "zarr_skims": "/tmp/workspace/activitysim/cache/skims.zarr"
    }


def test_run_workflow_uses_urbansim_run_output_path_provider():
    scenario = _FakeScenario()
    outputs_holder = StepOutputsHolder()
    outputs_holder.urbansim_preprocess = SimpleNamespace()
    workspace = _FakeWorkspace("/tmp/workspace")
    settings = SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            local_mutable_data_folder="urbansim/data",
            region_mappings={"region_to_region_id": {"test": "001"}},
            input_file_template="input_{region_id}.h5",
            output_file_template="usim_{year}.h5",
        ),
    )
    state = SimpleNamespace(
        year=2020,
        forecast_year=2020,
        iteration=0,
        is_start_year=lambda: False,
    )
    coupler = _FakeCoupler()

    run_step = make_urbansim_run_step(coupler=coupler, outputs_holder=outputs_holder)
    step = StepRef(
        name="urbansim_run",
        step_func=run_step,
        output_paths_provider=urbansim_run_output_paths,
    )

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

    assert scenario.calls[0]["output_paths"] == UrbansimRunner.expected_outputs(
        settings,
        state,
        workspace,
    )


def test_run_workflow_does_not_warn_for_component_local_expected_inputs(caplog):
    scenario = _FakeScenario()
    outputs_holder = StepOutputsHolder()
    workspace = _FakeWorkspace("/tmp/workspace")
    settings = SimpleNamespace(
        run=SimpleNamespace(region="test_region"),
        urbansim=SimpleNamespace(
            local_data_input_folder="pilates/urbansim/data",
            region_mappings={"region_to_region_id": {"test_region": "123"}},
            input_file_template="input_{region_id}.h5",
        ),
    )
    state = SimpleNamespace(year=2020, iteration=0)
    coupler = _FakeCoupler()

    @define_step(model="urbansim_preprocess")
    def _dummy_step(settings, state, workspace, **kwargs):
        return None

    step = StepRef(
        name="urbansim_preprocess",
        step_func=_dummy_step,
        inputs={
            "usim_source_data_dir": "/tmp/source",
            "usim_mutable_data_dir": "/tmp/workspace/usim",
        },
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

    assert not any(
        "[CONTRACT-ENFORCEMENT][urbansim_preprocess]" in record.message
        for record in caplog.records
    )


def test_run_workflow_accepts_binding_result_for_pilot_call_sites():
    scenario = _FakeScenario()
    outputs_holder = StepOutputsHolder()
    workspace = SimpleNamespace(full_path="/tmp/workspace")
    settings = SimpleNamespace()
    state = SimpleNamespace(year=2020, iteration=0)
    coupler = _FakeCoupler()

    @define_step(model="dummy_step")
    def _dummy_step(settings, state, workspace, **kwargs):
        return None

    binding = BindingPlan()
    step = StepRef(name="dummy_step", step_func=_dummy_step, binding=binding)

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

    call = scenario.calls[0]
    assert call["binding"] == binding.to_binding_result()
    assert call["inputs"] == {}
    assert call["input_keys"] == []
    assert call["optional_input_keys"] == []


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
    assert artifact_to_path(coupler.data["not_declared_output"]) == str(artifact_path)


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


def test_log_and_set_output_does_not_warn_for_declared_optional_output(caplog, tmp_path):
    coupler = _FakeCoupler()
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("artifact", encoding="utf-8")

    with caplog.at_level("WARNING"):
        log_and_set_output(
            key="linkstats_warmstart",
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
