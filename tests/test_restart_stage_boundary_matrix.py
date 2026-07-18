"""Focused BEAM restart diagnostics and explicit canary policy.

Whole-stage and whole-year execution are exercised through the native golden
workflow.  This module intentionally avoids legacy ``StageRunner``, manifest,
and fake-scenario recreation of unsupported mid-stage restart policies.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from consist import BindingResult

from pilates.workflows.artifact_keys import (
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS,
)
from pilates.workflows.binding import BindingPlan
from pilates.workflows.beam_checkpoint import PinnedClosureMember
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.stages import supply_demand_beam as beam_stage
from pilates.workflows.resume import (
    HistoricalOutputRequest,
    RestoreExecutionResult,
    ResumeDecision,
    ResumeDisposition,
)
from pilates.workflows.stages.supply_demand_beam import (
    _FAIL_AFTER_BEAM_RUN_ENV,
    _emit_beam_restart_recovery_readiness_diagnostic,
    _maybe_fail_after_beam_run_for_canary,
    beam_preprocess_binding_diagnostic_payload,
)


def _restart_binding(
    *,
    config_path: Path,
    vehicles_path: Path,
    source_by_key: dict[str, str],
) -> BindingPlan:
    return BindingPlan(
        step_name="beam_preprocess",
        inputs={
            BEAM_PLANS_IN: "/tmp/plans.parquet",
            BEAM_HOUSEHOLDS_IN: "/tmp/households.parquet",
            BEAM_PERSONS_IN: "/tmp/persons.parquet",
            ATLAS_VEHICLES2_OUTPUT: str(vehicles_path),
            BEAM_CONFIG_FILE: str(config_path),
        },
        input_keys=[
            BEAM_PLANS_IN,
            BEAM_HOUSEHOLDS_IN,
            BEAM_PERSONS_IN,
            BEAM_CONFIG_FILE,
            ATLAS_VEHICLES2_OUTPUT,
        ],
        source_by_key=source_by_key,
        coupler_key_by_key={ATLAS_VEHICLES2_OUTPUT: ATLAS_VEHICLES2_OUTPUT},
    )


def _restart_context(
    tmp_path: Path,
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, Path, Path]:
    beam_root = tmp_path / "beam" / "input"
    region_dir = beam_root / "sfbay"
    region_dir.mkdir(parents=True)
    config_path = region_dir / "beam.conf"
    config_path.write_text("beam config", encoding="utf-8")
    vehicles_path = tmp_path / "atlas" / "vehicles2_2021.csv"
    vehicles_path.parent.mkdir(parents=True)
    vehicles_path.write_text("vehicleId,householdId\n1,10\n", encoding="utf-8")
    workspace = SimpleNamespace(get_beam_mutable_data_dir=lambda: str(beam_root))
    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(config="beam.conf"),
    )
    state = SimpleNamespace(
        is_restart_run=True,
        year=2019,
        forecast_year=2021,
        iteration=0,
    )
    return workspace, settings, state, config_path, vehicles_path


def test_beam_restart_binding_diagnostic_classifies_complete_binding(tmp_path):
    workspace, settings, state, config_path, vehicles_path = _restart_context(tmp_path)

    payload = beam_preprocess_binding_diagnostic_payload(
        binding=_restart_binding(
            config_path=config_path,
            vehicles_path=vehicles_path,
            source_by_key={
                ATLAS_VEHICLES2_OUTPUT: "coupler",
                BEAM_CONFIG_FILE: "expected_inputs",
            },
        ),
        state=state,
        settings=settings,
        workspace=workspace,
    )

    assert payload["drift_classification"] == "binding_surface_complete"
    assert payload["missing_restart_inputs"] == []
    assert ATLAS_VEHICLES2_OUTPUT in payload["bound_input_keys"]
    assert payload["required_local_inputs"]["beam_primary_config_file"] == str(
        config_path
    )


def test_beam_restart_canary_failpoint_requires_explicit_env(monkeypatch):
    monkeypatch.delenv(_FAIL_AFTER_BEAM_RUN_ENV, raising=False)

    _maybe_fail_after_beam_run_for_canary(year=2021, iteration=0)

    monkeypatch.setenv(_FAIL_AFTER_BEAM_RUN_ENV, "1")
    with pytest.raises(RuntimeError, match="Injected failure after completed beam_run"):
        _maybe_fail_after_beam_run_for_canary(year=2021, iteration=0)


def test_committed_checkpoint_dispatch_skips_native_preprocess_and_run(monkeypatch):
    scenario = SimpleNamespace(coupler=object())
    context = SimpleNamespace(
        settings=object(),
        state=SimpleNamespace(is_restart_run=True),
        workspace=object(),
    )
    expected = {"events_parquet_2021_0_type_PathTraversal": "/tmp/events"}

    monkeypatch.setattr(
        beam_stage, "beam_checkpoint_resume_requested", lambda **_: True
    )
    monkeypatch.setattr(
        beam_stage,
        "_try_resume_committed_beam_postprocess",
        lambda **kwargs: expected,
    )
    monkeypatch.setattr(
        beam_stage,
        "execute_step",
        lambda **_kwargs: pytest.fail("resume must not execute preprocess or run"),
    )

    assert (
        beam_stage._run_beam_steps(
            scenario=scenario,
            year=2021,
            iteration=0,
            context=context,
        )
        == expected
    )


def test_fresh_checkpoint_is_published_before_canary_failpoint(monkeypatch):
    resolved_inputs = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
    )
    scenario = SimpleNamespace(coupler=object())
    context = SimpleNamespace(
        settings=object(),
        state=SimpleNamespace(is_restart_run=False),
        workspace=object(),
    )
    calls: list[str] = []
    postprocess_definition = SimpleNamespace(
        resolve_inputs=lambda **_kwargs: resolved_inputs
    )
    run_outputs = SimpleNamespace(_iter_record_items=lambda: ())

    def fake_execute_step(*, definition, phase, **_kwargs):
        calls.append(phase)
        if phase == "run":
            return SimpleNamespace(run=SimpleNamespace(id="beam-run-1")), run_outputs
        if phase == "preprocess":
            return SimpleNamespace(), SimpleNamespace()
        pytest.fail("canary failure must happen before native postprocess")

    def publish(**kwargs):
        calls.append("published")
        assert kwargs["producer_run_id"] == "beam-run-1"
        assert kwargs["postprocess_inputs"] is resolved_inputs

    def failpoint(**_kwargs):
        assert calls == ["preprocess", "run", "published"]
        raise RuntimeError("canary")

    monkeypatch.setattr(
        beam_stage, "beam_checkpoint_resume_requested", lambda **_: False
    )
    monkeypatch.setattr(beam_stage, "beam_postprocess", postprocess_definition)
    monkeypatch.setattr(beam_stage, "execute_step", fake_execute_step)
    monkeypatch.setattr(beam_stage, "_publish_completed_beam_run_checkpoint", publish)
    monkeypatch.setattr(beam_stage, "_maybe_fail_after_beam_run_for_canary", failpoint)

    with pytest.raises(RuntimeError, match="canary"):
        beam_stage._run_beam_steps(
            scenario=scenario,
            year=2021,
            iteration=0,
            context=context,
        )

    assert calls == ["preprocess", "run", "published"]


@pytest.mark.parametrize("run_mode", ("fresh", "cache"))
def test_public_beam_handoff_is_postprocess_owned_across_run_modes_and_resume(
    monkeypatch, run_mode
):
    """The public boundary exposes only the successor postprocess handoff."""
    postprocess_artifact = SimpleNamespace(id="postprocess-artifact")

    class Coupler:
        def get(self, key, default=None):
            if key == "postprocess_output":
                return postprocess_artifact
            return default

    scenario = SimpleNamespace(coupler=Coupler())
    context = SimpleNamespace(
        settings=object(),
        state=SimpleNamespace(is_restart_run=False),
        workspace=object(),
    )
    resolved_inputs = ResolvedStepInputs(
        step_name="beam_postprocess",
        binding=BindingResult(inputs={}),
    )
    postprocess_definition = SimpleNamespace(
        resolve_inputs=lambda **_kwargs: resolved_inputs
    )
    run_outputs = SimpleNamespace(
        _iter_record_items=lambda: (("beam_run_output", "/tmp/beam-run", ""),)
    )
    postprocess_outputs = SimpleNamespace(
        _iter_record_items=lambda: (("postprocess_output", "/tmp/postprocess", ""),)
    )

    def fake_execute_step(*, phase, **_kwargs):
        if phase == "run":
            return (
                SimpleNamespace(
                    run=SimpleNamespace(id=f"beam-run-{run_mode}"),
                    cache_hit=run_mode == "cache",
                ),
                run_outputs,
            )
        if phase == "postprocess":
            return SimpleNamespace(), postprocess_outputs
        return SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(
        beam_stage, "beam_checkpoint_resume_requested", lambda **_: False
    )
    monkeypatch.setattr(beam_stage, "beam_postprocess", postprocess_definition)
    monkeypatch.setattr(beam_stage, "execute_step", fake_execute_step)
    monkeypatch.setattr(
        beam_stage, "_publish_completed_beam_run_checkpoint", lambda **_: None
    )
    monkeypatch.setattr(
        beam_stage, "_maybe_fail_after_beam_run_for_canary", lambda **_: None
    )
    monkeypatch.setattr(beam_stage, "_archive_run_dir_for_restart", lambda _state: None)

    handoff = beam_stage._run_beam_steps(
        scenario=scenario,
        year=2021,
        iteration=0,
        context=context,
    )

    assert set(handoff) == {"postprocess_output"}
    assert "beam_run_output" not in handoff
    assert handoff["postprocess_output"] is postprocess_artifact

    monkeypatch.setattr(
        beam_stage, "beam_checkpoint_resume_requested", lambda **_: True
    )
    monkeypatch.setattr(
        beam_stage, "_try_resume_committed_beam_postprocess", lambda **_: handoff
    )

    resumed_handoff = beam_stage._run_beam_steps(
        scenario=scenario,
        year=2021,
        iteration=0,
        context=context,
    )

    assert resumed_handoff == handoff
    assert resumed_handoff["postprocess_output"] is postprocess_artifact


def test_rebound_checkpoint_requires_normal_resolver_identity_and_destination(tmp_path):
    destination = tmp_path / "beam" / ".pilates-consist-inputs" / "events.parquet"
    member = PinnedClosureMember(
        member_id="beam-run-1:events_parquet_2021_0",
        role="events_parquet_2021_0",
        producer_run_id="beam-run-1",
        output_key="events_parquet_2021_0",
        artifact_identity="events-hash",
        artifact_kind="file",
        driver="parquet",
        destination=destination,
        required=True,
    )

    def resolved(*, artifact_hash: str, resolved_destination: Path):
        return ResolvedStepInputs(
            step_name="beam_postprocess",
            binding=BindingResult(
                inputs={"events_parquet_2021_0": SimpleNamespace(hash=artifact_hash)}
            ),
            required_roles=("events_parquet_2021_0",),
            source_by_role={"events_parquet_2021_0": "coupler"},
            selected_key_by_role={"events_parquet_2021_0": "events_parquet_2021_0"},
            logical_destinations={"events_parquet_2021_0": resolved_destination},
        )

    checkpoint = SimpleNamespace(producer_run_id="beam-run-1")
    beam_stage._validate_rebound_postprocess_inputs(
        checkpoint=checkpoint,
        members=(member,),
        resolved_inputs=resolved(
            artifact_hash="events-hash", resolved_destination=destination
        ),
    )

    with pytest.raises(RuntimeError, match="identity drifted"):
        beam_stage._validate_rebound_postprocess_inputs(
            checkpoint=checkpoint,
            members=(member,),
            resolved_inputs=resolved(
                artifact_hash="different-hash", resolved_destination=destination
            ),
        )


def test_beam_restart_recovery_readiness_diagnostic_uses_existing_restore_result(
    monkeypatch, tmp_path
):
    from pilates.workflows.stages import supply_demand_beam as beam_stage

    run_id = "current-run__beam_run"
    state = SimpleNamespace(
        is_restart_run=True,
        year=2019,
        forecast_year=2021,
        current_year=2019,
    )
    decision = ResumeDecision(
        step_name="beam_run",
        disposition=ResumeDisposition.RESTORE,
        reason="completed_match",
        semantic_target={"run_scope": "current-run", "status": "completed"},
        source_run_id=run_id,
        outputs=(
            HistoricalOutputRequest("events_parquet_2021_0", tmp_path / "events", True),
            HistoricalOutputRequest(LINKSTATS, tmp_path / "linkstats", True),
        ),
        rerun_forbidden=True,
    )
    execution = RestoreExecutionResult(
        decision=decision,
        hydration_result=None,
        projected_outputs=object(),
        published_role_keys=(LINKSTATS,),
        failure_category=None,
        failed_keys=(),
    )
    events_emitted = []
    monkeypatch.setattr(
        beam_stage,
        "_emit_artifact_lifecycle_event",
        lambda event_type, **fields: events_emitted.append((event_type, fields)),
    )

    _emit_beam_restart_recovery_readiness_diagnostic(
        state=state,
        decision=decision,
        execution=execution,
        iteration=0,
    )

    assert events_emitted
    event_type, fields = events_emitted[0]
    assert event_type == "beam_restart_recovery_readiness"
    assert fields["matchable"] is True
    assert fields["matched_completed_run_id"] == run_id
    assert fields["matched_run_id"] == run_id
    assert fields["required_restored_inputs"] == ["events_parquet_2021_0", LINKSTATS]
    assert fields["missing_restored_inputs"] == []
    assert fields["missing_required_keys"] == []
    assert fields["hydration_api_available"] is True
    assert fields["drift_classification"] == "complete"


def test_beam_restart_binding_diagnostic_classifies_cache_drift(tmp_path):
    workspace, settings, state, config_path, vehicles_path = _restart_context(tmp_path)

    payload = beam_preprocess_binding_diagnostic_payload(
        binding=_restart_binding(
            config_path=config_path,
            vehicles_path=vehicles_path,
            source_by_key={
                ATLAS_VEHICLES2_OUTPUT: "coupler",
                BEAM_CONFIG_FILE: "expected_inputs",
            },
        ),
        state=state,
        settings=settings,
        workspace=workspace,
        identity_context={
            "identity_summary": {"adapter": {"hash": "abc"}},
            "cache_miss_explanation": {
                "reason": "config_and_inputs_changed",
                "adapter_identity_changed": ["config_bundle_hash"],
                "input_keys_changed": ["atlas_vehicles2_output"],
            },
        },
    )

    assert payload["drift_classification"] == "content_or_config_drift"
    assert payload["identity_summary"] == {"adapter": {"hash": "abc"}}
    assert payload["identity_drift_components"] == {
        "adapter_identity_changed": ["config_bundle_hash"],
        "input_keys_changed": ["atlas_vehicles2_output"],
    }
