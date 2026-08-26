import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilates.runtime.native_canary import (
    CanaryEvidence,
    CanaryLaunchObservation,
    CanaryMount,
    StructuralCanaryCapture,
    canary_step_capture,
    check_structural_canary,
    load_structural_canary,
    main,
    record_active_container_launch,
)
from pilates.generic.runner import GenericRunner
from pilates.utils import consist_runtime as cr


def _launch(
    *,
    command: str = "run-model",
    year: int | None = None,
    iteration: int | None = None,
) -> CanaryLaunchObservation:
    return CanaryLaunchObservation(
        model="activitysim",
        step="activitysim_run",
        roles={"asim_config": "/run/resolved/asim/configs"},
        launch_roots={"workspace": "/run/resolved/asim"},
        mounts=(CanaryMount("/run/resolved/asim", "/app/asim", "rw"),),
        command=command,
        working_dir="/app/asim",
        output_roots=("/run/resolved/asim/output",),
        year=year,
        iteration=iteration,
    )


def _workspace_launch(workspace_root: str) -> CanaryLaunchObservation:
    return CanaryLaunchObservation(
        model="activitysim",
        step="activitysim_run",
        roles={"households_asim_in": "households_asim_in"},
        launch_roots={
            "activitysim_launch_context.workspace_root": workspace_root,
            "activitysim_launch_context.output_dir": (
                f"{workspace_root}/activitysim/output"
            ),
        },
        mounts=(
            CanaryMount(
                f"{workspace_root}/activitysim/data",
                "/activitysim/data",
                "rw",
            ),
        ),
        command="run-model",
        working_dir="/activitysim",
        output_roots=(f"{workspace_root}/activitysim/output",),
        year=2018,
        iteration=0,
    )


def _beam_workspace_launch(workspace_root: str) -> CanaryLaunchObservation:
    config_root = (
        f"{workspace_root}/.pilates-beam-launch-config/y2018/i0/seattle"
    )
    return CanaryLaunchObservation(
        model="beam_run",
        step="beam_run",
        roles={"zarr_skims": "zarr_skims"},
        launch_roots={
            "beam_launch_config.root": config_root,
            "beam_launch_config.primary_config": (
                f"{config_root}/seattle-pilates-base-fasterrail.conf"
            ),
        },
        mounts=(
            CanaryMount(config_root, "/app/input", "rw"),
            CanaryMount(f"{workspace_root}/beam/beam_output", "/app/output", "rw"),
            CanaryMount(
                f"{workspace_root}/.pilates-beam-launch-config/y2018/i0",
                f"{workspace_root}/.pilates-beam-launch-config/y2018/i0",
                "rw",
            ),
        ),
        command="--config=/app/input/seattle-pilates-base-fasterrail.conf",
        working_dir="/app",
        output_roots=(f"{workspace_root}/beam/beam_output/",),
        year=2018,
        iteration=0,
    )


def _capture(tmp_path: Path) -> StructuralCanaryCapture:
    required = (
        CanaryEvidence(
            "consist_snapshot", ".consist/snapshots/latest/provenance.duckdb"
        ),
        CanaryEvidence("generated_settings", "generated/settings.yaml"),
        CanaryEvidence("launch_logs", "logs/launch.log"),
        CanaryEvidence("action_v2_census", "evidence/action-v2.jsonl"),
    )
    capture = StructuralCanaryCapture(
        expected_launches=(_launch(),), required_evidence=required
    )
    for evidence in required:
        path = tmp_path / evidence.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(evidence.name, encoding="utf-8")
        capture.record_evidence(evidence.name, evidence.relative_path)
    capture.record_launch(_launch())
    return capture


def test_structural_canary_round_trips_and_checks_required_evidence(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)
    manifest_path = tmp_path / "canary.json"
    capture.write(manifest_path)

    loaded = load_structural_canary(manifest_path)
    report = check_structural_canary(loaded, evidence_root=tmp_path)

    assert report.ok
    assert report.errors == ()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_structural_canary_reports_material_launch_mismatches(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    observed = _launch(command="run-model --wrong")
    capture = StructuralCanaryCapture(
        expected_launches=(_launch(),),
        required_evidence=capture.required_evidence,
    )
    capture.record_launch(observed)
    for evidence in capture.required_evidence:
        capture.record_evidence(evidence.name, evidence.relative_path)

    report = check_structural_canary(capture.manifest(), evidence_root=tmp_path)

    assert not report.ok
    assert any("command mismatch" in error for error in report.errors)


def test_structural_canary_ignores_run_local_workspace_identity(
    tmp_path: Path,
) -> None:
    expected = _workspace_launch("/local/job111/pilates-workspace/run-a")
    observed = _workspace_launch("/local/job222/pilates-workspace/run-b")
    capture = StructuralCanaryCapture(
        expected_launches=(expected,), required_evidence=()
    )
    capture.record_launch(observed)

    report = check_structural_canary(capture.manifest(), evidence_root=tmp_path)

    assert report.ok


def test_structural_canary_normalizes_beam_run_local_workspace_identity(
    tmp_path: Path,
) -> None:
    expected = _beam_workspace_launch("/local/job111/pilates-workspace/run-a")
    observed = _beam_workspace_launch("/local/job222/pilates-workspace/run-b")
    capture = StructuralCanaryCapture(
        expected_launches=(expected,), required_evidence=()
    )
    capture.record_launch(observed)

    report = check_structural_canary(capture.manifest(), evidence_root=tmp_path)

    assert report.ok


def test_structural_canary_rejects_missing_and_unsafe_evidence(tmp_path: Path) -> None:
    capture = StructuralCanaryCapture(
        expected_launches=(_launch(),),
        required_evidence=(
            CanaryEvidence("launch_logs", "../outside.log"),
            CanaryEvidence("generated_settings", "missing/settings.yaml"),
        ),
    )
    capture.record_launch(_launch())
    capture.record_evidence("launch_logs", "../outside.log")
    capture.record_evidence("generated_settings", "missing/settings.yaml")

    report = check_structural_canary(capture.manifest(), evidence_root=tmp_path)

    assert not report.ok
    assert any("must be relative" in error for error in report.errors)
    assert any("does not exist" in error for error in report.errors)


def test_structural_canary_rejects_duplicate_observations() -> None:
    capture = StructuralCanaryCapture(
        expected_launches=(_launch(),), required_evidence=()
    )
    capture.record_launch(_launch())
    with pytest.raises(ValueError, match="duplicate launch observation"):
        capture.record_launch(_launch())


def test_structural_canary_cli_reports_success(tmp_path: Path, capsys) -> None:
    capture = _capture(tmp_path)
    manifest_path = tmp_path / "canary.json"
    capture.write(manifest_path)

    assert main([str(manifest_path), "--evidence-root", str(tmp_path)]) == 0
    assert "native structural canary evidence: OK" in capsys.readouterr().out


def test_structural_canary_cli_initializes_capture_manifest(
    tmp_path: Path, capsys
) -> None:
    manifest_path = tmp_path / "capture-seed.json"

    assert main(["--init-capture", str(manifest_path)]) == 0

    manifest = load_structural_canary(manifest_path)
    assert manifest.expected_launches == ()
    assert manifest.observed_launches == ()
    assert manifest.evidence == ()
    assert manifest.required_evidence == (
        CanaryEvidence(
            "consist_snapshot", ".consist/snapshots/latest/provenance.duckdb"
        ),
        CanaryEvidence("generated_settings", "generated/settings.yaml"),
        CanaryEvidence("launch_logs", "logs/launch.log"),
        CanaryEvidence("action_v2_census", "evidence/action-v2.jsonl"),
    )
    assert (
        "initialized native structural canary capture manifest"
        in capsys.readouterr().out
    )


def test_structural_canary_accepts_empty_schema_v1_seed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "legacy-capture-seed.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_launches": [],
                "observed_launches": [],
                "required_evidence": [],
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )

    capture = StructuralCanaryCapture.from_manifest(
        load_structural_canary(manifest_path)
    )
    capture.write(manifest_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_structural_canary_rejects_schema_v1_recorded_observations(
    tmp_path: Path,
) -> None:
    legacy_launch = _launch().to_dict()
    manifest_path = tmp_path / "legacy-observed.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_launches": [],
                "observed_launches": [legacy_launch],
                "required_evidence": [],
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema v1.*cannot be reused"):
        load_structural_canary(manifest_path)


def test_structural_canary_cli_does_not_overwrite_capture_manifest(
    tmp_path: Path, capsys
) -> None:
    manifest_path = tmp_path / "capture-seed.json"
    manifest_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["--init-capture", str(manifest_path)])

    assert error.value.code == 2
    assert manifest_path.read_text(encoding="utf-8") == "existing\n"
    assert "already exists" in capsys.readouterr().err


def test_structural_canary_cli_records_evidence_before_checking(
    tmp_path: Path,
) -> None:
    """A retained file becomes checkable evidence without hand-editing JSON."""

    evidence = CanaryEvidence("launch_logs", "logs/launch.log")
    capture = StructuralCanaryCapture(
        expected_launches=(_launch(),), required_evidence=(evidence,)
    )
    capture.record_launch(_launch())
    manifest_path = tmp_path / "canary.json"
    capture.write(manifest_path)
    evidence_path = tmp_path / evidence.relative_path
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("launch", encoding="utf-8")

    assert (
        main(
            [
                str(manifest_path),
                "--evidence-root",
                str(tmp_path),
                "--record-evidence",
                evidence.name,
                evidence.relative_path,
            ]
        )
        == 0
    )
    assert load_structural_canary(manifest_path).evidence == (evidence,)


def test_opt_in_collector_records_the_actual_container_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real container boundary appends its observed launch to a seeded manifest."""

    capture = StructuralCanaryCapture(
        expected_launches=(_launch(),), required_evidence=()
    )
    manifest_path = tmp_path / "canary.json"
    capture.write(manifest_path)
    monkeypatch.setenv("PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST", str(manifest_path))

    with canary_step_capture(
        step="activitysim_run",
        roles={"asim_config": "/run/resolved/asim/configs"},
        launch_roots={"workspace": "/run/resolved/asim"},
    ):
        record_active_container_launch(
            model="activitysim",
            volumes={"/run/resolved/asim": {"bind": "/app/asim", "mode": "rw"}},
            command=["run-model"],
            working_dir="/app/asim",
            output_paths=["/run/resolved/asim/output"],
        )

    manifest = load_structural_canary(manifest_path)

    assert manifest.observed_launches == (_launch(),)
    assert check_structural_canary(manifest, evidence_root=tmp_path).ok


def test_opt_in_collector_distinguishes_repeated_step_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated model steps remain distinct observations by iteration."""

    manifest_path = tmp_path / "canary.json"
    expected = tuple(
        _launch(year=2018, iteration=iteration) for iteration in (0, 1)
    )
    StructuralCanaryCapture(expected_launches=expected, required_evidence=()).write(
        manifest_path
    )
    monkeypatch.setenv("PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST", str(manifest_path))

    for iteration in (0, 1):
        with canary_step_capture(
            step="activitysim_run",
            year=2018,
            iteration=iteration,
            roles={"asim_config": "/run/resolved/asim/configs"},
            launch_roots={"workspace": "/run/resolved/asim"},
        ):
            record_active_container_launch(
                model="activitysim",
                volumes={
                    "/run/resolved/asim": {"bind": "/app/asim", "mode": "rw"}
                },
                command=["run-model"],
                working_dir="/app/asim",
                output_paths=["/run/resolved/asim/output"],
            )

    observed = load_structural_canary(manifest_path).observed_launches

    assert [launch.key for launch in observed] == [
        "activitysim/activitysim_run/y2018/i0",
        "activitysim/activitysim_run/y2018/i1",
    ]
    assert check_structural_canary(
        load_structural_canary(manifest_path), evidence_root=tmp_path
    ).ok


def test_opt_in_collector_observes_the_shared_container_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the shared container hook would leave an active canary empty."""

    manifest_path = tmp_path / "canary.json"
    StructuralCanaryCapture(expected_launches=(_launch(),), required_evidence=()).write(
        manifest_path
    )
    monkeypatch.setenv("PILATES_NATIVE_STRUCTURAL_CANARY_MANIFEST", str(manifest_path))
    monkeypatch.setattr(cr, "current_tracker", lambda: object())

    from consist.integrations import containers

    monkeypatch.setattr(containers, "run_container", lambda **_kwargs: True)
    settings = SimpleNamespace(
        run=SimpleNamespace(use_stubs=False),
        infrastructure=SimpleNamespace(
            container_manager="docker",
            docker_config=SimpleNamespace(pull_latest=False, stdout=False),
        ),
    )
    with canary_step_capture(
        step="activitysim_run",
        roles={"asim_config": "/run/resolved/asim/configs"},
        launch_roots={"workspace": "/run/resolved/asim"},
    ):
        assert GenericRunner.run_container(
            client=None,
            settings=settings,
            image="activitysim:canary",
            volumes={"/run/resolved/asim": {"bind": "/app/asim", "mode": "rw"}},
            command="run-model",
            model_name="activitysim",
            working_dir="/app/asim",
            output_paths=["/run/resolved/asim/output"],
        )

    assert check_structural_canary(
        load_structural_canary(manifest_path), evidence_root=tmp_path
    ).ok
