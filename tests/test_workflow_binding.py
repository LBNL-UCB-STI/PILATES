"""ResolvedBinding V1 bridge contracts."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest
from consist import AdmissionEvidence, ResolvedBinding, Tracker, define_step
from consist.core.resolved_binding import ArtifactIdentity

from pilates.utils.usim_h5 import create_usim_population_source_snapshot
from pilates.workflows import binding
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_HOUSEHOLDS_TABLE,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.binding import build_resolved_binding


def _write_population_store(path: Path, tables: tuple[str, ...]) -> Path:
    with pd.HDFStore(path, mode="w") as store:
        for index, table_name in enumerate(tables):
            store.put(table_name, pd.DataFrame({"marker": [index]}))
    return path


class _LandUseContinuationState:
    forecast_year = 2030

    class Stage:
        land_use = object()

    def is_enabled(self, stage: object) -> bool:
        return stage is self.Stage.land_use

    def is_start_year(self) -> bool:
        return False


def _tracked_artifact(tmp_path: Path):
    tracker = Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
        hashing_strategy="full",
    )
    source = tmp_path / "payload.txt"
    source.write_text("selected\n", encoding="utf-8")
    with tracker.start_run("seed", "test"):
        return tracker.log_artifact(source, key="payload", direction="input")


def _step_identity(step_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"{step_name}__y2030__i1__phase_test",
        step_contract_identity="sha256:step-v1:" + "0" * 64,
    )


def test_activitysim_population_source_resolves_builder_snapshot_exact_year(
    tmp_path: Path,
) -> None:
    source_h5 = _write_population_store(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    snapshot_h5 = create_usim_population_source_snapshot(
        source_h5=source_h5,
        destination_h5=tmp_path / "valid_population_source.h5",
        year=2030,
    )

    resolved = binding._activitysim_population_source(
        explicit_fallback_inputs={USIM_POPULATION_SOURCE_H5: snapshot_h5},
        settings=SimpleNamespace(),
        state=_LandUseContinuationState(),
        workspace=SimpleNamespace(),
    )

    assert resolved is not None
    assert resolved[USIM_POPULATION_HOUSEHOLDS_TABLE] == "/2030/households"


def test_activitysim_population_source_does_not_repair_malformed_snapshot(
    tmp_path: Path,
) -> None:
    malformed_h5 = _write_population_store(
        tmp_path / "malformed_population_source.h5",
        ("households", "persons", "jobs", "blocks"),
    )

    with pytest.raises(KeyError, match="missing required tables"):
        binding._activitysim_population_source(
            explicit_fallback_inputs={USIM_POPULATION_SOURCE_H5: malformed_h5},
            settings=SimpleNamespace(),
            state=_LandUseContinuationState(),
            workspace=SimpleNamespace(),
        )

    with h5py.File(malformed_h5, "r") as handle:
        assert "/2030" not in handle


def test_build_resolved_binding_freezes_tracked_named_inputs_without_diagnostics_in_identity(
    tmp_path: Path,
) -> None:
    artifact = _tracked_artifact(tmp_path)

    @define_step(model="example")
    def consume(payload: Path, *, settings: object) -> None:
        del payload, settings

    first = build_resolved_binding(
        step_name="consume",
        function=consume,
        selected_artifacts={"payload": artifact},
        logical_destinations={"payload": Path("inputs/payload.txt")},
        selection_diagnostics={"selection": "preferred"},
        step_identity=_step_identity("consume"),
    )
    second = build_resolved_binding(
        step_name="consume",
        function=consume,
        selected_artifacts={"payload": artifact},
        logical_destinations={"payload": Path("inputs/payload.txt")},
        selection_diagnostics={"selection": "fallback"},
        step_identity=_step_identity("consume"),
    )

    assert isinstance(first, ResolvedBinding)
    assert first.identity_json() == second.identity_json()
    assert first.identity_digest() == second.identity_digest()
    assert first.evidence_json() != second.evidence_json()


def test_build_resolved_binding_rejects_invalid_v1_inputs(tmp_path: Path) -> None:
    artifact = _tracked_artifact(tmp_path)

    def consume(payload: Path) -> None:
        del payload

    with pytest.raises(ValueError, match="relative"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": artifact},
            logical_destinations={"payload": tmp_path / "absolute.txt"},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
        )
    with pytest.raises(TypeError, match="tracked Artifact"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": tmp_path / "untracked.txt"},
            logical_destinations={"payload": Path("inputs/payload.txt")},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
        )
    with pytest.raises(ValueError, match="named callable parameter"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"missing": artifact},
            logical_destinations={"missing": Path("inputs/payload.txt")},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
        )

    def consume_pair(first: Path, second: Path) -> None:
        del first, second

    with pytest.raises(ValueError, match="unique"):
        build_resolved_binding(
            step_name="consume_pair",
            function=consume_pair,
            selected_artifacts={"first": artifact, "second": artifact},
            logical_destinations={
                "first": Path("inputs/payload.txt"),
                "second": Path("inputs/payload.txt"),
            },
            selection_diagnostics={},
            step_identity=_step_identity("consume_pair"),
        )


def test_build_resolved_binding_rejects_admission_identity_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _tracked_artifact(tmp_path)

    def consume(payload: Path) -> None:
        del payload

    with pytest.raises(ValueError, match="observed identity"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": artifact},
            logical_destinations={"payload": Path("inputs/payload.txt")},
            selection_diagnostics={},
            step_identity=_step_identity("consume"),
            admission_evidence={
                "payload": AdmissionEvidence(
                    observed_identity=ArtifactIdentity.parse("sha256:file:" + "0" * 64),
                    expected_identity=None,
                    expected_source=None,
                )
            },
        )


def test_build_resolved_binding_requires_consist_preflight_identity(
    tmp_path: Path,
) -> None:
    artifact = _tracked_artifact(tmp_path)

    def consume(payload: Path) -> None:
        del payload

    with pytest.raises(TypeError, match="step_identity"):
        build_resolved_binding(
            step_name="consume",
            function=consume,
            selected_artifacts={"payload": artifact},
            logical_destinations={"payload": Path("inputs/payload.txt")},
            selection_diagnostics={},
        )


def test_restart_activitysim_required_artifacts_uses_forecast_year(
    tmp_path: Path,
) -> None:
    stage = SimpleNamespace(
        supply_demand_loop="supply_demand_loop",
        activity_demand="activity_demand",
        activity_demand_directly_from_land_use="activity_demand_directly_from_land_use",
        land_use="land_use",
    )
    state = SimpleNamespace(
        year=2017,
        start_year=2017,
        forecast_year=2019,
        current_major_stage="supply_demand_loop",
        is_enabled=lambda s: s in ("land_use", stage.land_use),
        is_start_year=lambda: True,
        Stage=stage,
    )
    settings = SimpleNamespace(
        run=SimpleNamespace(
            models=SimpleNamespace(activity_demand="activitysim"),
            start_year=2017,
            region="sfbay",
        ),
        activitysim=SimpleNamespace(
            main_configs_dir="configs",
            local_mutable_configs_folder=str(tmp_path / "configs"),
        ),
        urbansim=SimpleNamespace(
            output_file_template="model_data_{year}.h5",
            input_file_template="usim_base_{region_id}.h5",
            region_mappings={"region_to_region_id": {"sfbay": "1"}},
        ),
    )
    usim_data = tmp_path / "usim_data"
    usim_data.mkdir()
    base_h5 = usim_data / "usim_base_1.h5"
    base_h5.write_text("base", encoding="utf-8")
    ds_2019 = usim_data / "model_data_2019.h5"
    ds_2019.write_text("data", encoding="utf-8")
    workspace = SimpleNamespace(
        get_asim_mutable_configs_dir=lambda: str(tmp_path / "configs"),
        get_usim_mutable_data_dir=lambda: str(usim_data),
    )

    required = binding._restart_activitysim_required_artifacts(
        settings=settings,
        state=state,
        workspace=workspace,
        required_asim_config_dirs_fn=lambda _: ("configs",),
        workflow_stage=stage,
    )

    assert required is not None
    assert USIM_DATASTORE_CURRENT_H5 in required
    assert "2019" in required[USIM_DATASTORE_CURRENT_H5]
