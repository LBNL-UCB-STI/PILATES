from pathlib import Path
from types import SimpleNamespace

import pytest

from consist import BindingResult, resolve_step_contract

from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    ZARR_SKIMS,
)
from pilates.workflows.steps.beam import (
    _native_beam_postprocess,
    beam_full_skim,
    beam_postprocess,
    beam_preprocess,
    beam_run,
)
from pilates.workflows.resolved_inputs import ResolvedStepInputs


def _empty_resolved_inputs() -> ResolvedStepInputs:
    return ResolvedStepInputs(
        step_name="test",
        binding=BindingResult(inputs={}),
    )


def _write_event_types(path: Path, event_types: list[str]) -> None:
    import pandas as pd

    pd.DataFrame({"type": event_types}).to_parquet(path, index=False)


def test_native_beam_definitions_resolve_consist_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    definitions = (
        beam_preprocess,
        beam_run,
        beam_postprocess,
        beam_full_skim,
    )

    mutable_data_dir = tmp_path / "beam-input"
    output_dir = tmp_path / "beam-output"
    mutable_data_dir.mkdir()
    output_dir.mkdir()
    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_beam_mutable_data_dir=lambda: str(mutable_data_dir),
        get_beam_output_dir=lambda: str(output_dir),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    settings = SimpleNamespace(
        activitysim=None,
        beam=SimpleNamespace(config="beam.conf", full_skim=None),
        run=SimpleNamespace(
            region="test-region",
            models=SimpleNamespace(land_use=None),
        ),
        write_skims_to_omx=False,
    )
    state = SimpleNamespace(
        year=2030, current_year=2030, iteration=2, current_inner_iter=2
    )
    monkeypatch.setenv("PILATES_DISABLE_BEAM_CONFIG_ADAPTER", "1")
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda *_args, **_kwargs: {},
    )

    contracts = {
        definition.name: resolve_step_contract(
            definition.function,
            year=2030,
            iteration=2,
            phase="run",
            stage="traffic_assignment",
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )
        for definition in definitions
    }

    assert set(contracts) == {
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "beam_full_skim",
    }
    assert {name: contract.model for name, contract in contracts.items()} == {
        name: name for name in contracts
    }
    assert {name: contract.name for name, contract in contracts.items()} == {
        "beam_preprocess": "beam_preprocess__y2030__i2__phase_run",
        "beam_run": "beam_run__y2030__i2__phase_run",
        "beam_postprocess": "beam_postprocess__y2030__i2__phase_run",
        "beam_full_skim": "beam_full_skim__y2030__i2__phase_run",
    }
    assert all(contract.input_binding == "paths" for contract in contracts.values())
    assert contracts["beam_preprocess"].output_paths
    assert contracts["beam_run"].output_paths
    assert contracts["beam_full_skim"].output_paths


def test_beam_postprocess_resolver_stages_dynamic_closure_at_exact_destinations(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A", "PathTraversal"])

    class Coupler:
        def __init__(self) -> None:
            self.values = {
                "events_parquet_2030_2": events_path,
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
                ZARR_SKIMS: tmp_path / "source-skims.zarr",
            }

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

        def keys(self):
            return self.values.keys()

    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    state = SimpleNamespace(year=2030, iteration=2)
    resolved = beam_postprocess.resolve_inputs(
        settings=SimpleNamespace(activitysim=object()),
        state=state,
        workspace=workspace,
        coupler=Coupler(),
    )
    options = beam_postprocess.execution_options(
        settings=object(),
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )

    assert set(resolved.binding.inputs or {}) == {
        "events_parquet_2030_2",
        "raw_od_skims_2030_2",
        ZARR_SKIMS,
    }
    assert options.input_materialization == "requested"
    assert options.input_paths == resolved.logical_destinations
    assert options.runtime_kwargs["beam_run_dynamic_paths"] == {
        "events_parquet_2030_2": resolved.logical_destinations["events_parquet_2030_2"],
        "raw_od_skims_2030_2": resolved.logical_destinations["raw_od_skims_2030_2"],
    }
    assert (
        resolved.metadata["beam_postprocess_dynamic_paths"]
        == (options.runtime_kwargs["beam_run_dynamic_paths"])
    )
    assert resolved.logical_destinations["events_parquet_2030_2"] == (
        tmp_path
        / "beam-output"
        / ".pilates-consist-inputs"
        / "events_parquet_2030_2.parquet"
    )
    assert resolved.logical_destinations["raw_od_skims_2030_2"] == (
        tmp_path / "beam-output" / ".pilates-consist-inputs" / "raw_od_skims_2030_2.omx"
    )
    assert resolved.logical_destinations[ZARR_SKIMS] == (
        tmp_path / "asim-output" / "cache" / "skims.zarr"
    )
    assert resolved.metadata["beam_postprocess_output_paths"] == {
        "events_parquet_2030_2_type_Event_A": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "events_parquet_2030_2_type_Event_A.parquet"
        ),
        "events_parquet_2030_2_type_PathTraversal": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "events_parquet_2030_2_type_PathTraversal.parquet"
        ),
        "path_traversal_links_2030_2": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "path_traversal_links_2030_2.parquet"
        ),
    }


def test_beam_postprocess_projector_has_fresh_hit_parity_for_closed_split_outputs(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A", "PathTraversal"])
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )
    settings = SimpleNamespace(
        activitysim=None,
        run=SimpleNamespace(models=SimpleNamespace(land_use=None)),
        write_skims_to_omx=False,
    )
    state = SimpleNamespace(year=2030, iteration=2)

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            if key == "events_parquet_2030_2":
                return events_path
            if key == "raw_od_skims_2030_2":
                return tmp_path / "source-skims.omx"
            return default

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

    resolved = beam_postprocess.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=Coupler(),
    )
    declared = beam_postprocess.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    dynamic_paths = resolved.metadata["beam_postprocess_output_paths"]
    assert set(dynamic_paths).issubset(declared)
    for path in dynamic_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    fresh = beam_postprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=str(path))
            for key, path in dynamic_paths.items()
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    hit = beam_postprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{path.name}")
            for key, path in dynamic_paths.items()
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )

    assert (
        fresh.split_events
        == hit.split_events
        == {
            key: path
            for key, path in dynamic_paths.items()
            if key.startswith("events_parquet_")
        }
    )
    assert (
        fresh.split_event_links
        == hit.split_event_links
        == {
            key: path
            for key, path in dynamic_paths.items()
            if key.startswith("path_traversal_links_")
        }
    )


def test_beam_postprocess_resolver_refuses_uninspectable_selected_events(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {
                "events_parquet_2030_2": tmp_path / "not-present.parquet",
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
            }.get(key, default)

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

    with pytest.raises(
        RuntimeError,
        match="cannot inspect selected events input 'events_parquet_2030_2'",
    ):
        beam_postprocess.resolve_inputs(
            settings=SimpleNamespace(activitysim=None),
            state=SimpleNamespace(year=2030, iteration=2),
            workspace=workspace,
            coupler=Coupler(),
        )


def test_beam_postprocess_resolver_rejects_sanitized_event_key_collision(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    _write_event_types(events_path, ["Event A", "Event-A"])
    workspace = SimpleNamespace(
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
        get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
    )

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            return {
                "events_parquet_2030_2": events_path,
                "raw_od_skims_2030_2": tmp_path / "source-skims.omx",
            }.get(key, default)

        def keys(self):
            return ("events_parquet_2030_2", "raw_od_skims_2030_2")

    with pytest.raises(
        RuntimeError, match="do not map to injective semantic output keys"
    ):
        beam_postprocess.resolve_inputs(
            settings=SimpleNamespace(activitysim=None),
            state=SimpleNamespace(year=2030, iteration=2),
            workspace=workspace,
            coupler=Coupler(),
        )


def test_native_beam_postprocess_promotes_every_closed_typed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pilates.beam.outputs import BeamPostprocessOutputs

    source_events = tmp_path / "source-events.parquet"
    source_links = tmp_path / "source-links.parquet"
    source_events.write_text("events", encoding="utf-8")
    source_links.write_text("links", encoding="utf-8")
    output_paths = {
        "events_parquet_2030_2_type_PathTraversal": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "events_parquet_2030_2_type_PathTraversal.parquet"
        ),
        "path_traversal_links_2030_2": (
            tmp_path
            / "beam-output"
            / ".pilates-consist-outputs"
            / "beam_postprocess"
            / "path_traversal_links_2030_2.parquet"
        ),
    }

    class FakePostprocessor:
        def __init__(self, *_args: object) -> None:
            pass

        @staticmethod
        def expected_outputs(*_args: object) -> dict[str, str]:
            return {}

        def postprocess(
            self, *_args: object, **_kwargs: object
        ) -> BeamPostprocessOutputs:
            return BeamPostprocessOutputs(
                split_events={
                    "events_parquet_2030_2_type_PathTraversal": source_events
                },
                split_event_links={"path_traversal_links_2030_2": source_links},
            )

    monkeypatch.setattr(
        "pilates.beam.postprocessor.BeamPostprocessor", FakePostprocessor
    )
    settings = SimpleNamespace(
        activitysim=None,
        run=SimpleNamespace(models=SimpleNamespace(land_use=None)),
        write_skims_to_omx=False,
    )
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
    )

    _native_beam_postprocess(
        beam_run_dynamic_paths={},
        beam_postprocess_output_paths=output_paths,
        settings=settings,
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=workspace,
        _consist_ctx=object(),
    )

    assert {
        key: path.read_text(encoding="utf-8") for key, path in output_paths.items()
    } == {
        "events_parquet_2030_2_type_PathTraversal": "events",
        "path_traversal_links_2030_2": "links",
    }


def test_beam_preprocess_projector_is_pure_and_validates_persisted_outputs(
    tmp_path: Path,
) -> None:
    mutable_data_dir = tmp_path / "beam-input"
    mutable_data_dir.mkdir()
    workspace = SimpleNamespace(get_beam_mutable_data_dir=lambda: str(mutable_data_dir))
    declared = beam_preprocess.output_paths(
        settings=object(), state=object(), workspace=workspace, resolved_inputs=None
    )
    paths = {
        key: declared[key]
        for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    projected = beam_preprocess.project_outputs(
        {
            key: SimpleNamespace(
                container_uri=f"archive://historical/{path.name}",
                path=tmp_path / "historical" / path.name,
            )
            for key, path in paths.items()
        },
        settings=object(),
        state=object(),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )

    assert projected.prepared_inputs == paths


def test_beam_preprocess_projector_has_fresh_hit_path_parity(
    tmp_path: Path,
) -> None:
    mutable_data_dir = tmp_path / "beam-input"
    mutable_data_dir.mkdir()
    workspace = SimpleNamespace(get_beam_mutable_data_dir=lambda: str(mutable_data_dir))
    declared = beam_preprocess.output_paths(
        settings=object(), state=object(), workspace=workspace, resolved_inputs=None
    )
    paths = {
        key: declared[key]
        for key in (BEAM_PLANS_IN, BEAM_HOUSEHOLDS_IN, BEAM_PERSONS_IN)
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    fresh = beam_preprocess.project_outputs(
        {key: SimpleNamespace(container_uri=str(path)) for key, path in paths.items()},
        settings=object(),
        state=object(),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )
    hit = beam_preprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{path.name}")
            for key, path in paths.items()
        },
        settings=object(),
        state=object(),
        workspace=workspace,
        resolved_inputs=_empty_resolved_inputs(),
    )

    assert fresh.prepared_inputs == hit.prepared_inputs == paths


def test_beam_run_declares_closed_individually_keyed_handoff_outputs(
    tmp_path: Path,
) -> None:
    workspace = SimpleNamespace(
        get_beam_output_dir=lambda: str(tmp_path / "beam-output")
    )
    paths = beam_run.output_paths(
        settings=object(),
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=workspace,
        resolved_inputs=None,
    )

    assert "beam_output_dir" not in paths
    assert set(paths) == {
        "linkstats",
        "beam_plans_out",
        "raw_od_skims_2030_2",
        "raw_od_skims_zarr_2030_2",
        "events_parquet_2030_2",
    }
    assert len(set(paths.values())) == len(paths)
    assert all(
        ".pilates-consist-outputs/beam_run" in str(path) for path in paths.values()
    )


def test_beam_postprocess_dynamic_closure_destinations_do_not_collide(
    tmp_path: Path,
) -> None:
    selected_events = tmp_path / "source-events_parquet_2030_2.parquet"
    _write_event_types(selected_events, ["PathTraversal"])

    class Coupler:
        def get(self, key: str, default: object = None) -> object:
            if key == "events_parquet_2030_2":
                return selected_events
            return tmp_path / f"source-{key}"

        def keys(self):
            return (
                "events_parquet_2030_2",
                "events_parquet_2030_2_retry",
                "raw_od_skims_2030_2",
            )

    resolved = beam_postprocess.resolve_inputs(
        settings=SimpleNamespace(activitysim=None),
        state=SimpleNamespace(year=2030, iteration=2),
        workspace=SimpleNamespace(
            get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
            get_beam_output_dir=lambda: str(tmp_path / "beam-output"),
            get_asim_output_dir=lambda: str(tmp_path / "asim-output"),
        ),
        coupler=Coupler(),
    )
    closure = resolved.metadata["beam_postprocess_dynamic_paths"]

    assert set(closure) == {
        "events_parquet_2030_2",
        "events_parquet_2030_2_retry",
        "raw_od_skims_2030_2",
    }
    assert len(set(closure.values())) == len(closure)
    assert all(key in str(path) for key, path in closure.items())


def test_beam_preprocess_projector_rejects_stale_mount_when_current_destination_missing(
    tmp_path: Path,
) -> None:
    source_mount = tmp_path / "source-mount" / "plans.csv"
    source_mount.parent.mkdir()
    source_mount.write_text("historical plans", encoding="utf-8")
    current_dir = tmp_path / "current"
    current_households = current_dir / "households.csv"
    current_persons = current_dir / "persons.csv"
    current_dir.mkdir()
    (tmp_path / "beam-input").mkdir()
    current_households.write_text("current households", encoding="utf-8")
    current_persons.write_text("current persons", encoding="utf-8")

    workspace = SimpleNamespace(
        full_path=str(tmp_path),
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam-input"),
    )
    outputs = {
        BEAM_PLANS_IN: SimpleNamespace(
            container_uri="workspace://current/plans.csv",
            path=source_mount,
        ),
        BEAM_HOUSEHOLDS_IN: SimpleNamespace(
            container_uri="workspace://current/households.csv",
            path=current_households,
        ),
        BEAM_PERSONS_IN: SimpleNamespace(
            container_uri="workspace://current/persons.csv",
            path=current_persons,
        ),
    }

    with pytest.raises(
        RuntimeError,
        match="beam_preprocess output 'plans_beam_in' is missing at declared destination",
    ):
        beam_preprocess.project_outputs(
            outputs,
            settings=object(),
            state=object(),
            workspace=workspace,
            resolved_inputs=_empty_resolved_inputs(),
        )
