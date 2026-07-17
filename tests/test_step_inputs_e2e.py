import hashlib
from pathlib import Path
from types import SimpleNamespace

import consist
import pytest
from consist import Tracker
from consist.core.directory_artifacts import build_directory_manifest
from consist.types import ExecutionOptions

from pilates.activitysim.inputs import build_activitysim_inputs
from pilates.utils import consist_runtime as cr
from pilates.workflows.binding import BindingPlan
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_PERSONS_IN,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.beam_checkpoint import (
    PinnedClosureMember,
    hydrate_pinned_closure,
)


class DummyWorkspace:
    def __init__(self, root):
        self.full_path = str(root)
        self._root = root

    def get_asim_mutable_data_dir(self):
        return str(self._root / "activitysim" / "data")

    def get_usim_mutable_data_dir(self):
        return str(self._root / "urbansim" / "data")

    def get_asim_output_dir(self):
        return str(self._root / "activitysim" / "output")


def _surface(*, land_use_enabled: bool = False):
    return SimpleNamespace(
        profile=SimpleNamespace(land_use_enabled=land_use_enabled),
        step_surface=lambda _name: None,
    )


def _content_identity(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        return build_directory_manifest(path)["tree_hash"]
    raise AssertionError(f"Consist input has no consumable bytes: {path}")


def _input_content_identities(tracker: Tracker, run_id: str) -> dict[str, str]:
    artifacts = tracker.get_artifacts_for_run(run_id)
    identities: dict[str, str] = {}
    for artifact in sorted(artifacts.inputs.values(), key=lambda artifact: artifact.key):
        content_identity = _content_identity(artifact.as_path(tracker=tracker))
        assert content_identity == artifact.hash, (
            f"{artifact.key} bytes do not match Consist-selected artifact identity"
        )
        identities[artifact.key] = content_identity
    return identities


def _identity_lock_results(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, str]]:
    tracker = Tracker(
        run_dir=tmp_path / "consist_runs",
        db_path=str(tmp_path / "identity.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )
    plans = tmp_path / "beam" / "plans.parquet"
    events = tmp_path / "beam" / "events.parquet"
    postprocessed = tmp_path / "beam" / "postprocessed.txt"
    restart_destination = tmp_path / "restart" / "events.parquet"
    restart_postprocessed = tmp_path / "restart" / "postprocessed.txt"
    plans.parent.mkdir(parents=True)
    plans.write_bytes(b"selected plans bytes\n")
    consumed: dict[str, str] = {}

    def beam_run(beam_plans: str) -> None:
        consumed["fresh_beam_run"] = _content_identity(Path(beam_plans))
        events.write_bytes(Path(beam_plans).read_bytes())

    def beam_postprocess(beam_events: str, *, label: str) -> None:
        consumed[label] = _content_identity(Path(beam_events))
        target = (
            restart_postprocessed
            if label == "committed_beam_restart"
            else postprocessed
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(beam_events).read_bytes())

    execution_options = ExecutionOptions(input_binding="paths")
    with cr.scenario("identity-lock", tracker=tracker) as scenario:
        fresh_run = scenario.run(
            fn=beam_run,
            name="beam_run",
            model="beam_run",
            inputs={"beam_plans": str(plans)},
            output_paths={"beam_events": str(events)},
            execution_options=execution_options,
        )
        fresh_postprocess = scenario.run(
            fn=beam_postprocess,
            name="beam_postprocess",
            model="beam_postprocess",
            inputs={"beam_events": consist.ref(fresh_run, key="beam_events")},
            output_paths={"postprocessed": str(postprocessed)},
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={"label": "fresh_beam_postprocess"},
            ),
        )
        cache_hit = scenario.run(
            fn=beam_postprocess,
            name="beam_postprocess",
            model="beam_postprocess",
            inputs={"beam_events": consist.ref(fresh_run, key="beam_events")},
            output_paths={"postprocessed": str(postprocessed)},
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={"label": "cache_hit_must_not_execute"},
            ),
        )

        beam_events = tracker.get_artifacts_for_run(fresh_run.run.id).outputs[
            "beam_events"
        ]
        member = PinnedClosureMember(
            member_id="beam-events",
            role="beam_events",
            producer_run_id=fresh_run.run.id,
            output_key="beam_events",
            artifact_identity=beam_events.hash,
            artifact_kind="file",
            driver=beam_events.driver,
            destination=restart_destination.resolve(),
            required=True,
        )

        class _PinnedTracker:
            def get_run(self, run_id: str):
                return tracker.get_run(run_id)

            def get_run_outputs(self, run_id: str):
                return tracker.get_artifacts_for_run(run_id).outputs

            def hydrate_run_outputs_to_destinations(self, run_id: str, **kwargs):
                destinations = kwargs["destinations_by_key"]
                outputs = self.get_run_outputs(run_id)
                hydrated = {}
                for key, destination in destinations.items():
                    artifact = outputs[key]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(
                        artifact.as_path(tracker=tracker).read_bytes()
                    )
                    hydrated[key] = SimpleNamespace(
                        path=destination,
                        status="materialized_from_filesystem",
                        artifact_kind="file",
                        resolvable=True,
                        artifact=artifact,
                    )
                result = dict(hydrated)
                result["source_run_id"] = run_id
                return SimpleNamespace(
                    source_run_id=run_id,
                    get=lambda key: hydrated.get(key),
                )

        hydrate_pinned_closure(
            tracker=_PinnedTracker(),
            source_root=tmp_path,
            members=(member,),
        )
        committed_restart = scenario.run(
            fn=beam_postprocess,
            name="beam_postprocess_committed_restart",
            model="beam_postprocess",
            config={"restart_boundary": "beam_run_completed"},
            inputs={"beam_events": str(restart_destination)},
            output_paths={"postprocessed": str(restart_postprocessed)},
            execution_options=ExecutionOptions(
                input_binding="paths",
                runtime_kwargs={"label": "committed_beam_restart"},
            ),
        )

    assert fresh_run.cache_hit is False
    assert fresh_postprocess.cache_hit is False
    assert cache_hit.cache_hit is True
    assert "cache_hit_must_not_execute" not in consumed
    identities = {
        "fresh": {
            "beam_run": _input_content_identities(tracker, fresh_run.run.id),
            "beam_postprocess": _input_content_identities(
                tracker, fresh_postprocess.run.id
            ),
        },
        "cache_hit": {
            "beam_postprocess": _input_content_identities(tracker, cache_hit.run.id),
        },
        "committed_beam_restart": {
            "beam_postprocess": _input_content_identities(
                tracker, committed_restart.run.id
            ),
        },
    }
    return identities, consumed


def test_build_activitysim_inputs_merges_coupler_and_usim(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    coupler = {"zarr_skims": "skims.zarr"}
    usim_inputs = {USIM_DATASTORE_CURRENT_H5: "/tmp/usim.h5"}

    inputs, descriptions = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
        surface=_surface(),
    )

    assert inputs[ASIM_HOUSEHOLDS_IN] == str(asim_dir / "households.csv")
    assert inputs[ASIM_PERSONS_IN] == str(asim_dir / "persons.csv")
    assert inputs[ASIM_LAND_USE_IN] == str(asim_dir / "land_use.csv")
    assert inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/usim.h5"
    assert inputs[ZARR_SKIMS] == "skims.zarr"
    assert ASIM_HOUSEHOLDS_IN in descriptions


def test_build_activitysim_inputs_uses_base_datastore_fallback(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    coupler = {}
    usim_inputs = {USIM_DATASTORE_BASE_H5: "/tmp/usim_base.h5"}

    inputs, _ = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
        surface=_surface(),
    )

    assert inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/usim_base.h5"


def test_build_activitysim_inputs_prefers_explicit_base_over_stale_coupler_current(
    tmp_path,
) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    coupler = {USIM_DATASTORE_CURRENT_H5: "/tmp/coupler_current.h5"}
    usim_inputs = {
        USIM_DATASTORE_CURRENT_H5: "/tmp/explicit_current.h5",
        USIM_DATASTORE_BASE_H5: "/tmp/explicit_base.h5",
    }

    inputs, _ = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=workspace,
        year=2018,
        iteration=0,
        coupler=coupler,
        usim_inputs=usim_inputs,
        surface=_surface(),
    )

    assert inputs[USIM_POPULATION_SOURCE_H5] == "/tmp/explicit_base.h5"


def test_build_activitysim_inputs_resolves_population_source_for_forecast_year(
    monkeypatch,
    tmp_path,
) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    population_h5 = tmp_path / "urbansim" / "data" / "model_data_2021.h5"
    population_h5.parent.mkdir(parents=True)
    population_h5.write_text("population")
    captured_years = []

    def _fake_build_binding_plan(**kwargs):
        captured_years.append(kwargs["year"])
        return BindingPlan(
            inputs={USIM_POPULATION_SOURCE_H5: str(population_h5)},
            source_by_key={USIM_POPULATION_SOURCE_H5: "fallback"},
            metadata={
                "selected_key_by_semantic_key": {
                    USIM_POPULATION_SOURCE_H5: USIM_POPULATION_SOURCE_H5
                }
            },
        )

    monkeypatch.setattr(
        "pilates.activitysim.inputs.build_binding_plan",
        _fake_build_binding_plan,
    )

    inputs, descriptions = build_activitysim_inputs(
        settings=SimpleNamespace(),
        state=SimpleNamespace(year=2019, forecast_year=2021),
        workspace=workspace,
        year=2019,
        iteration=0,
        coupler={},
        usim_inputs={},
        surface=_surface(land_use_enabled=True),
    )

    assert captured_years == [2021]
    assert inputs[USIM_POPULATION_SOURCE_H5] == str(population_h5)
    assert "population year 2021" in descriptions[USIM_POPULATION_SOURCE_H5]


def test_build_activitysim_inputs_requires_surface(tmp_path) -> None:
    workspace = DummyWorkspace(tmp_path)
    asim_dir = tmp_path / "activitysim" / "data"
    asim_dir.mkdir(parents=True)
    (asim_dir / "households.csv").write_text("")
    (asim_dir / "persons.csv").write_text("")
    (asim_dir / "land_use.csv").write_text("")

    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'surface'"
    ):
        build_activitysim_inputs(
            settings=SimpleNamespace(),
            state=SimpleNamespace(),
            workspace=workspace,
            year=2018,
            iteration=0,
            coupler={},
            usim_inputs={USIM_DATASTORE_BASE_H5: "/tmp/usim_base.h5"},
        )


def test_consist_identity_roles_match_the_bytes_consumed_and_restored(tmp_path) -> None:
    identities, consumed = _identity_lock_results(tmp_path)

    assert identities["fresh"]["beam_run"] == {"beam_plans": consumed["fresh_beam_run"]}
    assert identities["fresh"]["beam_postprocess"] == {
        "beam_events": consumed["fresh_beam_postprocess"]
    }
    assert (
        identities["cache_hit"]["beam_postprocess"]
        == identities["fresh"]["beam_postprocess"]
    )
    assert identities["committed_beam_restart"]["beam_postprocess"] == {
        "beam_events": consumed["committed_beam_restart"]
    }
    assert (
        identities["committed_beam_restart"]["beam_postprocess"]
        == identities["fresh"]["beam_postprocess"]
    )
