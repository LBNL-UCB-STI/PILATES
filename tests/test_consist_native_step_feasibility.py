from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from consist import BindingResult, define_step
from consist.core.tracker import Tracker
from consist.types import CacheOptions, ExecutionOptions, OutputArtifactSpec

from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    FINAL_SKIMS_OMX,
)
from pilates.workflows.steps import StepOutputsHolder, make_activitysim_preprocess_step


def _tracker(tmp_path: Path) -> Tracker:
    return Tracker(
        run_dir=tmp_path / "consist-runs",
        db_path=str(tmp_path / "provenance.duckdb"),
    )


def test_named_path_bound_input_consumes_selected_bytes_and_publishes_output(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path)
    source = tmp_path / "selected-source.txt"
    source.write_text("selected bytes\n", encoding="utf-8")
    seen: list[bytes] = []

    @define_step(
        model="native_binding_probe",
        inputs={"source": None},
        output_paths={"copied": "copied.txt"},
        input_binding="paths",
    )
    def copy_source(source: Path, ctx) -> None:
        selected_bytes = source.read_bytes()
        seen.append(selected_bytes)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "copied.txt").write_bytes(selected_bytes)

    with tracker.scenario("named-input") as scenario:
        result = scenario.run(
            fn=copy_source,
            binding=BindingResult(inputs={"source": source}),
            execution_options=ExecutionOptions(
                input_binding="paths",
                inject_context="ctx",
            ),
        )

    assert seen == [b"selected bytes\n"]
    assert result.outputs["copied"].path.read_bytes() == b"selected bytes\n"


def test_absent_optional_input_is_neither_bound_nor_staged(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    source_destination = tracker.run_dir / "workspace" / "source.txt"
    absent_destination = tracker.run_dir / "workspace" / "warmstart.txt"
    seen: list[Path] = []

    @define_step(
        model="optional_binding_probe",
        inputs={"source": None},
        optional_input_keys=["warmstart"],
        input_binding="paths",
    )
    def consume_source(source: Path) -> None:
        seen.append(source)
        assert source.read_text(encoding="utf-8") == "source\n"

    with tracker.scenario("absent-optional") as scenario:
        result = scenario.run(
            fn=consume_source,
            binding=BindingResult(
                inputs={"source": source},
                optional_input_keys=["warmstart"],
            ),
            execution_options=ExecutionOptions(
                input_binding="paths",
                input_materialization="requested",
                input_paths={
                    "source": source_destination,
                },
            ),
        )

    assert result.cache_hit is False
    assert seen == [source_destination.resolve()]
    assert source_destination.read_text(encoding="utf-8") == "source\n"
    assert not absent_destination.exists()
    assert "warmstart" not in result.run.meta.get("staged_inputs", {})


def test_equivalent_execution_cache_hit_returns_complete_identical_outputs(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    calls: list[str] = []

    @define_step(
        model="cache_output_probe",
        inputs={"source": None},
        output_paths={"copied": "copied.txt", "summary": "summary.txt"},
        input_binding="paths",
    )
    def produce_outputs(source: Path, ctx) -> None:
        calls.append("executed")
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "copied.txt").write_bytes(source.read_bytes())
        (ctx.run_dir / "summary.txt").write_text("complete\n", encoding="utf-8")

    results = []
    for name in ("cache-output-first", "cache-output-second"):
        with tracker.scenario(name) as scenario:
            results.append(
                scenario.run(
                    fn=produce_outputs,
                    binding=BindingResult(inputs={"source": source}),
                    execution_options=ExecutionOptions(
                        input_binding="paths",
                        inject_context="ctx",
                    ),
                )
            )

    first, second = results
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == ["executed"]
    assert set(first.outputs) == {"copied", "summary"}
    assert set(second.outputs) == set(first.outputs)
    assert {key: artifact.hash for key, artifact in second.outputs.items()} == {
        key: artifact.hash for key, artifact in first.outputs.items()
    }


def test_dynamic_manual_outputs_are_available_directly_from_run_result(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path)
    output_paths = {
        "beam_linkstats": "linkstats.csv.gz",
        "beam_skims": "skims.omx",
    }

    @define_step(
        model="dynamic_output_probe",
        output_paths=output_paths,
    )
    def publish_dynamic_outputs(ctx) -> None:
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / output_paths["beam_linkstats"]).write_text(
            "linkstats\n", encoding="utf-8"
        )
        (ctx.run_dir / output_paths["beam_skims"]).write_bytes(b"omx bytes")

    with tracker.scenario("dynamic-outputs") as scenario:
        result = scenario.run(
            fn=publish_dynamic_outputs,
            execution_options=ExecutionOptions(inject_context="ctx"),
        )

    projected = {key: result.outputs[key].path for key in output_paths}
    assert set(projected) == set(output_paths)
    assert projected["beam_linkstats"].read_text(encoding="utf-8") == "linkstats\n"
    assert projected["beam_skims"].read_bytes() == b"omx bytes"


def test_dynamic_unmatched_inputs_stage_lineage_and_rehydrate_on_cache_hit(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path)
    sources = {
        "raw_od_skims_2035_1": tmp_path / "upstream" / "od_skims.omx",
        "events_parquet_2035_1": tmp_path / "upstream" / "events.parquet",
    }
    for key, path in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{key}\n", encoding="utf-8")
    destinations = {
        key: tracker.run_dir / "workspace" / "beam-postprocess" / path.name
        for key, path in sources.items()
    }
    observed: list[dict[str, bytes]] = []

    @define_step(model="dynamic_input_probe")
    def consume_materialized_dynamic_inputs(ctx) -> None:
        observed.append(
            {key: destination.read_bytes() for key, destination in destinations.items()}
        )
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "result.txt").write_text("complete\n", encoding="utf-8")

    options = ExecutionOptions(
        input_binding="paths",
        input_materialization="requested",
        input_paths=destinations,
        inject_context="ctx",
    )
    with tracker.scenario("dynamic-input-first") as scenario:
        first = scenario.run(
            fn=consume_materialized_dynamic_inputs,
            binding=BindingResult(inputs=sources),
            output_paths={"result": "result.txt"},
            execution_options=options,
        )

    first_inputs = tracker.get_artifacts_for_run(first.run.id).inputs
    assert set(first_inputs) == set(sources)
    assert observed == [{key: path.read_bytes() for key, path in sources.items()}]
    assert {key: path.read_bytes() for key, path in destinations.items()} == {
        key: path.read_bytes() for key, path in sources.items()
    }

    for destination in destinations.values():
        destination.unlink()

    with tracker.scenario("dynamic-input-second") as scenario:
        second = scenario.run(
            fn=consume_materialized_dynamic_inputs,
            binding=BindingResult(inputs=sources),
            output_paths={"result": "result.txt"},
            execution_options=options,
        )

    assert second.cache_hit is True
    assert len(observed) == 1
    assert {key: path.read_bytes() for key, path in destinations.items()} == {
        key: path.read_bytes() for key, path in sources.items()
    }
    assert set(tracker.get_artifacts_for_run(second.run.id).inputs) == set(sources)


def test_activitysim_preprocess_file_outputs_rehydrate_and_retain_omx_lineage(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path)
    workspace_three_first = (
        tracker.run_dir / "workspace-three-first" / "activitysim" / "data"
    )
    workspace_three_second = (
        tracker.run_dir / "workspace-three-second" / "activitysim" / "data"
    )
    workspace_four_first = (
        tracker.run_dir / "workspace-four-first" / "activitysim" / "data"
    )
    workspace_four_second = (
        tracker.run_dir / "workspace-four-second" / "activitysim" / "data"
    )
    workspace_four_miss = (
        tracker.run_dir / "workspace-four-miss" / "activitysim" / "data"
    )
    workspace_four_recovered = (
        tracker.run_dir / "workspace-four-recovered" / "activitysim" / "data"
    )
    upstream_omx = tmp_path / "upstream" / "skims.omx"
    upstream_omx.parent.mkdir(parents=True, exist_ok=True)
    upstream_omx.write_bytes(b"upstream omx bytes")

    class Coupler:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

        def set(self, key: str, value: object) -> None:
            self.values[key] = value

    actual_step = make_activitysim_preprocess_step(
        coupler=Coupler(),
        outputs_holder=StepOutputsHolder(),
    )
    actual_meta = actual_step.__consist_step__
    assert callable(actual_meta.output_paths)

    expected_keys = {
        ASIM_LAND_USE_IN,
        ASIM_HOUSEHOLDS_IN,
        ASIM_PERSONS_IN,
        ASIM_OMX_SKIMS,
    }
    actual_outputs = actual_meta.output_paths(
        settings=object(),
        state=object(),
        workspace=SimpleNamespace(
            get_asim_mutable_data_dir=lambda: str(workspace_three_first),
            get_asim_mutable_configs_dir=lambda: str(tmp_path / "configs"),
        ),
    )
    assert set(actual_outputs) == expected_keys
    assert all(
        isinstance(output, OutputArtifactSpec) for output in actual_outputs.values()
    )
    assert {output.path for output in actual_outputs.values()} == {
        str(workspace_three_first / "land_use.csv"),
        str(workspace_three_first / "households.csv"),
        str(workspace_three_first / "persons.csv"),
        str(workspace_three_first / "skims.omx"),
    }
    assert str(workspace_three_first) not in {
        output.path for output in actual_outputs.values()
    }
    assert str(tmp_path / "configs") not in {
        output.path for output in actual_outputs.values()
    }

    calls: list[str] = []

    @define_step(model="activitysim_preprocess_file_contract", input_binding="paths")
    def preprocess(final_skims_omx: Path, *, workspace: Path) -> None:
        calls.append("executed")
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "land_use.csv").write_text("land_use\n", encoding="utf-8")
        (workspace / "households.csv").write_text("households\n", encoding="utf-8")
        (workspace / "persons.csv").write_text("persons\n", encoding="utf-8")
        assert final_skims_omx == workspace / "skims.omx"
        assert final_skims_omx.read_bytes() == b"upstream omx bytes"

    def run_preprocess(
        workspace: Path,
        scenario_name: str,
        *,
        include_omx_output: bool,
        cache_version: int,
    ):
        output_paths: dict[str, Path] = {
            ASIM_LAND_USE_IN: workspace / "land_use.csv",
            ASIM_HOUSEHOLDS_IN: workspace / "households.csv",
            ASIM_PERSONS_IN: workspace / "persons.csv",
        }
        if include_omx_output:
            output_paths[ASIM_OMX_SKIMS] = workspace / "skims.omx"
        with tracker.scenario(scenario_name) as scenario:
            return scenario.run(
                fn=preprocess,
                binding=BindingResult(inputs={FINAL_SKIMS_OMX: upstream_omx}),
                output_paths=output_paths,
                cache_options=CacheOptions(
                    cache_hydration="outputs-requested",
                    cache_hydration_failure="miss",
                    cache_version=cache_version,
                ),
                execution_options=ExecutionOptions(
                    input_binding="paths",
                    input_materialization="requested",
                    input_paths={FINAL_SKIMS_OMX: workspace / "skims.omx"},
                    runtime_kwargs={"workspace": workspace},
                ),
            )

    expected_three_file_keys = expected_keys - {ASIM_OMX_SKIMS}
    three_first = run_preprocess(
        workspace_three_first,
        "activitysim-preprocess-three-first",
        include_omx_output=False,
        cache_version=3,
    )
    assert three_first.cache_hit is False
    assert set(three_first.outputs) == expected_three_file_keys
    assert all(path.is_file() for path in workspace_three_first.iterdir())
    assert not any(
        artifact.path == workspace_three_first
        for artifact in three_first.outputs.values()
    )
    three_first_inputs = tracker.get_artifacts_for_run(three_first.run.id).inputs
    assert set(three_first_inputs) == {FINAL_SKIMS_OMX}
    assert three_first_inputs[
        FINAL_SKIMS_OMX
    ].hash == tracker.identity.compute_file_checksum(upstream_omx)

    three_second = run_preprocess(
        workspace_three_second,
        "activitysim-preprocess-three-second",
        include_omx_output=False,
        cache_version=3,
    )
    assert three_second.cache_hit is True
    assert calls == ["executed"]
    assert set(three_second.outputs) == expected_three_file_keys
    assert {
        path.name: path.read_bytes() for path in workspace_three_second.iterdir()
    } == {
        "land_use.csv": b"land_use\n",
        "households.csv": b"households\n",
        "persons.csv": b"persons\n",
        "skims.omx": b"upstream omx bytes",
    }
    three_second_inputs = tracker.get_artifacts_for_run(three_second.run.id).inputs
    assert set(three_second_inputs) == {FINAL_SKIMS_OMX}
    assert three_second_inputs[
        FINAL_SKIMS_OMX
    ].hash == tracker.identity.compute_file_checksum(upstream_omx)

    four_first = run_preprocess(
        workspace_four_first,
        "activitysim-preprocess-four-first",
        include_omx_output=True,
        cache_version=4,
    )
    assert four_first.cache_hit is False
    assert calls == ["executed", "executed"]
    assert set(four_first.outputs) == expected_keys
    assert all(path.is_file() for path in workspace_four_first.iterdir())
    four_first_inputs = tracker.get_artifacts_for_run(four_first.run.id).inputs
    assert set(four_first_inputs) == {FINAL_SKIMS_OMX}
    assert four_first_inputs[
        FINAL_SKIMS_OMX
    ].hash == tracker.identity.compute_file_checksum(upstream_omx)

    four_second = run_preprocess(
        workspace_four_second,
        "activitysim-preprocess-four-second",
        include_omx_output=True,
        cache_version=4,
    )
    assert four_second.cache_hit is True
    assert calls == ["executed", "executed"]
    assert set(four_second.outputs) == expected_keys
    assert {
        path.name: path.read_bytes() for path in workspace_four_second.iterdir()
    } == {
        "land_use.csv": b"land_use\n",
        "households.csv": b"households\n",
        "persons.csv": b"persons\n",
        "skims.omx": b"upstream omx bytes",
    }
    assert {key: artifact.hash for key, artifact in four_second.outputs.items()} == {
        key: artifact.hash for key, artifact in four_first.outputs.items()
    }
    four_second_inputs = tracker.get_artifacts_for_run(four_second.run.id).inputs
    assert set(four_second_inputs) == {FINAL_SKIMS_OMX}
    assert four_second_inputs[
        FINAL_SKIMS_OMX
    ].hash == tracker.identity.compute_file_checksum(upstream_omx)

    (workspace_four_first / "households.csv").unlink()
    four_miss = run_preprocess(
        workspace_four_miss,
        "activitysim-preprocess-four-miss",
        include_omx_output=True,
        cache_version=4,
    )
    assert four_miss.cache_hit is False
    assert calls == ["executed", "executed", "executed"]
    assert set(four_miss.outputs) == expected_keys
    assert {path.name: path.read_bytes() for path in workspace_four_miss.iterdir()} == {
        "land_use.csv": b"land_use\n",
        "households.csv": b"households\n",
        "persons.csv": b"persons\n",
        "skims.omx": b"upstream omx bytes",
    }
    four_miss_inputs = tracker.get_artifacts_for_run(four_miss.run.id).inputs
    assert set(four_miss_inputs) == {FINAL_SKIMS_OMX}
    assert four_miss_inputs[
        FINAL_SKIMS_OMX
    ].hash == tracker.identity.compute_file_checksum(upstream_omx)

    (workspace_four_first / "households.csv").write_text(
        "households\n", encoding="utf-8"
    )
    four_recovered = run_preprocess(
        workspace_four_recovered,
        "activitysim-preprocess-four-recovered",
        include_omx_output=True,
        cache_version=4,
    )
    assert four_recovered.cache_hit is True
    assert calls == ["executed", "executed", "executed"]
    assert set(four_recovered.outputs) == expected_keys
    assert {key: artifact.hash for key, artifact in four_recovered.outputs.items()} == {
        key: artifact.hash for key, artifact in four_first.outputs.items()
    }
    four_recovered_inputs = tracker.get_artifacts_for_run(four_recovered.run.id).inputs
    assert set(four_recovered_inputs) == {FINAL_SKIMS_OMX}
    assert four_recovered_inputs[
        FINAL_SKIMS_OMX
    ].hash == tracker.identity.compute_file_checksum(upstream_omx)
