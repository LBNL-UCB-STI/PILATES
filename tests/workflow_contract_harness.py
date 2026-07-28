"""Shared production-like harness for workflow contract tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from pilates.config import load_config
from pilates.runtime.context import WorkflowRuntimeContext


class FakeTracker:
    """Minimal tracker stub for workflow contract tests."""

    def __init__(self, matching_run: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.matching_run = matching_run

    def find_matching_run(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.matching_run is None:
            return None
        if isinstance(self.matching_run, dict):
            key = (
                kwargs.get("model_name"),
                kwargs.get("year"),
                kwargs.get("iteration"),
            )
            if key in self.matching_run:
                val = self.matching_run[key]
                return SimpleNamespace(id=val) if isinstance(val, str) else val
            return None
        if isinstance(self.matching_run, str):
            return SimpleNamespace(id=self.matching_run)
        return self.matching_run

    def get_run_outputs(self, queried_run_id: str) -> dict[str, Any]:
        return {}

    def hydrate_run_outputs(self, **kwargs: Any) -> Any:
        return {}


class CouplerStub:
    """Minimal in-memory coupler for workflow contract tests."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def set_from_artifact(self, key: str, value: Any) -> None:
        self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._values.pop(key, default)

    def keys(self) -> list[str]:
        return list(self._values.keys())

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"Coupler missing key={key!r}")
        return self._values[key]


class FakeScenario:
    """
    Scenario stub that records calls and enforces production-like coupling.

    Explicit ``inputs`` stay as explicit call arguments. ``input_keys`` must
    already be present in the coupler before the step starts.
    """

    def __init__(self, coupler: CouplerStub, tracker: Any = None) -> None:
        self.coupler = coupler
        self.tracker = tracker
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> dict[str, str]:
        binding = kwargs.get("binding")
        if binding is not None:
            inputs = binding.inputs or {}
            input_keys = binding.input_keys or []
            optional_input_keys = binding.optional_input_keys or []
        else:
            inputs = kwargs.get("inputs") or {}
            input_keys = kwargs.get("input_keys") or []
            optional_input_keys = kwargs.get("optional_input_keys") or []
        fn = kwargs["fn"]
        model = kwargs.get("model")
        if model is None:
            step_meta = getattr(fn, "__consist_step__", None)
            model = getattr(step_meta, "model", None)
        self.calls.append(
            {
                "fn_name": getattr(fn, "__name__", "<unknown>"),
                "model": model,
                "inputs": dict(inputs),
                "input_keys": list(input_keys),
                "optional_input_keys": list(optional_input_keys),
                "binding": binding,
            }
        )

        for key in input_keys:
            self.coupler.require(key)

        execution_options = kwargs.get("execution_options")
        runtime_kwargs = kwargs.get("runtime_kwargs") or getattr(
            execution_options, "runtime_kwargs", None
        )
        fn_kwargs = dict(runtime_kwargs or {})
        sig = inspect.signature(fn)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if accepts_kwargs:
            fn_kwargs.update(inputs)
            for key in input_keys:
                fn_kwargs.setdefault(key, self.coupler.get(key))
        else:
            allowed = set(sig.parameters.keys())
            for key, value in inputs.items():
                if key in allowed:
                    fn_kwargs[key] = value
            for key in input_keys:
                if key in allowed:
                    fn_kwargs.setdefault(key, self.coupler.get(key))

        inject_context = getattr(execution_options, "inject_context", None)
        if inject_context:
            context_name = (
                inject_context if isinstance(inject_context, str) else "_consist_ctx"
            )
            if context_name not in fn_kwargs:
                if context_name in sig.parameters or accepts_kwargs:
                    fn_kwargs[context_name] = _test_run_context(
                        model=model,
                        fn_kwargs=fn_kwargs,
                    )
                else:
                    raise TypeError(
                        f"inject_context requested {context_name!r}, but fn does not accept it."
                    )

        fn(**fn_kwargs)
        return {"status": "ok"}


def _test_run_context(*, model: Any, fn_kwargs: dict[str, Any]) -> SimpleNamespace:
    """Build the minimal Consist context required by a contract-test step."""

    if model != "beam_run":
        return SimpleNamespace(canonicalization=None)

    state = fn_kwargs["state"]
    workspace = fn_kwargs["workspace"]
    from pilates.beam.launch_paths import resolve_r5_network_reference

    execution_reference = resolve_r5_network_reference(
        settings=state.full_settings,
        workspace=workspace,
    )
    artifact_key = (
        f"test:r5_osm_source:{execution_reference.selected_osm_physical_target_path}"
    )
    r5_reference = SimpleNamespace(
        reference=SimpleNamespace(config_key="beam.routing.r5.directory"),
        artifact_keys=(artifact_key,),
        artifact_members=(
            SimpleNamespace(
                role="r5_osm_source",
                resolved_path=execution_reference.selected_osm_physical_target_path,
                artifact_key=artifact_key,
            ),
        ),
    )
    return SimpleNamespace(canonicalization=SimpleNamespace(references=(r5_reference,)))


def build_runtime_context(
    *, settings: Any, state: Any, workspace: Any
) -> WorkflowRuntimeContext:
    """Build the explicit runtime context expected by stage entrypoints."""
    return WorkflowRuntimeContext.from_parts(
        settings=settings,
        state=state,
        workspace=workspace,
    )


def _invoke_record_builder(
    record_builder: Callable[..., Any],
    model_name: str,
    phase: str,
    **kwargs: Any,
) -> Any:
    """Call ``record_builder`` with only the keyword arguments it accepts."""
    sig = inspect.signature(record_builder)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    ):
        return record_builder(model_name, phase, **kwargs)

    accepted = set(sig.parameters.keys())
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return record_builder(model_name, phase, **filtered_kwargs)


def build_settings(tmp_path: Any) -> Any:
    """Build the compact workflow config used by the hybrid stub tests."""
    config = {
        "run": {
            "region": "test",
            "scenario": "test",
            "start_year": 2017,
            "end_year": 2018,
            "output_directory": str(tmp_path / "outputs"),
            "output_run_name": "golden_stub",
            "supply_demand_iters": 1,
            "models": {
                "land_use": "urbansim",
                "travel": "beam",
                "activity_demand": "activitysim",
                "vehicle_ownership": "atlas",
            },
        },
        "shared": {
            "geography": {
                "FIPS": {"county": ["00001"]},
                "local_crs": "EPSG:4326",
            },
            "skims": {"fname": "skims.omx"},
            "database": {
                "enabled": False,
                "type": "duckdb",
                "path": str(tmp_path / "db.duckdb"),
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        "urbansim": {
            "local_data_input_folder": str(tmp_path / "usim_input"),
            "local_mutable_data_folder": "urbansim/data",
            "client_base_folder": "/usim",
            "client_data_folder": "/usim/data",
            "input_file_template": "usim_{region_id}.h5",
            "input_file_template_year": "usim_{region_id}_{year}.h5",
            "output_file_template": "usim_{year}.h5",
            "command_template": "run_usim",
            "region_mappings": {"region_to_region_id": {"test": "000"}},
        },
        "atlas": {
            "host_input_folder": "atlas/input",
            "warmstart_input_folder": "atlas/warmstart",
            "host_mutable_input_folder": "atlas/atlas_input",
            "host_output_folder": "atlas/atlas_output",
            "container_input_folder": "/atlas/input",
            "container_output_folder": "/atlas/output",
            "basedir": "/atlas",
            "codedir": "/atlas/code",
            "command_template": "atlas {0}",
        },
        "activitysim": {
            "local_input_folder": "activitysim/input",
            "local_mutable_data_folder": "activitysim/data",
            "local_output_folder": "activitysim/output",
            "local_configs_folder": "activitysim/configs",
            "local_mutable_configs_folder": "activitysim/configs_mutable",
            "validation_folder": "activitysim/validation",
            "command_template": "asim run",
            "final_plans_folder": "activitysim/final_plans",
            "region_mappings": {"region_to_subdir": {"test": "test"}},
        },
        "beam": {
            "config": "beam.conf",
            "local_input_folder": "beam/input",
            "local_mutable_data_folder": "beam/input",
            "local_output_folder": "beam/output",
            "scenario_folder": "beam/scenario",
            "router_directory": "router",
            "skims_shapefile": "beam/skims.shp",
            "skim_zone_source_id_col": "id",
            "skim_zone_geoid_col": "geoid",
        },
    }
    config_path = tmp_path / "settings.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return load_config(str(config_path))


class DummyPreprocessor:
    """Deterministic preprocessor stub backed by ``record_builder``."""

    def __init__(
        self, model_name: str, record_builder: Callable[..., Any], state: Any = None
    ) -> None:
        self.model_name = model_name
        self._record_builder = record_builder
        self.state = state

    def preprocess(
        self,
        workspace: Any,
        previous_records: Any = None,
        activity_demand_outputs: Any = None,
        previous_beam_outputs: Any = None,
        beam_preprocess_inputs: Any = None,
        **kwargs: Any,
    ) -> Any:
        return _invoke_record_builder(
            self._record_builder,
            self.model_name,
            "preprocess",
            state=self.state,
            workspace=workspace,
        )


class DummyRunner:
    """Deterministic runner stub backed by ``record_builder``."""

    def __init__(
        self, model_name: str, record_builder: Callable[..., Any], state: Any = None
    ) -> None:
        self.model_name = model_name
        self._record_builder = record_builder
        self.state = state

    def run(
        self,
        input_store: Any,
        workspace: Any,
        extra_inputs: Any = None,
        previous_beam_outputs: Any = None,
        skim_mode: Any = None,
        skip_numba_warmup: bool = False,
    ) -> Any:
        del extra_inputs, previous_beam_outputs, skim_mode, skip_numba_warmup
        return _invoke_record_builder(
            self._record_builder,
            self.model_name,
            "run",
            state=self.state,
            workspace=workspace,
            input_store=input_store,
        )


class DummyPostprocessor:
    """Deterministic postprocessor stub backed by ``record_builder``."""

    def __init__(
        self, model_name: str, record_builder: Callable[..., Any], state: Any = None
    ) -> None:
        self.model_name = model_name
        self._record_builder = record_builder
        self._state = state

    def postprocess(
        self,
        raw_outputs: Any,
        workspace: Any,
        model_run_hash: Any = None,
        population_source_h5_path: Any = None,
        current_input_h5_path: Any = None,
    ) -> Any:
        return _invoke_record_builder(
            self._record_builder,
            self.model_name,
            "postprocess",
            state=self._state,
            workspace=workspace,
            raw_outputs=raw_outputs,
            population_source_h5_path=population_source_h5_path,
            current_input_h5_path=current_input_h5_path,
        )


def write_file(path: Any, content: str = "x") -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)


def write_csv(path: Any, df: pd.DataFrame) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def write_parquet(path: Any, df: pd.DataFrame) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def write_usim_toy_h5(path: Any, *, with_year_prefix: int | None = None) -> None:
    """Create a minimal UrbanSim-style HDF5 with the core tables used in tests."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    table_prefix = f"/{with_year_prefix}" if with_year_prefix is not None else ""
    households_key = f"{table_prefix}/households" if table_prefix else "households"
    blocks_key = f"{table_prefix}/blocks" if table_prefix else "blocks"
    persons_key = f"{table_prefix}/persons" if table_prefix else "persons"
    residential_key = (
        f"{table_prefix}/residential_units" if table_prefix else "residential_units"
    )
    jobs_key = f"{table_prefix}/jobs" if table_prefix else "jobs"
    graveyard_key = f"{table_prefix}/graveyard" if table_prefix else "graveyard"

    households = pd.DataFrame({"income": [100000.0, 70000.0]}, index=[1, 2])
    households.index.name = "household_id"

    blocks = pd.DataFrame({"zone_id": [10, 11]}, index=["0001", "0002"])
    blocks.index.name = "block_id"

    persons = pd.DataFrame(
        {"household_id": [1, 2], "age": [40, 35]},
        index=[101, 102],
    )
    persons.index.name = "person_id"

    residential_units = pd.DataFrame(
        {"block_id": ["0001", "0002"], "year_built": [1990, 2005]},
        index=[1001, 1002],
    )
    residential_units.index.name = "unit_id"

    jobs = pd.DataFrame({"block_id": ["0001", "0002"]}, index=[5001, 5002])
    jobs.index.name = "job_id"

    graveyard = pd.DataFrame({"household_id": [1]}, index=[201])
    graveyard.index.name = "person_id"

    with pd.HDFStore(file_path, mode="w") as store:
        store.put(households_key, households)
        store.put(blocks_key, blocks)
        store.put(persons_key, persons)
        store.put(residential_key, residential_units)
        store.put(jobs_key, jobs)
        store.put(graveyard_key, graveyard)
