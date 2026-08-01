from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from consist import BindingResult, resolve_step_contract

from pilates.activitysim.outputs import ASIM_REQUIRED_RUN_OUTPUT_KEYS
from pilates.config.models import BeamArtifactFormatsConfig
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ATLAS_VEHICLES2_OUTPUT,
    BEAM_CONFIG_FILE,
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    FINAL_SKIMS_OMX,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.resolved_inputs import ResolvedStepInputs
from pilates.workflows.steps import (
    STEP_DEFINITIONS,
    activitysim_preprocess,
    atlas_postprocess,
    beam_postprocess,
    postprocessing_definition,
    urbansim_run,
)


_DEFAULT_ACTIVITYSIM = object()


def _settings(*, activitysim: object | None = _DEFAULT_ACTIVITYSIM) -> SimpleNamespace:
    if activitysim is _DEFAULT_ACTIVITYSIM:
        activitysim = SimpleNamespace(
            output_tables={
                "tables": [
                    "accessibility",
                    "beam_plans",
                    "disaggregate_accessibility",
                    "households",
                    "joint_tour_participants",
                    "land_use",
                    "non_mandatory_tour_destination_accessibility",
                    "persons",
                    "tours",
                    "trips",
                ]
            }
        )
    return SimpleNamespace(
        run=SimpleNamespace(
            region="test",
            models=SimpleNamespace(land_use=None),
        ),
        urbansim=SimpleNamespace(
            command_template="urbansim {0}",
            input_file_template="input_{region_id}.h5",
            input_file_template_year=None,
            output_file_template="output_{year}.h5",
            region_id="001",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
        atlas=SimpleNamespace(beamac=0),
        activitysim=activitysim,
        beam=SimpleNamespace(
            config="beam.conf",
            full_skim=None,
            artifact_formats=BeamArtifactFormatsConfig(),
        ),
        write_skims_to_omx=False,
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        year=2030,
        current_year=2030,
        forecast_year=2030,
        iteration=1,
        current_inner_iter=1,
        start_year=2020,
        is_start_year=lambda: False,
    )


def _workspace(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        full_path=str(tmp_path),
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data"),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "input"),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas" / "output"),
        get_asim_mutable_data_dir=lambda: str(tmp_path / "activitysim" / "data"),
        get_asim_mutable_configs_dir=lambda: str(tmp_path / "activitysim" / "configs"),
        get_asim_output_dir=lambda: str(tmp_path / "activitysim" / "output"),
        get_beam_mutable_data_dir=lambda: str(tmp_path / "beam" / "input"),
        get_beam_output_dir=lambda: str(tmp_path / "beam" / "output"),
    )


def _path(value: Any) -> Path:
    return Path(getattr(value, "path", value))


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.name, encoding="utf-8")


def _configure_urbansim_static_sources(
    settings: SimpleNamespace, tmp_path: Path
) -> None:
    """Provide the configured files fingerprinted by the UrbanSim run contract."""

    source_root = tmp_path / "urbansim-source"
    beam_root = tmp_path / "beam-source"
    zone_source = tmp_path / "geography" / "zones.geojson"
    settings.urbansim.local_data_input_folder = str(source_root)
    settings.beam.local_input_folder = str(beam_root)
    settings.shared = SimpleNamespace(
        skims=SimpleNamespace(fname="skims.omx"),
        geography=SimpleNamespace(
            zones=SimpleNamespace(
                zone_type="taz",
                source_file=str(zone_source),
                canonical_id_col="TAZ",
            ),
            alternative_zones=None,
        ),
    )
    region = settings.run.region
    region_id = settings.urbansim.region_mappings["region_to_region_id"][region]
    for source in (
        beam_root / region / "skims.omx",
        source_root / f"hsize_ct_{region_id}.csv",
        source_root / f"income_rates_{region_id}.csv",
        source_root / f"relmap_{region_id}.csv",
        source_root / "schools_2010.csv",
        source_root / "blocks_school_districts_2010.csv",
        zone_source,
    ):
        _write_file(source)


def _contract_output_paths(
    output_paths: Mapping[str, Any],
) -> dict[str, Path]:
    return {key: _path(value) for key, value in output_paths.items()}


def test_native_step_definition_registry_is_complete_and_consist_resolvable(
    monkeypatch, tmp_path: Path
) -> None:
    expected = {
        "urbansim_run",
        "urbansim_postprocess",
        "atlas_preprocess",
        "atlas_run",
        "atlas_postprocess",
        "activitysim_preprocess",
        "activitysim_run",
        "activitysim_postprocess",
        "beam_preprocess",
        "beam_run",
        "beam_postprocess",
        "beam_full_skim",
        "postprocessing",
    }
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)
    _configure_urbansim_static_sources(settings, tmp_path)
    monkeypatch.setattr(
        "pilates.workflows.step_consist_meta.build_step_consist_kwargs",
        lambda *_args, **_kwargs: {},
    )

    assert set(STEP_DEFINITIONS) == expected
    for name, definition in STEP_DEFINITIONS.items():
        contract = resolve_step_contract(
            definition.function,
            year=2030,
            iteration=1,
            phase="run",
            stage="test",
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )

        assert contract.name == f"{name}__y2030__i1__phase_run"
        assert contract.model == (
            "activitysim" if name.startswith("activitysim_") else name
        )
        assert contract.input_binding == "paths"

        if definition.output_paths is None:
            assert contract.outputs is None
            assert contract.output_paths is None
            continue

        resolved_inputs = (
            ResolvedStepInputs(
                step_name="activitysim_run",
                binding=BindingResult(inputs={}),
                metadata={"activitysim_produces_zarr": True},
            )
            if name == "activitysim_run"
            else None
        )
        declared_paths = definition.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
            resolved_inputs=resolved_inputs,
        )
        if contract.output_paths is None:
            assert contract.outputs is not None
            assert set(contract.outputs).issubset(declared_paths)
        else:
            assert _contract_output_paths(
                contract.output_paths
            ) == _contract_output_paths(declared_paths)


def test_native_steps_declare_static_consist_metadata() -> None:
    expected = {
        "urbansim_run": {
            "inputs": {USIM_DATASTORE_H5},
            "optional": {FINAL_SKIMS_OMX},
            "schema_outputs": {USIM_DATASTORE_H5, "usim_forecast_output"},
        },
        "urbansim_postprocess": {
            "inputs": {USIM_DATASTORE_H5},
            "optional": set(),
            "schema_outputs": {USIM_DATASTORE_H5, USIM_POPULATION_SOURCE_H5},
        },
        "atlas_preprocess": {
            "inputs": {USIM_DATASTORE_H5},
            "optional": {FINAL_SKIMS_OMX},
            "schema_outputs": {
                "atlas_households_csv",
                "atlas_blocks_csv",
                "atlas_persons_csv",
                "atlas_residential_csv",
                "atlas_jobs_csv",
                "atlas_grave_csv",
                "atlas_accessibility_csv",
            },
        },
        "atlas_run": {
            "inputs": {
                "atlas_households_csv",
                "atlas_blocks_csv",
                "atlas_persons_csv",
                "atlas_residential_csv",
                "atlas_jobs_csv",
                "atlas_grave_csv",
                "atlas_accessibility_csv",
            },
            "optional": set(),
            "schema_outputs": set(),
        },
        "atlas_postprocess": {
            "inputs": {
                "atlas_householdv_csv",
                "atlas_vehicles_csv",
                USIM_DATASTORE_H5,
            },
            "optional": set(),
            "schema_outputs": {USIM_POPULATION_SOURCE_H5, ATLAS_VEHICLES2_OUTPUT},
        },
        "activitysim_preprocess": {
            "inputs": {USIM_POPULATION_SOURCE_H5},
            "optional": {FINAL_SKIMS_OMX},
            "schema_outputs": {
                ASIM_LAND_USE_IN,
                ASIM_HOUSEHOLDS_IN,
                ASIM_PERSONS_IN,
                ASIM_OMX_SKIMS,
            },
        },
        "activitysim_run": {
            "inputs": {
                ASIM_LAND_USE_IN,
                ASIM_HOUSEHOLDS_IN,
                ASIM_PERSONS_IN,
            },
            "optional": {ZARR_SKIMS, ASIM_OMX_SKIMS},
            "schema_outputs": {ZARR_SKIMS, *ASIM_REQUIRED_RUN_OUTPUT_KEYS},
        },
        "activitysim_postprocess": {
            "inputs": {
                ASIM_HOUSEHOLDS_IN,
                ASIM_PERSONS_IN,
                ASIM_LAND_USE_IN,
                ASIM_OMX_SKIMS,
                ZARR_SKIMS,
            },
            "optional": {
                *ASIM_REQUIRED_RUN_OUTPUT_KEYS,
                USIM_POPULATION_SOURCE_H5,
                USIM_DATASTORE_CURRENT_H5,
                USIM_DATASTORE_BASE_H5,
            },
            "schema_outputs": {USIM_DATASTORE_H5, *ASIM_REQUIRED_RUN_OUTPUT_KEYS},
        },
        "beam_preprocess": {
            "inputs": {
                BEAM_CONFIG_FILE,
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
            },
            "optional": {LINKSTATS_WARMSTART, ATLAS_VEHICLES2_OUTPUT},
            "schema_outputs": {
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
                LINKSTATS_WARMSTART,
                "vehicles_beam_in",
            },
        },
        "beam_run": {
            "inputs": {
                BEAM_CONFIG_FILE,
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
            },
            "optional": {LINKSTATS_WARMSTART, ZARR_SKIMS},
            "schema_outputs": {LINKSTATS, "beam_plans_out"},
        },
        "beam_postprocess": {
            "inputs": set(),
            "optional": {ZARR_SKIMS},
            "schema_outputs": {ZARR_SKIMS, "final_skims_omx"},
        },
        "beam_full_skim": {
            "inputs": {
                BEAM_CONFIG_FILE,
                BEAM_PLANS_IN,
                BEAM_HOUSEHOLDS_IN,
                BEAM_PERSONS_IN,
            },
            "optional": {LINKSTATS_WARMSTART},
            "schema_outputs": {"beam_full_skims"},
        },
    }

    for name, expected_metadata in expected.items():
        metadata = STEP_DEFINITIONS[name].function.__consist_step__
        assert set(metadata.inputs or ()) == expected_metadata["inputs"]
        assert set(metadata.optional_input_keys or ()) == expected_metadata["optional"]
        assert set(metadata.schema_outputs or ()) == expected_metadata["schema_outputs"]


def test_native_model_family_projectors_agree_with_declared_destinations(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    activitysim_paths = activitysim_preprocess.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=None,
    )
    activitysim_outputs = {
        key: _path(activitysim_paths[key])
        for key in (ASIM_LAND_USE_IN, ASIM_HOUSEHOLDS_IN, ASIM_PERSONS_IN)
    }
    for path in activitysim_outputs.values():
        _write_file(path)
    activitysim_projected = activitysim_preprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{path.name}")
            for key, path in activitysim_outputs.items()
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=ResolvedStepInputs(
            step_name="activitysim_preprocess",
            binding=BindingResult(inputs={}),
        ),
    )
    assert activitysim_projected.land_use_table == activitysim_outputs[ASIM_LAND_USE_IN]
    assert (
        activitysim_projected.households_table
        == activitysim_outputs[ASIM_HOUSEHOLDS_IN]
    )
    assert activitysim_projected.persons_table == activitysim_outputs[ASIM_PERSONS_IN]

    urbansim_paths = urbansim_run.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=None,
    )
    urbansim_datastore = _path(urbansim_paths[USIM_DATASTORE_H5])
    _write_file(urbansim_datastore)
    urbansim_projected = urbansim_run.project_outputs(
        {USIM_DATASTORE_H5: SimpleNamespace(container_uri="archive://prior/store.h5")},
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=ResolvedStepInputs(
            step_name="urbansim_run",
            binding=BindingResult(inputs={}),
        ),
    )
    assert urbansim_projected.usim_datastore_h5 == urbansim_datastore

    atlas_paths = atlas_postprocess.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=None,
    )
    atlas_output_dir = Path(workspace.get_atlas_output_dir())
    atlas_population = _path(atlas_paths[USIM_POPULATION_SOURCE_H5])
    atlas_vehicles = _path(atlas_paths[ATLAS_VEHICLES2_OUTPUT])
    _write_file(atlas_population)
    _write_file(atlas_vehicles)
    atlas_projected = atlas_postprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{key}")
            for key in (
                USIM_POPULATION_SOURCE_H5,
                ATLAS_VEHICLES2_OUTPUT,
            )
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=ResolvedStepInputs(
            step_name="atlas_postprocess",
            binding=BindingResult(inputs={}),
        ),
    )
    assert atlas_projected.atlas_output_dir == atlas_output_dir
    assert atlas_projected.usim_datastore_h5 == atlas_population
    assert atlas_projected.processed_outputs == {ATLAS_VEHICLES2_OUTPUT: atlas_vehicles}


def test_beam_postprocess_dynamic_output_map_is_resolved_once_for_contract_and_projection(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "source-events.parquet"
    pd.DataFrame({"type": ["Event A", "PathTraversal"]}).to_parquet(
        events_path, index=False
    )
    settings = _settings(activitysim=None)
    state = _state()
    workspace = _workspace(tmp_path)

    class Coupler:
        values = {
            "events_parquet_2030_1": events_path,
            "raw_od_skims_2030_1": tmp_path / "source-skims.omx",
        }

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

        def keys(self) -> tuple[str, ...]:
            return tuple(self.values)

    resolved = beam_postprocess.resolve_inputs(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=Coupler(),
    )
    dynamic_paths = resolved.metadata["beam_postprocess_output_paths"]
    assert isinstance(dynamic_paths, Mapping)
    declared_paths = beam_postprocess.output_paths(
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    assert {
        key: _path(declared_paths[key]) for key in dynamic_paths
    } == _contract_output_paths(dynamic_paths)
    for path in dynamic_paths.values():
        _write_file(_path(path))

    projected = beam_postprocess.project_outputs(
        {
            key: SimpleNamespace(container_uri=f"archive://prior/{_path(path).name}")
            for key, path in dynamic_paths.items()
        },
        settings=settings,
        state=state,
        workspace=workspace,
        resolved_inputs=resolved,
    )
    assert projected.split_events == {
        key: _path(path)
        for key, path in dynamic_paths.items()
        if key.startswith("events_parquet_")
    }
    assert projected.split_event_links == {
        key: _path(path)
        for key, path in dynamic_paths.items()
        if key.startswith("path_traversal_links_")
    }


def test_native_postprocessing_definition_is_overwrite_only_and_outputless() -> None:
    options = postprocessing_definition.cache_options()

    assert options.cache_mode == "overwrite"
    assert (
        postprocessing_definition.project_outputs(
            {}, settings=object(), state=object(), workspace=object()
        )
        is None
    )
