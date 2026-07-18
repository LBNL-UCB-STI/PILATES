from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from consist import resolve_step_contract

from pilates.workflows.artifact_keys import (
    ATLAS_OUTPUT_DIR,
    ATLAS_VEHICLES2_OUTPUT,
    USIM_DATASTORE_H5,
    USIM_MUTABLE_DATA_DIR,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.steps import urbansim_atlas
from pilates.workflows.steps.urbansim_atlas import (
    ATLAS_POSTPROCESS,
    ATLAS_PREPROCESS,
    ATLAS_RUN,
    URBANSIM_POSTPROCESS,
    URBANSIM_PREPROCESS,
    URBANSIM_RUN,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(region="test"),
        urbansim=SimpleNamespace(
            command_template="urbansim {0}",
            input_file_template="input_{region_id}.h5",
            input_file_template_year=None,
            output_file_template="output_{year}.h5",
            region_id="001",
            region_mappings={"region_to_region_id": {"test": "001"}},
        ),
        atlas=SimpleNamespace(),
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        year=2030,
        current_year=2030,
        forecast_year=2030,
        start_year=2020,
        is_start_year=lambda: False,
    )


def _workspace(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        full_path=str(tmp_path),
        get_usim_mutable_data_dir=lambda: str(tmp_path / "urbansim" / "data"),
        get_atlas_mutable_input_dir=lambda: str(tmp_path / "atlas" / "input"),
        get_atlas_output_dir=lambda: str(tmp_path / "atlas" / "output"),
    )


def test_native_urbansim_atlas_definitions_resolve_to_canonical_contracts(
    tmp_path: Path,
) -> None:
    definitions = (
        URBANSIM_PREPROCESS,
        URBANSIM_RUN,
        URBANSIM_POSTPROCESS,
        ATLAS_PREPROCESS,
        ATLAS_RUN,
        ATLAS_POSTPROCESS,
    )
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    for definition in definitions:
        contract = resolve_step_contract(
            definition.function,
            year=2030,
            iteration=1,
            phase=definition.name.rsplit("_", 1)[1],
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )

        assert contract.name.startswith(f"{definition.name}__")
        assert contract.model == definition.name
        assert contract.input_binding == "paths"
        assert contract.output_paths == definition.output_paths(
            settings=settings,
            state=state,
            workspace=workspace,
        )


def test_native_output_path_providers_preserve_typed_downstream_contracts(
    tmp_path: Path,
) -> None:
    settings = _settings()
    state = _state()
    workspace = _workspace(tmp_path)

    urbansim_preprocess_paths = URBANSIM_PREPROCESS.output_paths(
        settings=settings, state=state, workspace=workspace
    )
    urbansim_run_paths = URBANSIM_RUN.output_paths(
        settings=settings, state=state, workspace=workspace
    )
    atlas_postprocess_paths = ATLAS_POSTPROCESS.output_paths(
        settings=settings, state=state, workspace=workspace
    )

    assert set(urbansim_preprocess_paths) >= {
        USIM_MUTABLE_DATA_DIR,
        USIM_DATASTORE_H5,
    }
    assert set(urbansim_run_paths) == {USIM_DATASTORE_H5}
    assert set(atlas_postprocess_paths) >= {
        ATLAS_OUTPUT_DIR,
        USIM_POPULATION_SOURCE_H5,
        ATLAS_VEHICLES2_OUTPUT,
    }


def test_native_resolver_selects_one_coupler_artifact_at_its_model_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "urbansim-data"
    source.parent.mkdir()
    source.write_text("h5 bytes\n", encoding="utf-8")

    class _Coupler:
        def get(self, key: str):
            return {USIM_MUTABLE_DATA_DIR: source}.get(key)

    workspace = _workspace(tmp_path)
    resolved = URBANSIM_RUN.resolve_inputs(
        settings=_settings(),
        state=_state(),
        workspace=workspace,
        coupler=_Coupler(),
    )

    assert resolved.binding.inputs == {USIM_MUTABLE_DATA_DIR: source}
    assert resolved.source_by_role == {USIM_MUTABLE_DATA_DIR: "coupler"}
    assert resolved.logical_destinations == {
        USIM_MUTABLE_DATA_DIR: Path(workspace.get_usim_mutable_data_dir())
    }


def test_native_atlas_postprocess_projector_uses_persisted_outputs_only(
    monkeypatch, tmp_path: Path
) -> None:
    atlas_output_dir = tmp_path / "atlas-output"
    atlas_output_dir.mkdir()
    usim_h5 = tmp_path / "population.h5"
    usim_h5.touch()
    vehicles2 = atlas_output_dir / "vehicles2_2030.csv"
    vehicles2.write_text("vehicle_id\n1\n", encoding="utf-8")
    declared_outputs = {
        ATLAS_OUTPUT_DIR: atlas_output_dir,
        USIM_POPULATION_SOURCE_H5: usim_h5,
        ATLAS_VEHICLES2_OUTPUT: vehicles2,
    }
    monkeypatch.setattr(
        urbansim_atlas,
        "_atlas_postprocess_native_output_paths",
        lambda **_kwargs: declared_outputs,
    )

    projected = ATLAS_POSTPROCESS.project_outputs(
        {
            ATLAS_OUTPUT_DIR: SimpleNamespace(path=atlas_output_dir),
            USIM_POPULATION_SOURCE_H5: SimpleNamespace(path=usim_h5),
            ATLAS_VEHICLES2_OUTPUT: SimpleNamespace(path=vehicles2),
        },
        settings=_settings(),
        state=_state(),
        workspace=_workspace(tmp_path),
    )

    assert projected.atlas_output_dir == atlas_output_dir
    assert projected.usim_datastore_h5 == usim_h5
    assert projected.processed_outputs == {ATLAS_VEHICLES2_OUTPUT: vehicles2}


def test_native_atlas_projector_rejects_source_mount_when_destination_missing(
    monkeypatch, tmp_path: Path
) -> None:
    atlas_output_dir = tmp_path / "current" / "atlas-output"
    atlas_output_dir.mkdir(parents=True)
    usim_h5 = tmp_path / "current" / "population.h5"
    usim_h5.touch()
    declared_vehicles2 = atlas_output_dir / "vehicles2_2030.csv"
    source_mount = tmp_path / "source-mount" / "vehicles2_2030.csv"
    source_mount.parent.mkdir()
    source_mount.write_text("historical vehicle output\n", encoding="utf-8")
    declared_outputs = {
        ATLAS_OUTPUT_DIR: atlas_output_dir,
        USIM_POPULATION_SOURCE_H5: usim_h5,
        ATLAS_VEHICLES2_OUTPUT: declared_vehicles2,
    }
    monkeypatch.setattr(
        urbansim_atlas,
        "_atlas_postprocess_native_output_paths",
        lambda **_kwargs: declared_outputs,
    )

    with pytest.raises(
        RuntimeError,
        match="atlas_postprocess output 'atlas_vehicles2_output' is missing at declared destination",
    ):
        ATLAS_POSTPROCESS.project_outputs(
            {
                ATLAS_OUTPUT_DIR: SimpleNamespace(path=atlas_output_dir),
                USIM_POPULATION_SOURCE_H5: SimpleNamespace(path=usim_h5),
                ATLAS_VEHICLES2_OUTPUT: SimpleNamespace(
                    path=source_mount,
                    container_uri="workspace://current/atlas-output/vehicles2_2030.csv",
                ),
            },
            settings=_settings(),
            state=_state(),
            workspace=_workspace(tmp_path),
        )
