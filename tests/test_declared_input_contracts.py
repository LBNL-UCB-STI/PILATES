from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock

if "openmatrix" not in sys.modules:
    openmatrix_stub = types.ModuleType("openmatrix")
    openmatrix_stub.File = object
    sys.modules["openmatrix"] = openmatrix_stub

if "geopandas" not in sys.modules:
    geopandas_stub = types.ModuleType("geopandas")
    geopandas_stub.GeoDataFrame = object
    sys.modules["geopandas"] = geopandas_stub

from pilates.config.models import PilatesConfig
from pilates.activitysim.postprocessor import ActivitysimPostprocessor
from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.activitysim.runner import ActivitysimRunner
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    FINAL_SKIMS_OMX,
    ZARR_SKIMS,
)


class _StubState:
    def __init__(self, forecast_year: int = 2018, start_year: bool = True):
        self.forecast_year = forecast_year
        self._start_year = start_year

    def is_start_year(self) -> bool:
        return self._start_year


class _StubWorkspace:
    def __init__(self, root: Path):
        self._root = root
        self.full_path = str(root)

    def get_usim_mutable_data_dir(self):
        return os.path.join(self._root, "usim", "data")

    def get_asim_mutable_configs_dir(self):
        return os.path.join(self._root, "asim", "configs")

    def get_asim_mutable_data_dir(self):
        return os.path.join(self._root, "asim", "data")

    def get_asim_output_dir(self):
        return os.path.join(self._root, "asim", "output")

    def get_beam_mutable_data_dir(self):
        return os.path.join(self._root, "beam", "input")


def _raise_fs_call(*_args, **_kwargs):
    raise AssertionError("filesystem access is not allowed in declared contracts")


def _build_settings(tmp_path: Path) -> PilatesConfig:
    return PilatesConfig(
        run={
            "region": "test",
            "scenario": "test",
            "start_year": 2018,
            "end_year": 2018,
            "output_directory": str(tmp_path / "outputs"),
            "output_run_name": "test_run",
            "models": {
                "land_use": None,
                "travel": None,
                "activity_demand": None,
                "vehicle_ownership": None,
            },
        },
        shared={
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
        infrastructure={
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        urbansim={
            "local_data_input_folder": "usim/input",
            "local_mutable_data_folder": "usim/data",
            "client_base_folder": "/usim",
            "client_data_folder": "/usim/data",
            "input_file_template": "usim_{region_id}.h5",
            "input_file_template_year": "usim_{region_id}_{year}.h5",
            "output_file_template": "usim_{year}.h5",
            "command_template": "run_usim",
            "region_mappings": {"region_to_region_id": {"test": "123"}},
        },
        activitysim={
            "local_input_folder": "asim/input",
            "local_mutable_data_folder": "asim/data",
            "local_output_folder": "asim/output",
            "local_configs_folder": "asim/configs",
            "local_mutable_configs_folder": "asim/configs_mutable",
            "validation_folder": "asim/validation",
            "command_template": "asim run",
            "final_plans_folder": "asim/final_plans",
            "region_mappings": {"region_to_subdir": {"test": "test"}},
        },
    )


def test_declared_expected_inputs_are_filesystem_free(tmp_path):
    settings = _build_settings(tmp_path)
    workspace = _StubWorkspace(tmp_path)
    state = _StubState()

    with mock.patch(
        "pilates.activitysim.preprocessor.os.path.exists", side_effect=_raise_fs_call
    ), mock.patch(
        "pilates.activitysim.runner.os.path.exists", side_effect=_raise_fs_call
    ), mock.patch(
        "pilates.activitysim.postprocessor.os.path.exists", side_effect=_raise_fs_call
    ), mock.patch.object(Path, "exists", side_effect=_raise_fs_call):
        activitysim_contract = ActivitysimPreprocessor.declared_expected_inputs(
            settings, state, workspace
        )
        assert activitysim_contract == {
            "asim_mutable_configs_dir": os.path.join(tmp_path, "asim", "configs"),
            "usim_population_source_h5": os.path.join(
                tmp_path, "usim", "data", "usim_123.h5"
            ),
            FINAL_SKIMS_OMX: os.path.join(
                tmp_path, "beam", "input", "test", "skims.omx"
            ),
        }

        runner_contract = ActivitysimRunner.declared_expected_inputs(
            settings, state, workspace
        )
        assert runner_contract == {
            ASIM_LAND_USE_IN: tmp_path
            / "activitysim"
            / "native-launch"
            / "data"
            / "land_use.csv",
            ASIM_HOUSEHOLDS_IN: tmp_path
            / "activitysim"
            / "native-launch"
            / "data"
            / "households.csv",
            ASIM_PERSONS_IN: tmp_path
            / "activitysim"
            / "native-launch"
            / "data"
            / "persons.csv",
            ASIM_OMX_SKIMS: tmp_path
            / "activitysim"
            / "native-launch"
            / "data"
            / "skims.omx",
            ZARR_SKIMS: tmp_path / "asim" / "output" / "cache" / "skims.zarr",
        }

        postprocessor_contract = ActivitysimPostprocessor.declared_expected_inputs(
            settings, state, workspace
        )
        assert postprocessor_contract["asim_output_dir"] == os.path.join(
            tmp_path, "asim", "output"
        )
        assert postprocessor_contract[ZARR_SKIMS] == os.path.join(
            tmp_path, "asim", "output", "cache", "skims.zarr"
        )
