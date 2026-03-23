from __future__ import annotations

from pathlib import Path
import csv

from pilates.impacts.postprocessor import ImpactsPostprocessor
from pilates.impacts.preprocessor import ImpactsPreprocessor
from pilates.impacts.runner import ImpactsRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(value):
    if isinstance(value, dict):
        return AttrDict({key: _wrap(val) for key, val in value.items()})
    return value


def _make_settings(tmp_path: Path) -> AttrDict:
    del tmp_path
    return _wrap(
        {
            "infrastructure": {
                "container_manager": "docker",
                "docker_images": {"impacts": "impacts:latest"},
            },
            "run": {
                "use_stubs": True,
                "models": {
                    "land_use": None,
                    "travel": None,
                    "activity_demand": None,
                    "vehicle_ownership": None,
                    "impacts": "impacts",
                },
            },
            "impacts": {
                "local_input_folder": "impacts/input",
                "local_output_folder": "impacts/output",
                "container_input_folder": "/app/input",
                "container_output_folder": "/app/output",
                "command_template": (
                    "python -m impacts "
                    "--input-manifest {container_input_manifest} "
                    "--output {container_exposure_output}"
                ),
                "exposure_output_filename": "exposure_table.csv",
                "raw_exposure_output_filename": "exposure_raw.csv",
                "input_manifest_filename": "inputs_manifest.yaml",
                "run_manifest_filename": "run_manifest.yaml",
                "postprocess_manifest_filename": "postprocess_manifest.yaml",
            },
            "beam": {
                "local_mutable_data_folder": "beam/input",
                "local_output_folder": "beam/output",
                "config": "beam.conf",
            },
            "activitysim": {
                "local_mutable_data_folder": "activitysim/data",
                "local_output_folder": "activitysim/output",
            },
            "urbansim": {"local_mutable_data_folder": "urbansim/data"},
            "atlas": {
                "host_mutable_input_folder": "atlas/input",
                "host_output_folder": "atlas/output",
            },
        }
    )


def _make_state(settings: AttrDict):
    return AttrDict(
        {
            "full_settings": settings,
            "set_sub_stage_progress": lambda *_args, **_kwargs: None,
        }
    )


def test_impacts_scaffold_stages_inputs_and_stub_output(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    workspace = Workspace(settings, str(tmp_path), "run")

    beam_output = Path(workspace.get_beam_output_dir())
    beam_output.mkdir(parents=True, exist_ok=True)
    (beam_output / "network.csv.gz").write_text("network", encoding="utf-8")
    (beam_output / "skims_emissions.csv").write_text("emissions", encoding="utf-8")

    beam_input = Path(workspace.get_beam_mutable_data_dir())
    beam_input.mkdir(parents=True, exist_ok=True)
    (beam_input / "region.osm.pbf").write_text("pbf", encoding="utf-8")

    asim_output = Path(workspace.get_asim_output_dir())
    asim_output.mkdir(parents=True, exist_ok=True)
    (asim_output / "households.parquet").write_text("hh", encoding="utf-8")
    (asim_output / "persons.parquet").write_text("pp", encoding="utf-8")

    asim_data = Path(workspace.get_asim_mutable_data_dir())
    asim_data.mkdir(parents=True, exist_ok=True)
    (asim_data / "land_use.csv").write_text("TAZ\n1\n", encoding="utf-8")

    state = _make_state(settings)

    preprocess = ImpactsPreprocessor("impacts", state)
    preprocess_outputs = preprocess.preprocess(workspace)
    assert preprocess_outputs.input_manifest.exists()
    assert preprocess_outputs.staged_inputs["beam_network"].endswith("network.csv.gz")
    assert preprocess_outputs.staged_inputs["beam_emissions_skims"].endswith(
        "skims_emissions.csv"
    )

    runner = ImpactsRunner("impacts", state)
    run_outputs = runner.run(preprocess_outputs, workspace)
    assert run_outputs.run_manifest.exists()
    assert run_outputs.raw_exposure_table.exists()

    postprocessor = ImpactsPostprocessor("impacts", state)
    postprocess_outputs = postprocessor.postprocess(run_outputs, workspace)
    assert postprocess_outputs.exposure_table.exists()
    assert postprocess_outputs.postprocess_manifest.exists()

    with postprocess_outputs.exposure_table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "exposure_value" in rows[0]
    assert "population_mix" in rows[0]


def test_workflow_state_enables_terminal_postprocessing_for_impacts() -> None:
    state = WorkflowState(
        start_year=2020,
        end_year=2021,
        travel_model_freq=1,
        land_use_enabled=False,
        vehicle_ownership_model_enabled=False,
        activity_demand_enabled=False,
        traffic_assignment_enabled=False,
        impacts_enabled=True,
        year=2020,
        major_stage=None,
        inner_iter=0,
        sub_stage=None,
        file_loc=None,
        asim_compiled=False,
        full_settings=AttrDict({"postprocessing": None}),
    )
    assert WorkflowState.Stage.postprocessing in state.enabled_stages
