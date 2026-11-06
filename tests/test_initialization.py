"""Unit tests for step-owned workspace preparation helpers that replaced Initialization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pilates.activitysim.preprocessor import ensure_activitysim_mutable_configs
from pilates.atlas.preprocessor import stage_selected_atlas_static_inputs
from pilates.beam.preprocessor import ensure_beam_mutable_workspace
from pilates.urbansim.preprocessor import resolve_urbansim_initial_datastore_source_path


class DummyWorkspace:
    def __init__(self, root: Path):
        self.full_path = str(root)
        self.beam_mutable_dir = str(root / "beam" / "input")
        self.usim_mutable_dir = str(root / "urbansim" / "data")
        self.atlas_mutable_input_dir = str(root / "atlas" / "atlas_input")
        self.atlas_output_dir = str(root / "atlas" / "atlas_output")
        self.activitysim_mutable_dir = str(root / "activitysim" / "data")
        self.activitysim_configs_dir = str(root / "activitysim" / "configs")

    def get_beam_mutable_data_dir(self) -> str:
        return self.beam_mutable_dir

    def get_usim_mutable_data_dir(self) -> str:
        return self.usim_mutable_dir

    def get_atlas_mutable_input_dir(self) -> str:
        return self.atlas_mutable_input_dir

    def get_atlas_output_dir(self) -> str:
        return self.atlas_output_dir

    def get_asim_mutable_data_dir(self) -> str:
        return self.activitysim_mutable_dir

    def get_asim_mutable_configs_dir(self) -> str:
        return self.activitysim_configs_dir


def test_ensure_activitysim_mutable_configs_creates_required_dirs_and_copies_zones(
    tmp_path: Path,
) -> None:
    workspace = DummyWorkspace(tmp_path / "workspace")
    zones_path = tmp_path / "canonical_zones.geojson"
    zones_path.write_text(
        '{"type": "FeatureCollection", "features": []}', encoding="utf-8"
    )

    asim_configs_src = tmp_path / "asim_configs_src"
    (asim_configs_src / "test_region" / "configs").mkdir(parents=True)
    (asim_configs_src / "test_region" / "configs" / "settings.yaml").write_text(
        "models:\n  - initialize_landuse\n", encoding="utf-8"
    )
    (asim_configs_src / "common").mkdir(parents=True)
    (asim_configs_src / "common" / "constants.yaml").write_text(
        "sample: 1\n", encoding="utf-8"
    )

    settings = SimpleNamespace(
        run=SimpleNamespace(
            region="test_region",
            models=SimpleNamespace(vehicle_ownership=None),
        ),
        activitysim=SimpleNamespace(
            local_configs_folder=str(asim_configs_src),
            local_mutable_configs_folder="activitysim/configs",
            local_mutable_data_folder="activitysim/data",
            main_configs_dir="configs",
            clipped_geoms_path=None,
        ),
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                zones=SimpleNamespace(
                    source_file=str(zones_path),
                    activitysim_index_col="TAZ",
                    zone_type="block_group",
                    canonical_id_col="zone_key",
                )
            )
        ),
    )

    configs_dir = ensure_activitysim_mutable_configs(settings, workspace)
    assert configs_dir == Path(workspace.get_asim_mutable_configs_dir())
    assert (configs_dir / "configs" / "settings.yaml").is_file()
    assert (configs_dir / "configs_common" / "constants.yaml").is_file()
    assert (
        Path(workspace.get_asim_mutable_data_dir()) / "canonical_zones.geojson"
    ).is_file()


def test_ensure_beam_mutable_workspace_preserves_existing_files(
    tmp_path: Path,
) -> None:
    workspace = DummyWorkspace(tmp_path / "workspace")
    prod_dir = tmp_path / "beam_prod" / "sfbay"
    (prod_dir / "urbansim").mkdir(parents=True)
    (prod_dir / "sfbay.conf").write_text("beam.conf = 1\n", encoding="utf-8")
    (prod_dir / "urbansim" / "plans.csv.gz").write_text(
        "static_plans\n", encoding="utf-8"
    )

    dest_plans = (
        Path(workspace.get_beam_mutable_data_dir())
        / "sfbay"
        / "urbansim"
        / "plans.csv.gz"
    )
    dest_plans.parent.mkdir(parents=True, exist_ok=True)
    dest_plans.write_text("step_generated_plans\n", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(
            local_input_folder=str(tmp_path / "beam_prod"),
            local_mutable_data_folder="beam/input",
        ),
        shared=SimpleNamespace(
            skims=SimpleNamespace(fname="skims.omx"),
            geography=SimpleNamespace(zones=None),
        ),
        activitysim=None,
    )

    dest_region = ensure_beam_mutable_workspace(settings, workspace)
    assert (dest_region / "sfbay.conf").is_file()
    assert dest_plans.read_text(encoding="utf-8") == "step_generated_plans\n"


def test_stage_selected_atlas_static_inputs_is_idempotent(
    tmp_path: Path,
) -> None:
    src_file = tmp_path / "src" / "cpi.csv"
    src_file.parent.mkdir(parents=True)
    src_file.write_text("year,cpi\n2015,1.0\n", encoding="utf-8")

    dest_dir = tmp_path / "atlas_input"
    selected = (("cpi.csv", src_file),)
    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        atlas=SimpleNamespace(
            local_data_input_folder=str(tmp_path / "src"),
            default_data_input_folder=str(tmp_path / "src"),
        ),
    )

    stage_selected_atlas_static_inputs(
        settings=settings,
        output_dir=str(dest_dir),
        selected_sources=selected,
    )
    assert (dest_dir / "cpi.csv").read_text(encoding="utf-8") == "year,cpi\n2015,1.0\n"

    # Modify destination and verify idempotent call does not overwrite existing file
    (dest_dir / "cpi.csv").write_text("modified\n", encoding="utf-8")
    stage_selected_atlas_static_inputs(
        settings=settings,
        output_dir=str(dest_dir),
        selected_sources=selected,
    )
    assert (dest_dir / "cpi.csv").read_text(encoding="utf-8") == "modified\n"


def test_resolve_urbansim_initial_datastore_source_path_prefers_year_template(
    tmp_path: Path,
) -> None:
    usim_dir = tmp_path / "urbansim_data"
    usim_dir.mkdir(parents=True)
    year_h5 = usim_dir / "custom_mpo_06197001_model_data_2015.h5"
    base_h5 = usim_dir / "custom_mpo_06197001_model_data.h5"
    year_h5.write_text("year", encoding="utf-8")
    base_h5.write_text("base", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay", start_year=2015),
        urbansim=SimpleNamespace(
            local_data_input_folder=str(usim_dir),
            input_file_template="custom_mpo_{region_id}_model_data.h5",
            input_file_template_year="custom_mpo_{region_id}_model_data_{start_year}.h5",
            region_mappings={"region_to_region_id": {"sfbay": "06197001"}},
        ),
    )

    assert resolve_urbansim_initial_datastore_source_path(settings) == year_h5
