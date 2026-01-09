import os
import tempfile
import unittest
from datetime import datetime

import duckdb

import sys
import types

def _stub_module(module_name: str, attrs=None):
    mod = sys.modules.get(module_name)
    if mod is None:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
    if attrs:
        for key, value in attrs.items():
            setattr(mod, key, value)
    return mod


# `pilates.activitysim.preprocessor` imports heavy optional deps at module import time.
# Stub them so this unit test can run in lightweight environments.
_stub_module("openmatrix", attrs={"File": object, "open_file": lambda *a, **k: None})
_stub_module("geopandas", attrs={"GeoDataFrame": object, "GeoSeries": object})
shapely_mod = _stub_module("shapely")
shapely_wkt_mod = _stub_module("shapely.wkt")
shapely_geometry_mod = _stub_module("shapely.geometry", attrs={"Polygon": object})
setattr(shapely_mod, "wkt", shapely_wkt_mod)
setattr(shapely_mod, "geometry", shapely_geometry_mod)
_stub_module("tqdm", attrs={"tqdm": lambda x, *a, **k: x})
matplotlib_mod = _stub_module("matplotlib")
matplotlib_pyplot_mod = _stub_module("matplotlib.pyplot")
setattr(matplotlib_mod, "pyplot", matplotlib_pyplot_mod)

# Minimal OpenLineage stubs (used by provenance + record models).
openlineage_mod = _stub_module("openlineage")
openlineage_client_mod = _stub_module(
    "openlineage.client",
    attrs={"set_producer": lambda *a, **k: None, "OpenLineageClient": object},
)
setattr(openlineage_mod, "client", openlineage_client_mod)
openlineage_transport_mod = _stub_module("openlineage.client.transport")
setattr(openlineage_client_mod, "transport", openlineage_transport_mod)
_stub_module(
    "openlineage.client.facet",
    attrs={
        "SchemaField": object,
        "SchemaDatasetFacet": object,
        "DocumentationJobFacet": object,
        "SourceCodeLocationJobFacet": object,
    },
)
_stub_module(
    "openlineage.client.run",
    attrs={
        "Dataset": object,
        "InputDataset": object,
        "OutputDataset": object,
        "RunEvent": object,
        "Run": object,
        "Job": object,
        "RunState": object,
    },
)
_stub_module(
    "openlineage.client.transport.http",
    attrs={"HttpTransport": object, "HttpConfig": object},
)
_stub_module(
    "openlineage.client.transport.file",
    attrs={"FileTransport": object, "FileConfig": object},
)
_stub_module(
    "openlineage.client.transport.composite",
    attrs={"CompositeTransport": object, "CompositeConfig": object},
)

from pilates.activitysim.preprocessor import ActivitysimPreprocessor
from pilates.config.models import load_config
from pilates.generic.records_legacy import FileRecord, PilatesRunInfo
from pilates.utils.duckdb_manager import DuckDBManager
from pilates.utils.provenance import FileProvenanceTracker
from workflow_state import WorkflowState


class TestLegacyCompatibility(unittest.TestCase):
    def test_duckdb_upsert_works_with_legacy_run_table(self):
        conn = duckdb.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE run (
                run_id VARCHAR PRIMARY KEY,
                created_at VARCHAR,
                start_year INTEGER,
                end_year INTEGER,
                models_used JSON,
                settings_hash VARCHAR,
                code_version VARCHAR,
                hostname VARCHAR,
                config_snapshot_id VARCHAR,
                config_content_hash VARCHAR
            )
            """
        )

        manager = DuckDBManager(database_path=":memory:")
        run_info = PilatesRunInfo(run_id="r1", created_at=datetime.now().isoformat())
        manager.upsert_pilates_run(conn, run_info, config_snapshot_id=None)

        self.assertEqual(conn.execute("SELECT COUNT(*) FROM run").fetchone()[0], 1)

    def test_activitysim_preprocessor_inner_iter_falls_back_without_activitysim_run(self):
        temp_dir = tempfile.mkdtemp(prefix="pilates_legacy_asim_pre_")
        try:
            # Minimal settings sufficient for the preprocessor path under test
            settings_dict = {
                "run": {
                    "region": "test",
                    "scenario": "test",
                    "start_year": 2020,
                    "end_year": 2020,
                    "output_directory": temp_dir,
                    "output_run_name": "test_run",
                    "models": {
                        "land_use": None,
                        "travel": None,
                        "activity_demand": None,
                        "vehicle_ownership": None,
                    },
                },
                "shared": {
                    "skims": {
                        "zone_type": "taz",
                        "fname": "skims.omx",
                        "geoms_fname": "geoms.geojson",
                        "geoms_index_col": "TAZ",
                    },
                    "database": {"enabled": False, "type": "duckdb", "path": ":memory:"},
                    "geography": {"FIPS": {"county": ["00000"]}, "local_crs": "EPSG:4326"},
                },
                "infrastructure": {
                    "container_manager": "docker",
                    "docker_images": {},
                    "singularity_images": {},
                    "docker_config": {"stdout": False, "pull_latest": False},
                },
            }
            settings_path = os.path.join(temp_dir, "settings.yaml")
            with open(settings_path, "w") as f:
                import yaml

                yaml.safe_dump(settings_dict, f)
            settings = load_config(settings_path)

            state = WorkflowState(
                start_year=settings.run.start_year,
                end_year=settings.run.end_year,
                travel_model_freq=1,
                land_use_enabled=False,
                vehicle_ownership_model_enabled=False,
                activity_demand_enabled=False,
                traffic_assignment_enabled=False,
                replanning_enabled=False,
                year=settings.run.start_year,
                major_stage=WorkflowState.Stage.supply_demand_loop,
                inner_iter=1,  # triggers reuse path
                sub_stage=WorkflowState.Stage.activity_demand,
                file_loc=temp_dir,
                asim_compiled=False,
                full_settings=settings,
            )

            provenance = FileProvenanceTracker(run_id="r1", output_path=temp_dir)

            # Seed cached asim inputs (no activitysim model run recorded)
            now = datetime.now().isoformat()
            cached_records = [
                FileRecord(
                    file_path=os.path.join(temp_dir, "households.csv"),
                    models=["activitysim_preprocessor"],
                    short_name="households_asim_in",
                    created_at=now,
                ),
                FileRecord(
                    file_path=os.path.join(temp_dir, "persons.csv"),
                    models=["activitysim_preprocessor"],
                    short_name="persons_asim_in",
                    created_at=now,
                ),
                FileRecord(
                    file_path=os.path.join(temp_dir, "land_use.csv"),
                    models=["activitysim_preprocessor"],
                    short_name="land_use_asim_in",
                    created_at=now,
                ),
            ]
            for rec in cached_records:
                provenance.run_info.file_records[rec.unique_id] = rec

            # Minimal workspace stub with only the methods used in this code path
            class _WorkspaceStub:
                def __init__(self, root):
                    self.root = root
                    self.output_data = {}

                def get_beam_mutable_data_dir(self):
                    return os.path.join(self.root, "beam", "data")

                def get_asim_mutable_data_dir(self):
                    return os.path.join(self.root, "asim", "data")

            workspace = _WorkspaceStub(temp_dir)

            # Create a dummy BEAM skims file in the expected location so the preprocessor can copy it.
            beam_skims_dir = os.path.join(
                workspace.get_beam_mutable_data_dir(), settings.run.region
            )
            os.makedirs(beam_skims_dir, exist_ok=True)
            with open(os.path.join(beam_skims_dir, settings.shared.skims.fname), "wb") as f:
                f.write(b"omx")

            pre = ActivitysimPreprocessor(
                model_name="activitysim_preprocessor",
                state=state,
                provenance_tracker=provenance,
            )

            outputs = pre._preprocess(workspace)
            short_names = {r.short_name for r in outputs.all_records()}

            self.assertIn("households_asim_in", short_names)
            self.assertIn("persons_asim_in", short_names)
            self.assertIn("land_use_asim_in", short_names)
            self.assertIn("omx_skims", short_names)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
