"""
Canonical 3-zone, 3-block, 5-household, 8-person Micro-World Fixture Builder.

Provides a self-contained, referentially consistent synthetic environment in ``tmp_path``
with real HDF5, OpenMatrix (OMX), Zarr, Parquet, GeoJSON/Shapefile, RData, and YAML/HOCON
configs for end-to-end contract testing of UrbanSim, ATLAS, ActivitySim, and BEAM stages
without mocking data formats or preprocessors/runners/postprocessors.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Iterator

import consist
import geopandas as gpd
import numpy as np
import openmatrix as omx
import pandas as pd
from shapely.geometry import box
import xarray as xr
import yaml

from pilates.config import PilatesConfig, load_config
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_types import ScenarioWithCoupler
from pilates.workflows.artifact_keys import (
    ASIM_OMX_SKIMS,
    ATLAS_VEHICLES2_OUTPUT,
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_POPULATION_SOURCE_H5,
    ZARR_SKIMS,
)
from pilates.workflows.state import WorkflowState
from pilates.workspace import Workspace

MICRO_ZONES: tuple[int, ...] = (1, 2, 3)
MICRO_BLOCKS: tuple[str, ...] = (
    "060014001001001",
    "060014002001001",
    "060014003001001",
)
MICRO_TRACTS: tuple[str, ...] = (
    "06001400100",
    "06001400200",
    "06001400300",
)
MICRO_HOUSEHOLD_IDS: tuple[int, ...] = (101, 102, 103, 104, 105)
MICRO_PERSON_IDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)
MICRO_TIME_PERIODS: tuple[str, ...] = ("AM", "PM")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MicroWorldPaths:
    root: Path
    scratch_root: Path
    archive_root: Path
    static_inputs_root: Path
    usim_static_input_dir: Path
    atlas_static_input_dir: Path
    asim_static_input_dir: Path
    asim_static_configs_dir: Path
    beam_static_input_dir: Path
    zones_geojson: Path
    zones_shp: Path
    base_h5: Path
    base_skims_omx: Path
    base_skims_zarr: Path
    db_path: Path


@dataclass
class MicroWorldEnv:
    paths: MicroWorldPaths
    settings: PilatesConfig
    workspace: Workspace
    state: WorkflowState
    tracker: consist.Tracker

    @contextmanager
    def scenario_context(
        self,
        name: str = "micro_world_scenario",
        *,
        seed_coupler: bool = True,
    ) -> Iterator[ScenarioWithCoupler]:
        cr.set_enabled(True)
        try:
            with cr.use_tracker(self.tracker):
                with cr.scenario(
                    name=name,
                    tracker=self.tracker,
                    model="test_orchestrator",
                ) as sc:
                    if seed_coupler:
                        self.seed_scenario_coupler(sc)
                    yield sc
        finally:
            cr.set_enabled(None)

    def seed_scenario_coupler(self, sc: ScenarioWithCoupler) -> None:
        """Seed initial tracked artifacts into the scenario coupler."""
        usim_mutable = Path(self.workspace.get_usim_mutable_data_dir())
        active_h5 = usim_mutable / self.paths.base_h5.name
        if not active_h5.exists():
            active_h5 = self.paths.base_h5
        asim_mutable = Path(self.workspace.get_asim_mutable_data_dir())
        omx_path = asim_mutable / "skims.omx"
        zarr_path = Path(self.workspace.get_asim_runtime_cache_dir()) / "skims.zarr"
        atlas_out_dir = Path(self.workspace.get_atlas_output_dir())
        vehicles2_path = atlas_out_dir / f"vehicles2_{self.state.forecast_year}.csv"

        with sc.trace(
            f"{sc.name}_init",
            model="initialization",
            year=self.state.current_year,
            iteration=self.state.current_inner_iter,
        ):
            base_h5_art = cr.log_input(self.paths.base_h5, key=USIM_DATASTORE_BASE_H5)
            sc.coupler.set_from_artifact(USIM_DATASTORE_BASE_H5, base_h5_art)

            h5_art = cr.log_input(active_h5, key=USIM_DATASTORE_H5)
            for h5_key in (
                USIM_DATASTORE_H5,
                USIM_DATASTORE_CURRENT_H5,
                USIM_POPULATION_SOURCE_H5,
            ):
                sc.coupler.set_from_artifact(h5_key, h5_art)

            if omx_path.exists():
                omx_art = cr.log_input(omx_path, key=ASIM_OMX_SKIMS)
                sc.coupler.set_from_artifact(ASIM_OMX_SKIMS, omx_art)
                sc.coupler.set_from_artifact(FINAL_SKIMS_OMX, omx_art)

            if zarr_path.exists():
                zarr_art = self.tracker.log_artifact(
                    zarr_path,
                    key=ZARR_SKIMS,
                    direction="output",
                    driver="zarr",
                )
                sc.coupler.set_from_artifact(ZARR_SKIMS, zarr_art)

            if vehicles2_path.exists():
                veh_art = cr.log_input(vehicles2_path, key=ATLAS_VEHICLES2_OUTPUT)
                sc.coupler.set_from_artifact(ATLAS_VEHICLES2_OUTPUT, veh_art)


def create_micro_world_tables(base_year: int = 2010) -> dict[str, pd.DataFrame]:
    """Build referentially consistent 3-zone, 3-block, 5-household, 8-person DataFrames."""
    blocks = pd.DataFrame(
        {
            "block_id": list(MICRO_BLOCKS),
            "zone_id": [1, 2, 3],
            "taz_id": ["1", "2", "3"],
            "tract_id": list(MICRO_TRACTS),
            "county_id": ["06001", "06001", "06001"],
            "square_meters_land": [1_000_000.0, 1_500_000.0, 2_000_000.0],
            "x": [-122.40, -122.35, -122.30],
            "y": [37.78, 37.80, 37.82],
        }
    ).set_index("block_id")

    households = pd.DataFrame(
        {
            "household_id": list(MICRO_HOUSEHOLD_IDS),
            "block_id": [
                MICRO_BLOCKS[0],
                MICRO_BLOCKS[0],
                MICRO_BLOCKS[1],
                MICRO_BLOCKS[1],
                MICRO_BLOCKS[2],
            ],
            "zone_id": [1, 1, 2, 2, 3],
            "TAZ": [1, 1, 2, 2, 3],
            "income": [45_000.0, 85_000.0, 120_000.0, 32_000.0, 68_000.0],
            "persons": [2, 2, 1, 2, 1],
            "cars": [1, 2, 1, 0, 1],
            "workers": [1, 2, 1, 1, 0],
            "tenure": [1, 1, 2, 2, 1],
            "recent_mover": [0, 1, 0, 1, 0],
            "building_id": [1, 2, 3, 4, 5],
            "unit_id": [1, 2, 3, 4, 5],
            "lcm_county_id": ["06001"] * 5,
            "hispanic_head": ["no"] * 5,
            "race_of_head": [1, 1, 2, 1, 3],
            "age_of_head": [35, 42, 29, 50, 68],
            "children": [0, 0, 0, 1, 0],
            "seniors": [0, 0, 0, 0, 1],
            "veh": [1, 2, 1, 0, 1],
            "hinccat1": [2, 3, 4, 1, 3],
            "hinc": [45_000.0, 85_000.0, 120_000.0, 32_000.0, 68_000.0],
            "hworkers": [1, 2, 1, 1, 0],
            "hsize": [2, 2, 1, 2, 1],
            "hadults": [2, 2, 1, 1, 1],
            "OWN": [1, 1, 0, 0, 1],
            "TYPE": [1, 1, 1, 1, 1],
            "BLD": [2, 2, 5, 5, 2],
            "MoveIn": [0, 1, 0, 1, 0],
        }
    ).set_index("household_id")

    persons = pd.DataFrame(
        {
            "person_id": list(MICRO_PERSON_IDS),
            "household_id": [101, 101, 102, 102, 103, 104, 104, 105],
            "age": [35, 33, 42, 40, 29, 50, 12, 68],
            "member_id": [1, 2, 1, 2, 1, 1, 2, 1],
            "PNUM": [1, 2, 1, 2, 1, 1, 2, 1],
            "sex": [1, 2, 1, 2, 2, 1, 2, 1],
            "edu": [21, 20, 22, 21, 21, 18, 8, 19],
            "earning": [
                45_000.0,
                0.0,
                50_000.0,
                35_000.0,
                120_000.0,
                32_000.0,
                0.0,
                0.0,
            ],
            "hours": [40, 0, 40, 40, 45, 40, 0, 0],
            "hispanic": [0, 0, 0, 0, 0, 0, 0, 0],
            "race_id": [1, 1, 1, 1, 2, 1, 1, 3],
            "relate": [0, 1, 0, 1, 0, 0, 2, 0],
            "student": [0, 0, 0, 0, 0, 0, 1, 0],
            "worker": [1, 0, 1, 1, 1, 1, 0, 0],
            "work_at_home": [0, 0, 0, 0, 0, 0, 0, 0],
            "work_zone_id": [2, -1, 3, 2, 1, 3, -1, -1],
            "school_zone_id": [-1, -1, -1, -1, -1, -1, 2, -1],
            "workplace_taz": [2, -1, 3, 2, 1, 3, -1, -1],
            "school_taz": [-1, -1, -1, -1, -1, -1, 2, -1],
            "TAZ": [1, 1, 1, 1, 2, 2, 2, 3],
            "ptype": [1, 4, 1, 1, 1, 1, 6, 5],
            "pemploy": [1, 3, 1, 1, 1, 1, 4, 3],
            "pstudent": [3, 3, 3, 3, 3, 3, 1, 3],
        }
    ).set_index("person_id")

    jobs = pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4, 5, 6],
            "block_id": [
                MICRO_BLOCKS[0],
                MICRO_BLOCKS[0],
                MICRO_BLOCKS[1],
                MICRO_BLOCKS[1],
                MICRO_BLOCKS[2],
                MICRO_BLOCKS[2],
            ],
            "zone_id": [1, 1, 2, 2, 3, 3],
            "sector_id": [1, 2, 3, 4, 5, 6],
            "building_id": [1, 1, 3, 3, 5, 5],
        }
    ).set_index("job_id")

    residential_units = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4, 5],
            "building_id": [1, 2, 3, 4, 5],
            "block_id": [
                MICRO_BLOCKS[0],
                MICRO_BLOCKS[0],
                MICRO_BLOCKS[1],
                MICRO_BLOCKS[1],
                MICRO_BLOCKS[2],
            ],
            "tenure": [1, 1, 2, 2, 1],
        }
    ).set_index("unit_id")

    cars = pd.DataFrame(
        {
            "car_id": [1, 2, 3, 4, 5],
            "household_id": [101, 102, 102, 103, 105],
            "vehicle_id": [1, 1, 2, 1, 1],
            "bodytype": ["car", "suv", "car", "van", "pickup"],
            "modelyear": [2018, 2020, 2016, 2019, 2015],
            "adopt_fuel": ["conv", "ev", "conv", "hybrid", "conv"],
            "pred_power": ["ICE", "BEV", "ICE", "HEV", "ICE"],
            "vehicleTypeId": [
                "car-conventional",
                "suv-bev",
                "car-conventional",
                "van-hev",
                "pickup-conventional",
            ],
            "vintage_category": ["0_5", "0_5", "6_10", "0_5", "6_10"],
            "MainDriver_id": [1001, 1003, 1004, 1005, 1008],
            "ownlease": [1, 1, 1, 0, 1],
            "year": [base_year] * 5,
        }
    ).set_index("car_id")

    graveyard = pd.DataFrame(
        {
            "person_id": [9999],
            "household_id": [101],
            "member_id": [1],
            "age": [85],
            "dispose_year": [base_year],
        }
    ).set_index("person_id")

    return {
        "blocks": blocks,
        "households": households,
        "persons": persons,
        "jobs": jobs,
        "residential_units": residential_units,
        "cars": cars,
        "graveyard": graveyard,
    }


def write_urbansim_h5(
    h5_path: Path,
    tables: dict[str, pd.DataFrame],
    years: tuple[int | None, ...] = (None, 2010),
) -> Path:
    """Write tables to an HDF5 store under unprefixed and/or year-prefixed keys."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(str(h5_path), mode="w") as store:
        for year in years:
            prefix = f"/{year}/" if year is not None else "/"
            for name, df in tables.items():
                store.put(f"{prefix}{name}", df, format="fixed")
    return h5_path


def write_micro_skims_omx(omx_path: Path, zones: tuple[int, ...] = MICRO_ZONES) -> Path:
    """Create a real 3x3 OpenMatrix file with highway, transit, and distance matrices."""
    omx_path.parent.mkdir(parents=True, exist_ok=True)
    if omx_path.exists():
        omx_path.unlink()

    n = len(zones)
    dist = np.array(
        [
            [0.5, 2.0, 3.5],
            [2.0, 0.6, 2.5],
            [3.5, 2.5, 0.7],
        ],
        dtype=np.float64,
    )
    time_am = np.array(
        [
            [3.0, 10.0, 16.0],
            [10.0, 4.0, 12.0],
            [16.0, 12.0, 4.0],
        ],
        dtype=np.float64,
    )
    time_pm = time_am * 1.15
    cost = dist * 0.25

    f = omx.open_file(str(omx_path), "w")
    try:
        f.create_mapping("zone_id", list(zones))
        f.create_mapping("TAZ", list(zones))
        f["DIST"] = dist
        f["DISTBIKE"] = dist
        f["DISTWALK"] = dist
        for period, t_mat in (("AM", time_am), ("PM", time_pm), ("MD", time_am)):
            for mode in ("SOV", "HOV2", "HOV3", "SOVTOLL", "HOV2TOLL", "HOV3TOLL"):
                f[f"{mode}_TIME__{period}"] = t_mat
                f[f"{mode}_DIST__{period}"] = dist
                f[f"{mode}_BTOLL__{period}"] = cost
                f[f"{mode}_VTOLL__{period}"] = cost
                f[f"{mode}_TRIPS__{period}"] = np.full((n, n), 10.0, dtype=np.float64)
                f[f"{mode}_FAILURES__{period}"] = np.zeros((n, n), dtype=np.float64)
            for tr_path in (
                "WLK_LOC_WLK",
                "WLK_HVY_WLK",
                "WLK_COM_WLK",
                "WLK_EXP_WLK",
                "WLK_LRF_WLK",
            ):
                # AtlasPreprocessor._get_time_ODmatrix sums WACC, IWAIT, XWAIT, TOTIVT, WEGR,
                # treats <= 500 as 1e6, and divides by 100 (expecting <= 30 minutes -> <= 3000).
                f[f"{tr_path}_TOTIVT__{period}"] = np.full(
                    (n, n), 900.0, dtype=np.float64
                )
                f[f"{tr_path}_WACC__{period}"] = np.full(
                    (n, n), 100.0, dtype=np.float64
                )
                f[f"{tr_path}_WEGR__{period}"] = np.full(
                    (n, n), 100.0, dtype=np.float64
                )
                f[f"{tr_path}_IWAIT__{period}"] = np.full(
                    (n, n), 100.0, dtype=np.float64
                )
                f[f"{tr_path}_XWAIT__{period}"] = np.full(
                    (n, n), 50.0, dtype=np.float64
                )
                f[f"{tr_path}_FARE__{period}"] = np.full(
                    (n, n), 250.0, dtype=np.float64
                )
                f[f"{tr_path}_BOARDS__{period}"] = np.ones((n, n), dtype=np.float64)
                f[f"{tr_path}_TRIPS__{period}"] = np.full((n, n), 5.0, dtype=np.float64)
                f[f"{tr_path}_FAILURES__{period}"] = np.zeros((n, n), dtype=np.float64)
    finally:
        f.close()
    return omx_path


def write_micro_skims_zarr(
    zarr_path: Path,
    zones: tuple[int, ...] = MICRO_ZONES,
    time_periods: tuple[str, ...] = MICRO_TIME_PERIODS,
    marker_value: float = 0.0,
) -> Path:
    """Create a real 3x3x2 ActivitySim/BEAM Zarr skim store passing ``validate_activitysim_zarr_skims``."""
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(zones)
    p = len(time_periods)
    orig_ids = [str(z) for z in zones]
    zero_coords = np.arange(n, dtype=np.int64)
    base_2d = np.array(
        [
            [0.5, 2.0, 3.5],
            [2.0, 0.6, 2.5],
            [3.5, 2.5, 0.7],
        ],
        dtype=np.float32,
    ) + np.float32(marker_value)
    base_3d = np.stack([base_2d * (1.0 + 0.1 * idx) for idx in range(p)], axis=-1)
    trips_3d = np.full((n, n, p), 10.0, dtype=np.float32)
    zero_3d = np.zeros((n, n, p), dtype=np.float32)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "DIST": (("otaz", "dtaz"), base_2d),
        "DISTBIKE": (("otaz", "dtaz"), base_2d),
        "DISTWALK": (("otaz", "dtaz"), base_2d),
        "SOV_TIME": (("otaz", "dtaz", "time_period"), base_3d * 5.0),
        "SOV_DIST": (("otaz", "dtaz", "time_period"), base_3d),
        "SOV_BTOLL": (("otaz", "dtaz", "time_period"), base_3d * 0.2),
        "SOV_TRIPS": (("otaz", "dtaz", "time_period"), trips_3d),
        "SOV_FAILURES": (("otaz", "dtaz", "time_period"), zero_3d),
        "HOV2_TIME": (("otaz", "dtaz", "time_period"), base_3d * 4.8),
        "HOV2_DIST": (("otaz", "dtaz", "time_period"), base_3d),
        "HOV2_TRIPS": (("otaz", "dtaz", "time_period"), trips_3d),
        "HOV2_FAILURES": (("otaz", "dtaz", "time_period"), zero_3d),
        "WLK_LOC_WLK_TOTIVT": (
            ("otaz", "dtaz", "time_period"),
            np.full((n, n, p), 900.0, dtype=np.float32),
        ),
        "WLK_LOC_WLK_WAIT": (
            ("otaz", "dtaz", "time_period"),
            np.full((n, n, p), 150.0, dtype=np.float32),
        ),
        "WLK_LOC_WLK_TRIPS": (("otaz", "dtaz", "time_period"), trips_3d),
        "WLK_LOC_WLK_FAILURES": (("otaz", "dtaz", "time_period"), zero_3d),
    }

    otaz_da = xr.DataArray(
        zero_coords,
        dims=("otaz",),
        attrs={
            "preprocessed": "zero-based-contiguous",
            "original_zone_ids": orig_ids,
        },
    )
    dtaz_da = xr.DataArray(
        zero_coords,
        dims=("dtaz",),
        attrs={
            "preprocessed": "zero-based-contiguous",
            "original_zone_ids": orig_ids,
        },
    )
    tp_da = xr.DataArray(
        np.array(time_periods, dtype="<U2"),
        dims=("time_period",),
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "otaz": otaz_da,
            "dtaz": dtaz_da,
            "time_period": tp_da,
        },
        attrs={"original_zone_ids": orig_ids},
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True, zarr_format=2)
    return zarr_path


def write_beam_raw_od_skims_zarr(
    zarr_path: Path,
    zones: tuple[int, ...] = MICRO_ZONES,
    time_periods: tuple[str, ...] = MICRO_TIME_PERIODS,
) -> Path:
    """Create a real BEAM raw OD skims Zarr store matching ``merge_beam_skims_to_zarr`` expectations."""
    return write_micro_skims_zarr(zarr_path, zones=zones, time_periods=time_periods)


def _stage_static_inputs(
    static_root: Path,
    tables: dict[str, pd.DataFrame],
    region: str = "sfbay",
    region_id: str = "06197001",
    base_year: int = 2010,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    usim_dir = static_root / "urbansim"
    atlas_dir = static_root / "atlas"
    asim_input_dir = static_root / "activitysim" / "data"
    asim_configs_dir = static_root / "activitysim" / "configs_root"
    beam_dir = static_root / "beam"
    geo_dir = static_root / "geography"

    for d in (usim_dir, atlas_dir, asim_input_dir, asim_configs_dir, beam_dir, geo_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Geography zones (GeoJSON + Shapefile)
    geoms = [
        box(-122.42, 37.76, -122.38, 37.80),
        box(-122.38, 37.76, -122.34, 37.80),
        box(-122.34, 37.76, -122.30, 37.80),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "TAZ1454": ["1", "2", "3"],
            "zone_id": [1, 2, 3],
            "OBJECTID": [1, 2, 3],
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    zones_geojson = geo_dir / "zones.geojson"
    zones_shp = geo_dir / "zones.shp"
    gdf.to_file(zones_geojson, driver="GeoJSON")
    gdf.to_file(zones_shp)

    block_geoms = [
        box(-122.415, 37.765, -122.385, 37.795),
        box(-122.375, 37.765, -122.345, 37.795),
        box(-122.335, 37.765, -122.305, 37.795),
    ]
    blocks_gdf = gpd.GeoDataFrame(
        {
            "GEOID": list(MICRO_BLOCKS),
            "STATE": ["06", "06", "06"],
            "COUNTY": ["001", "001", "001"],
            "TRACT": ["400100", "400200", "400300"],
            "BLKGRP": ["1", "1", "1"],
            "BLOCK": ["1001", "1001", "1001"],
        },
        geometry=block_geoms,
        crs="EPSG:4326",
    )
    blocks_gdf.to_file(geo_dir / f"block_{region}.shp")

    # 2. UrbanSim static inputs
    base_h5 = usim_dir / f"custom_mpo_{region_id}_model_data.h5"
    write_urbansim_h5(base_h5, tables, years=(None, base_year))
    pd.DataFrame({"tract_id": list(MICRO_TRACTS), "hsize": [2, 2, 1]}).to_csv(
        usim_dir / f"hsize_ct_{region_id}.csv", index=False
    )
    pd.DataFrame({"income_cat": [1, 2, 3], "rate": [0.1, 0.2, 0.3]}).to_csv(
        usim_dir / f"income_rates_{region_id}.csv", index=False
    )
    pd.DataFrame({"rel": [0, 1, 2], "mapped": [0, 1, 2]}).to_csv(
        usim_dir / f"relmap_{region_id}.csv", index=False
    )
    pd.DataFrame(
        {"school_id": [1, 2], "block_id": [MICRO_BLOCKS[0], MICRO_BLOCKS[1]]}
    ).to_csv(usim_dir / "schools_2010.csv", index=False)
    pd.DataFrame({"block_id": list(MICRO_BLOCKS), "district_id": [1, 1, 2]}).to_csv(
        usim_dir / "blocks_school_districts_2010.csv", index=False
    )

    # 3. ATLAS static inputs (copy real static RData and vehicle_type_mapping CSVs from repo)
    repo_atlas_input = REPO_ROOT / "pilates" / "atlas" / "atlas_input"
    if repo_atlas_input.is_dir():
        for item in repo_atlas_input.iterdir():
            if item.is_file():
                shutil.copy2(item, atlas_dir / item.name)
    # Create 3-zone x 3-tract mapping with non-constant values per tract so .std() > 0
    taz_to_tract = pd.DataFrame(
        {
            "taz_id": [1, 2, 3],
            MICRO_TRACTS[0]: [0.8, 0.1, 0.1],
            MICRO_TRACTS[1]: [0.1, 0.8, 0.1],
            MICRO_TRACTS[2]: [0.1, 0.1, 0.8],
        }
    ).set_index("taz_id")
    taz_to_tract.to_csv(atlas_dir / f"taz_to_tract_{region}.csv")

    # 4. ActivitySim static inputs & configs
    base_skims_omx = asim_input_dir / "skims.omx"
    write_micro_skims_omx(base_skims_omx)
    base_skims_zarr = asim_input_dir / "skims.zarr"
    write_micro_skims_zarr(base_skims_zarr)

    asim_region_cfg_root = asim_configs_dir / region
    for cfg_sub in ("configs", "configs_extended", "configs_mp", "configs_sh_compile"):
        sub_path = asim_region_cfg_root / cfg_sub
        sub_path.mkdir(parents=True, exist_ok=True)
        settings_payload: dict[str, Any] = {
            "households_sample_size": 0,
            "chunk_size": 0,
            "num_processes": 2,
            "sharrow": True,
            "models": [
                "initialize_landuse",
                "initialize_households",
                "auto_ownership_simulate",
                "MandatoryTourFrequency",
                "write_tables",
            ],
            "output_tables": {
                "action": "include",
                "tables": [
                    {"tablename": "households", "decode_columns": {}},
                    {"tablename": "persons", "decode_columns": {}},
                    {"tablename": "tours", "decode_columns": {}},
                    {"tablename": "trips", "decode_columns": {}},
                    {"tablename": "joint_tour_participants", "decode_columns": {}},
                    {"tablename": "land_use", "decode_columns": {}},
                    {"tablename": "beam_plans", "decode_columns": {}},
                    {"tablename": "vehicles", "decode_columns": {}},
                ],
            },
        }
        with (sub_path / "settings.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(settings_payload, fh)
        with (sub_path / "constants.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump({"SAMPLE_RATE": 1.0}, fh)
        with (sub_path / "network_los.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {
                    "zone_system": 1,
                    "taz_skims": "skims.omx",
                    "skim_time_periods": {
                        "time_window": 1440,
                        "period_minutes": 60,
                        "periods": [0, 6, 12, 19, 24],
                        "labels": ["EA", "AM", "MD", "PM"],
                    },
                },
                fh,
            )

    # 5. BEAM static inputs
    beam_region_dir = beam_dir / region
    beam_urbansim_dir = beam_region_dir / "urbansim"
    beam_r5_dir = beam_region_dir / "r5-simple-no-local-tr"
    beam_urbansim_dir.mkdir(parents=True, exist_ok=True)
    beam_r5_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_skims_omx, beam_region_dir / "skims.omx")
    shutil.copy2(zones_geojson, beam_region_dir / "zones.geojson")
    for shp_part in geo_dir.glob("zones.*"):
        shutil.copy2(shp_part, beam_region_dir / shp_part.name)

    (beam_r5_dir / "physsim-network.xml").write_text(
        "<network><links/><nodes/></network>\n", encoding="utf-8"
    )
    (beam_r5_dir / "ce.osm.pbf").write_bytes(b"OSM_PBF_MICRO_STUB")

    pd.DataFrame(
        {
            "vehicleId": [1, 2, 3, 4, 5],
            "vehicleTypeId": [
                "car-conventional",
                "suv-bev",
                "car-conventional",
                "van-hev",
                "pickup-conventional",
            ],
            "householdId": [101, 102, 102, 103, 105],
        }
    ).to_csv(beam_urbansim_dir / "vehicles.csv", index=False)

    pd.DataFrame(
        {
            "vehicleTypeId": [
                "car-conventional",
                "suv-bev",
                "van-hev",
                "pickup-conventional",
                "BODY-TYPE-DEFAULT",
            ],
            "seatingCapacity": [4, 5, 7, 3, 1],
            "standingRoomCapacity": [0, 0, 0, 0, 0],
            "lengthInMeter": [4.5, 4.8, 5.1, 5.5, 0.5],
            "primaryFuelType": [
                "Gasoline",
                "Electricity",
                "Gasoline",
                "Gasoline",
                "Food",
            ],
            "primaryFuelConsumptionInJoulePerMeter": [
                2000.0,
                600.0,
                1500.0,
                2800.0,
                0.0,
            ],
            "primaryFuelCapacityInJoule": [1.5e9, 2.5e8, 1.5e9, 2.0e9, 1e6],
            "vehicleCategory": [
                "Car",
                "Car",
                "Car",
                "Car",
                "Body",
            ],
        }
    ).to_csv(beam_region_dir / "vehicleTypes.csv", index=False)

    # Sample urbansim plans/households/persons/blocks in beam_urbansim_dir
    sample_plans = pd.DataFrame(
        {
            "person_id": [1001, 1001, 1001],
            "PlanElementIndex": [1, 2, 3],
            "ActivityElement": ["activity", "leg", "activity"],
            "ActivityType": ["Home", None, "Work"],
            "x": [-122.40, None, -122.35],
            "y": [37.78, None, 37.80],
            "departure_time": [8.0, 8.0, 17.0],
            "leg_mode": [None, "WALK", None],
        }
    )
    sample_plans.to_parquet(
        beam_urbansim_dir / "sample_urbansim_plans.parquet", index=False
    )
    sample_plans.to_csv(beam_urbansim_dir / "sample_urbansim_plans.csv", index=False)
    tables["households"].reset_index().to_parquet(
        beam_urbansim_dir / "sample_urbansim_households.parquet", index=False
    )
    tables["persons"].reset_index().to_parquet(
        beam_urbansim_dir / "sample_urbansim_persons.parquet", index=False
    )
    tables["blocks"].reset_index().to_parquet(
        beam_urbansim_dir / "sample_urbansim_blocks.parquet", index=False
    )

    beam_conf_text = (
        f'beam.inputDirectory = "/app/input/{region}"\n'
        f'beam.outputs.baseOutputDirectory = "/app/output/{region}"\n'
        'beam.agentsim.simulationName = "micro-sim"\n'
        "beam.agentsim.agentSampleSizeAsFractionOfPopulation = 1.0\n"
        "beam.agentsim.firstIteration = 0\n"
        "beam.agentsim.lastIteration = 0\n"
        'beam.agentsim.taz.filePath = ${beam.inputDirectory}"/zones.shp"\n'
        'beam.agentsim.taz.tazIdFieldName = "TAZ"\n'
        'beam.routing.r5.directory = ${beam.inputDirectory}"/r5-simple-no-local-tr"\n'
        'beam.routing.r5.osmFile = ${beam.routing.r5.directory}"/ce.osm.pbf"\n'
        'matsim.modules.network.inputNetworkFile = ${beam.routing.r5.directory}"/physsim-network.xml"\n'
        'beam.physsim.inputNetworkFilePath = ${beam.routing.r5.directory}"/physsim-network.xml"\n'
        'beam.physsim.linkStatsOutputFileType = "csv.gz"\n'
        'beam.agentsim.agents.vehicles.vehicleTypesFilePath = ${beam.inputDirectory}"/vehicleTypes.csv"\n'
        'beam.agentsim.agents.vehicles.vehiclesFilePath = ${beam.inputDirectory}"/urbansim/vehicles.csv"\n'
        'beam.exchange.scenario.folder = ${beam.inputDirectory}"/urbansim"\n'
        'beam.exchange.scenario.fileFormat = "parquet"\n'
        "beam.urbansim.fractionOfModesToClear { allModes = 0.0 }\n"
        "beam.replanning.clearModes.iteration = 0\n"
        "beam.replanning.clearModes.modes = []\n"
        "beam.replanning.maxAgentPlanMemorySize = 5\n"
        'beam.warmStart.type = "disabled"\n'
        'beam.warmStart.path = ""\n'
        'beam.warmStart.initialLinkstatsFilePath = ""\n'
        'beam.router.skim.activity-sim-skimmer.fileBaseName = "skimsActivitySimOD"\n'
        'beam.router.skim.activity-sim-skimmer.fileOutputFormat = "zarr"\n'
        "beam.router.skim.full-activity-sim-skimmer.calculationTimeoutInMinutes = 60\n"
        'beam.Calibration.activitySimSkimmer.fileBaseName = "skimsActivitySimOD"\n'
        'beam.Calibration.activitySimSkimmer.fileFormat = "zarr"\n'
        'beam.Calibration.activitySimSkimmer.tazFilePath = ${beam.inputDirectory}"/zones.shp"\n'
        'beam.Calibration.activitySimSkimmer.tazIdFieldName = "TAZ1454"\n'
        'beam.outputs.events.fileOutputFormats = "parquet"\n'
        'beam.exchange.output.events.fileFormat = "parquet"\n'
        "beam.outputs.writePlans = true\n"
    )
    (beam_region_dir / "beam.conf").write_text(beam_conf_text, encoding="utf-8")

    return (
        usim_dir,
        atlas_dir,
        asim_input_dir,
        asim_configs_dir,
        beam_dir,
        zones_geojson,
        zones_shp,
        base_h5,
        base_skims_omx,
        base_skims_zarr,
    )


def build_micro_world(
    tmp_path: Path,
    *,
    region: str = "sfbay",
    region_id: str = "06197001",
    start_year: int = 2010,
    end_year: int = 2015,
    land_use_freq: int = 5,
    travel_model_freq: int = 5,
    vehicle_ownership_freq: int = 5,
    supply_demand_iters: int = 1,
    enable_archive_copy: bool = True,
    activitysim_skims_format: str = "zarr",
    sharrow: bool | None = True,
    sharrow_enabled: bool | None = None,
    persist_sharrow_cache: bool | None = True,
    beamac: int = 0,
    atlas_beamac: int | None = None,
    split_events_by_type: bool = False,
    replan_iters: int = 0,
    replan_hh_samp_size: int = 0,
    database_enabled: bool = False,
    land_use_model: str | None = "urbansim",
    vehicle_ownership_model: str | None = "atlas",
    activity_demand_model: str | None = "activitysim",
    travel_model: str | None = "beam",
    max_year_intervals: int | None = None,
    scratch_root: Path | None = None,
    archive_root: Path | None = None,
) -> MicroWorldEnv:
    """Create a complete, self-contained 3-zone micro-world in ``tmp_path``."""
    effective_sharrow = sharrow_enabled if sharrow_enabled is not None else sharrow
    effective_beamac = atlas_beamac if atlas_beamac is not None else beamac
    resolved_scratch = (scratch_root or (tmp_path / "scratch")).resolve()
    resolved_archive = (archive_root or (tmp_path / "archive")).resolve()
    static_root = (tmp_path / "static_inputs").resolve()
    db_path = (tmp_path / "provenance.duckdb").resolve()

    resolved_scratch.mkdir(parents=True, exist_ok=True)
    resolved_archive.mkdir(parents=True, exist_ok=True)

    tables = create_micro_world_tables(base_year=start_year)
    (
        usim_dir,
        atlas_dir,
        asim_input_dir,
        asim_configs_dir,
        beam_dir,
        zones_geojson,
        zones_shp,
        base_h5,
        base_skims_omx,
        base_skims_zarr,
    ) = _stage_static_inputs(
        static_root,
        tables=tables,
        region=region,
        region_id=region_id,
        base_year=start_year,
    )

    config_dict: dict[str, Any] = {
        "run": {
            "region": region,
            "scenario": "micro_contract_test",
            "start_year": start_year,
            "end_year": end_year,
            "land_use_freq": land_use_freq,
            "travel_model_freq": travel_model_freq,
            "vehicle_ownership_freq": vehicle_ownership_freq,
            "supply_demand_iters": supply_demand_iters,
            "output_directory": str(resolved_archive),
            "local_workspace_root": str(resolved_scratch),
            "output_run_name": "run",
            "enable_archive_copy": enable_archive_copy,
            "consist_db_seed_from_shared_on_start": database_enabled,
            "consist_db_snapshot_interval_seconds": 0,
            "consist_code_identity": "callable_module",
            "consist_hashing_strategy": "full",
            "models": {
                "land_use": land_use_model,
                "travel": travel_model,
                "activity_demand": activity_demand_model,
                "vehicle_ownership": vehicle_ownership_model,
            },
            **(
                {"max_year_intervals": max_year_intervals}
                if max_year_intervals is not None
                else {}
            ),
        },
        "write_skims_to_omx": True,
        "shared": {
            "geography": {
                "FIPS": {"counties": ["001"], "state": "06"},
                "local_crs": "EPSG:26910",
                "zones": {
                    "zone_type": "taz",
                    "source_file": str(zones_geojson),
                    "canonical_id_col": "TAZ1454",
                    "activitysim_index_col": "TAZ",
                    "source_crs": "EPSG:4326",
                },
            },
            "skims": {
                "fname": "skims.omx",
                "periods": list(MICRO_TIME_PERIODS),
                "hwy_paths": ["SOV", "HOV2"],
                "transit_paths": {
                    "WLK_LOC_WLK": ["TOTIVT", "WACC", "WEGR", "IWAIT", "XWAIT"]
                },
            },
            "database": {
                "enabled": database_enabled,
                "type": "duckdb",
                "path": str(db_path),
            },
        },
        "infrastructure": {
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {
                "urbansim": "pilates/urbansim:test",
                "atlas": "pilates/atlas:test",
                "activitysim": "pilates/activitysim:test",
                "beam": "pilates/beam:test",
            },
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        "urbansim": {
            "region_id": region_id,
            "local_data_input_folder": str(usim_dir),
            "local_mutable_data_folder": "urbansim/data",
            "client_base_folder": "/usim",
            "client_data_folder": "/usim/data",
            "input_file_template": "custom_mpo_{region_id}_model_data.h5",
            "input_file_template_year": "custom_mpo_{region_id}_model_data_{start_year}.h5",
            "output_file_template": "model_data_{year}.h5",
            "command_template": "-c -r {0} -i {1} -y {2} -f {3} -t {4}",
            "region_mappings": {"region_to_region_id": {region: region_id}},
        },
        "atlas": {
            "host_input_folder": str(atlas_dir),
            "warmstart_input_folder": str(atlas_dir),
            "host_mutable_input_folder": "atlas/atlas_input",
            "host_output_folder": "atlas/atlas_output",
            "container_input_folder": "/atlas/input",
            "container_output_folder": "/atlas/output",
            "basedir": "/atlas",
            "codedir": "/atlas/code",
            "beamac": effective_beamac,
            "mod": 2,
            "sample_size": 0,
            "num_processes": 2,
            "command_template": (
                "--freq {0} --outyear {1} --npe {2} --nsample {3} "
                "--basedir {4} --codedir {5} --beamac {6} --mod {7} "
                "--adscen {8} --rebfactor {9} --taxfactor {10} --discIncent {11}"
            ),
        },
        "activitysim": {
            "household_sample_size": 0,
            "chunk_size": 0,
            "num_processes": 2,
            "file_format": "parquet",
            "sharrow": effective_sharrow,
            "persist_sharrow_cache": persist_sharrow_cache,
            "local_input_folder": str(asim_input_dir),
            "local_mutable_data_folder": "activitysim/data",
            "local_output_folder": "activitysim/output",
            "local_configs_folder": str(asim_configs_dir),
            "local_mutable_configs_folder": "activitysim/configs_mutable",
            "validation_folder": "activitysim/validation",
            "main_configs_dir": "configs_extended",
            "command_template": "-h {0} -n {1} -c {2}",
            "final_plans_folder": "activitysim/final_plans",
            "replan_iters": replan_iters,
            "replan_hh_samp_size": replan_hh_samp_size,
            "region_mappings": {"region_to_subdir": {region: region}},
            "from_urbansim_col_maps": {
                "households": {
                    "persons": "PERSONS",
                    "PERSONS": "hhsize",
                    "cars": "auto_ownership",
                    "workers": "num_workers",
                },
                "persons": {
                    "member_id": "PNUM",
                },
                "land_use": {
                    "ZONE": "TAZ",
                    "COUNTY": "county_id",
                },
            },
            "to_urbansim_col_maps": {
                "households": {
                    "hhsize": "persons",
                    "num_workers": "workers",
                    "auto_ownership": "cars",
                },
                "persons": {
                    "PNUM": "member_id",
                    "workplace_taz": "work_zone_id",
                    "school_taz": "school_zone_id",
                },
            },
            "output_tables": {
                "prefix": "final_",
                "tables": [
                    "checkpoints",
                    "land_use",
                    "households",
                    "persons",
                    "tours",
                    "trips",
                    "joint_tour_participants",
                    "plans",
                ],
            },
        },
        "beam": {
            "config": "beam.conf",
            "sample": 1.0,
            "replanning_portion": 0.4,
            "memory": "4g",
            "local_input_folder": str(beam_dir),
            "local_mutable_data_folder": "beam/input",
            "local_output_folder": "beam/output",
            "scenario_folder": "urbansim",
            "router_directory": "r5-simple-no-local-tr",
            "skims_shapefile": "zones.shp",
            "skim_zone_source_id_col": "TAZ1454",
            "skim_zone_geoid_col": "TAZ1454",
            "split_events_by_type": split_events_by_type,
            "artifact_formats": {
                "activitysim_skims": activitysim_skims_format,
                "exchange": "parquet",
                "events": "parquet",
                "linkstats": "csv.gz",
            },
            "simulated_hwy_paths": ["SOV", "HOV2"],
            "asim_hwy_measure_map": {
                "TIME": "TIME_minutes",
                "DIST": "DIST_miles",
                "BTOLL": "VTOLL_cost",
            },
            "asim_transit_measure_map": {
                "WAIT": None,
                "TOTIVT": "IVT_minutes",
            },
            "asim_ridehail_measure_map": {},
            "ridehail_path_map": {},
        },
        "postprocessing": {
            "output_folder": "postprocessing/output",
            "mep_output_folder": "postprocessing/mep",
        },
    }

    cfg_file = tmp_path / "micro_world_settings.yaml"
    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_dict, fh)

    from pilates.utils import geog as pilates_geog

    pilates_geog.get_block_geoms.__defaults__ = (str(zones_geojson.parent), None)

    settings = load_config(str(cfg_file))
    archive_run_dir = str(resolved_archive / "run")
    workspace = Workspace(
        settings,
        output_path=str(resolved_scratch),
        folder_name="run",
        archive_run_dir=archive_run_dir if enable_archive_copy else None,
    )
    state = WorkflowState.from_settings(settings)
    state.current_year = start_year
    state.forecast_year = end_year
    state.current_inner_iter = 0
    if enable_archive_copy:
        state.archive_run_dir = archive_run_dir

    # Seed mutable workspace directories with initial base datastore and skims
    usim_mutable = Path(workspace.get_usim_mutable_data_dir())
    usim_mutable.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_h5, usim_mutable / base_h5.name)
    shutil.copy2(base_skims_omx, usim_mutable / f"skims_mpo_{region_id}.omx")

    asim_mutable = Path(workspace.get_asim_mutable_data_dir())
    asim_mutable.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_skims_omx, asim_mutable / "skims.omx")
    for asim_static_file in asim_input_dir.iterdir():
        if asim_static_file.is_file():
            shutil.copy2(asim_static_file, asim_mutable / asim_static_file.name)

    asim_mutable_configs = Path(workspace.get_asim_mutable_configs_dir())
    if asim_mutable_configs.exists():
        shutil.rmtree(asim_mutable_configs)
    shutil.copytree(asim_configs_dir / region, asim_mutable_configs)

    asim_cache = Path(workspace.get_asim_runtime_cache_dir())
    asim_cache.mkdir(parents=True, exist_ok=True)
    write_micro_skims_zarr(asim_cache / "skims.zarr")

    atlas_mutable_in = Path(workspace.get_atlas_mutable_input_dir())
    atlas_mutable_in.mkdir(parents=True, exist_ok=True)
    for atlas_static_file in atlas_dir.iterdir():
        if atlas_static_file.is_file():
            shutil.copy2(atlas_static_file, atlas_mutable_in / atlas_static_file.name)

    atlas_out_dir = Path(workspace.get_atlas_output_dir())
    atlas_out_dir.mkdir(parents=True, exist_ok=True)
    default_vehicles2 = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 1, 1, 1],
            "household_id": [101, 102, 102, 103, 105],
            "bodytype": ["car", "suv", "car", "car", "suv"],
            "pred_power": ["gasoline", "electricity", "gasoline", "gasoline", "hybrid"],
            "modelyear": [2015, 2015, 2012, 2014, 2015],
            "vehicleTypeId": [
                "car_conv_2015",
                "suv_ev_2015",
                "car_conv_2015",
                "car_conv_2015",
                "suv_ev_2015",
            ],
            "state_of_charge": [np.nan, 0.85, np.nan, np.nan, np.nan],
        }
    )
    for yr in sorted({start_year, end_year}):
        default_vehicles2.to_csv(atlas_out_dir / f"vehicles2_{yr}.csv", index=False)

    beam_mutable_region = Path(workspace.get_beam_mutable_data_dir()) / region
    if beam_mutable_region.exists():
        shutil.rmtree(beam_mutable_region)
    shutil.copytree(beam_dir / region, beam_mutable_region)

    Path(archive_run_dir).mkdir(parents=True, exist_ok=True)
    tracker = consist.Tracker(
        run_dir=Path(archive_run_dir),
        db_path=str(db_path),
        allow_external_paths=True,
        mounts={
            "inputs": str(static_root),
            "workspace": str(workspace.full_path),
            "beam_input": str(Path(workspace.full_path) / "beam" / "input"),
            "scratch": str(resolved_archive),
        },
        archive_mounts={
            "workspace": ".",
            "beam_input": "beam/input",
        },
        project_root=str(REPO_ROOT),
        hashing_strategy="full",
    )

    paths = MicroWorldPaths(
        root=tmp_path,
        scratch_root=resolved_scratch,
        archive_root=resolved_archive,
        static_inputs_root=static_root,
        usim_static_input_dir=usim_dir,
        atlas_static_input_dir=atlas_dir,
        asim_static_input_dir=asim_input_dir,
        asim_static_configs_dir=asim_configs_dir,
        beam_static_input_dir=beam_dir,
        zones_geojson=zones_geojson,
        zones_shp=zones_shp,
        base_h5=base_h5,
        base_skims_omx=base_skims_omx,
        base_skims_zarr=base_skims_zarr,
        db_path=db_path,
    )

    return MicroWorldEnv(
        paths=paths,
        settings=settings,
        workspace=workspace,
        state=state,
        tracker=tracker,
    )
