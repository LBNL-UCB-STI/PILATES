# IMPACTS container smoke test

On AMD64, run from the PILATES repository root with Docker running and the
PILATES Python environment active.

```sh
python -m scripts.run_impacts_container_smoke --image ghcr.io/lbnl-ucb-sti/impacts:v0.2.5
```

The test uses the bundled fixture in `tests/fixtures/impacts_smoke/` and runs
the containers with networking disabled.
Its bridge settings select `basemap: none`, while retaining map plots, road overlays, and scale bars.
Each run prints its output path and keeps the workspace and `run.log` in a new
folder under `test-output/impacts-replay/`, including after a test failure.

- Presim reads the sampled raw EMFAC export and generates inventories, rates,
  and the passenger and freight fleets. The test checks that the generated
  assignments reference generated BEAM vehicle types with valid rate files.
- For the archived BEAM replay, the test stages the new rates at the paths
  referenced by the recorded vehicle types and checks that they cover every
  vehicle type and emissions process used by the archived skims.
- Postsim reads the archived skims and the new presim inventories. The test
  checks the run, scenario, source paths, road and vehicle IDs, and county
  emissions keys. It independently computes InMAP primary PM2.5 and NO2 from
  the sample coefficients, checks how exposure combines InMAP and AERMOD
  concentrations, and reconciles the per-person assignments and cell counts.
- Both smoke fixtures include a synthetic UrbanSim HDF5 separately from the
  BEAM population. Each includes one extra resident with no BEAM plan. The
  exposure check requires that resident in the output and verifies the source
  HDF5 and assignment counts. The replay also checks that IMPACTS rejects
  exposure settings missing the HDF5 while BEAM inputs are mounted.

Pass `--replay-without-fleet` to exercise the archived BEAM outputs and
new EMFAC rates without regenerating the fleet.
The replay uses archived BEAM outputs. A live handoff check uses skims from a
BEAM run launched with the generated fleet. Optional baseline delta plots
require a second scenario.

## Optional live test in San Francisco

With Docker running and the PILATES Python environment active, run:

```sh
python -m scripts.run_impacts_live_smoke --impacts-image ghcr.io/lbnl-ucb-sti/impacts:v0.2.5
```

On Apple Silicon, use an ARM64 IMPACTS build with raw HDF5 support. The
published AMD64 v0.2.5 image stalled during DuckDB import under Docker Desktop
emulation.

The command runs IMPACTS presim, one BEAM iteration, and IMPACTS postsim using
the bundled San Francisco County fixture at `tests/fixtures/impacts_live_sf/`.
It uses `ghcr.io/lbnl-ucb-sti/beam:1.1.2` from the BREATHE scenario settings
and runs BEAM with a 3 GB Java heap. Each run prints its output path and keeps the
workspace and `run.log` in a new folder under `test-output/impacts-live-sf/`,
including after a test failure. Once the images are available, the containers
run with networking disabled. The command fails if Docker is unavailable.
The bridge and derived settings for each year are saved together under the
IMPACTS output directory at `settings/<run-name>_y<year>/`.

Before BEAM starts, the test checks that presim's manifests describe the
expected run and year, that all source vehicle and tour IDs are preserved, and
that each assigned vehicle type has a readable rate file. It compares the
generated fleet and rates with BEAM's launch files and compares postsim's
inputs byte for byte with the live BEAM network and skims. It then requires
known vehicle types and matching county emissions keys, reconciles annualized
pollutant mass across county, InMAP, and AERMOD allocations, and verifies
concentrations and population exposure.

The smoke test verifies that postsim reads the exact PBF selected by BEAM and
records its Consist artifact ID. PILATES requires the matching artifact. In
the live smoke test, the PBF is available at
`/app/workspace/beam/output/<region>/year-<year>-iteration-<iteration>/.pilates-consist-outputs/beam_run/beam_r5_osm_file_<year>_<iteration>.osm.pbf`.

R5 builds from one street network file. PILATES selects the first PBF by
filename in the configured R5 directory and logs any VEX files it excludes
from the launch copy. PILATES allows a VEX-only directory for BEAM-only runs
and rejects it before IMPACTS presim.

## BREATHE deployment

Enabling IMPACTS in `run.models` runs both EMFAC activities and fleet
generation in presim. PILATES binds the generated passenger and freight fleet
to BEAM through Consist. With the pinned IMPACTS v0.2.5 image, presim writes
`activities_manifest.yaml` and `fleet_manifest.yaml`. PILATES gives postsim
the fleet manifest path in `impacts.presim_fleet_manifest` alongside the final
BEAM network, OSM PBF, and Parquet emissions skims.
BREATHE specifies its source ATLAS vehicles file through
`impacts.overrides.fleet.atlas.vehicles_file` and uses the supplied 2018 EMFAC
archive. In the SF Bay production data, `impacts/emissions/` holds the raw EMFAC
archive, fleet assignment rules, and vehicle-category catalog.
`impacts/dispersion/` holds the InMAP grid, NOₓ-to-NO₂ matrix, and AERMOD patterns.
The scenario resolves these paths under `/app/beam_input/sfbay`. Deploy the updated
production data together with the scenario settings. IMPACTS v0.2.5 omits the
explicit vehicle-category path when writing derived settings, so the production
data provides a relative link at `vehicle-tech/emissions/emissions_vehicle_categories.csv`.
Keep that link until the deployed IMPACTS image preserves the configured path.
The optional `impacts/baseline/beam_concentration_distribution.parquet` is a
saved IMPACTS result for delta comparisons and requires an explicit analysis
setting.

The BREATHE 2018 Census bundle is distributed under `impacts/census/`. To use it
offline, unpack the versioned archive into the shared cache directory mounted
at `/app/impacts_cache`. The UrbanSim HDF5 and the ISRM Zarr use their existing
inputs and mounts.

PILATES archives the presim activities, generated fleet, and postsim results,
including the emissions, concentration, and exposure tables. IMPACTS postsim runs
independently of general postprocessing. Configure `IMPACTS_INMAP_DATA_DIR` on the
machine that runs the container.

Consist can reuse BEAM results when its inputs match, including the generated
fleet files. Fresh IMPACTS steps still run because some sources
can be downloaded at runtime.

The launcher sets `LD_LIBRARY_PATH=/opt/conda/lib` inside the v0.2.0 image so
postsim can load the C++ runtime bundled with its SQLite dependencies.
It also gives DuckDB writable temporary mounts beside the two fleet assignment
files. The assignment files themselves stay read only.
