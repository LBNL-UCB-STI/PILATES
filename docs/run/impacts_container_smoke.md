# IMPACTS container smoke test

From the PILATES repository root, with Docker running and the PILATES Python
environment active, run:

```sh
python -m scripts.run_impacts_container_smoke
```

The test uses the bundled fixture in `tests/fixtures/impacts_smoke/`, pulls
IMPACTS v0.2.3 if needed, and runs the containers with networking disabled.
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

In v0.2.0, presim fleet generation fails with `KeyError: 'model_year_groups'`.
The default command exits with a failure. To verify the raw activities and the
complete postsim replay while that release is fixed, run:

```sh
python -m scripts.run_impacts_container_smoke --replay-without-fleet
```

On an amd64 Linux host, pass `--image ghcr.io/lbnl-ucb-sti/impacts:v0.2.3` to
run the replay with fleet generation using the published fix.
The replay uses archived BEAM outputs. A live handoff check uses skims from a
BEAM run launched with the generated fleet. Optional baseline delta plots
require a second scenario.

## Optional live test in San Francisco

On amd64 Linux, with Docker running and the PILATES Python environment active,
run:

```sh
python -m scripts.run_impacts_live_smoke --impacts-image ghcr.io/lbnl-ucb-sti/impacts:v0.2.3
```

The example uses the public IMPACTS v0.2.3 image. The same command accepts an
alternative fleet-capable image by tag or digest. On the Apple Silicon Mac used
for local validation, DuckDB 2.0.1 in the public amd64 image fails under Docker
Desktop's Rosetta emulation. The local `pilates-impacts:v0.2.1` build with
DuckDB 1.5.6 passed the test on that Mac.
On that Mac, run:

```sh
conda run --no-capture-output -n pilates-macos \
  python -m scripts.run_impacts_live_smoke --impacts-image pilates-impacts:v0.2.1
```

The command runs IMPACTS presim, one BEAM iteration, and IMPACTS postsim using
the bundled San Francisco County fixture at `tests/fixtures/impacts_live_sf/`.
It uses a 3 GB BEAM Java heap. Each run prints its output path and keeps the
workspace and `run.log` in a new folder under `test-output/impacts-live-sf/`,
including after a test failure. Once the images are available, the containers
run with networking disabled. The command fails if Docker is unavailable.

Before BEAM starts, the test checks that presim's manifests describe the
expected run and year, that all source vehicle and tour IDs are preserved, and
that each assigned vehicle type has a readable rate file. It compares the
generated fleet and rates with BEAM's launch files and compares postsim's
inputs byte for byte with the live BEAM network and skims. It then requires
known vehicle types and matching county emissions keys, reconciles annualized
pollutant mass across county, InMAP, and AERMOD allocations, and verifies
concentrations and population exposure.

With an IMPACTS image that supports explicit OSM input, add
`--use-beam-osm-artifact` to verify that postsim reads the exact PBF selected
by BEAM and records its Consist artifact ID. For scenario runs using such an
image, set `impacts.use_beam_osm_artifact: true`. PILATES then requires the
matching artifact. In the live smoke test, the PBF is available at
`/app/workspace/beam/output/<region>/year-<year>-iteration-<iteration>/.pilates-consist-outputs/beam_run/beam_r5_osm_file_<year>_<iteration>.osm.pbf`.

## BREATHE deployment

For a live BREATHE run, use an IMPACTS image that implements
`fleet_manifest.yaml` and set `impacts.use_generated_fleet: true`. Check that
`impacts.overrides.fleet.atlas.vehicles_file` identifies the source ATLAS vehicle
table if it has a different name from `vehicles.parquet`. The BREATHE ATLAS
table includes the required source columns in
`urbansim/atlas-2019/vehicles--2019-Baseline--EM.parquet`. Run the 2018 interval
with the supplied EMFAC archive, which contains 2018 data. PILATES archives the
presim activities and generated fleet, then binds the final BEAM network and
Parquet emissions skims to postsim. It also archives the postsim results,
including the emissions, concentration, and exposure tables. IMPACTS postsim runs
independently of general postprocessing. Configure `IMPACTS_INMAP_DATA_DIR` on the
machine that runs the container.

IMPACTS v0.2.0 writes `dispersion_completed: false` in the final exposure
manifest even when both dispersion stages ran. The replay verifies the
referenced concentration tables and their values directly.

The launcher sets `LD_LIBRARY_PATH=/opt/conda/lib` inside the v0.2.0 image so
postsim can load the C++ runtime bundled with its SQLite dependencies.
It also gives DuckDB writable temporary mounts beside the two fleet assignment
files. The assignment files themselves stay read only.
