# IMPACTS container smoke test

From the PILATES repository root, with Docker running and the PILATES Python
environment active, run:

```sh
python -m scripts.run_impacts_container_smoke
```

The test uses the bundled fixture in `tests/fixtures/impacts_smoke/`, pulls
IMPACTS v0.2.0 if needed, and runs the containers with networking disabled.
Every run starts with a fresh temporary workspace.

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

Pass `--image` to verify a fixed candidate before a versioned image is published.
The replay uses archived BEAM outputs. A live handoff check uses skims from a
BEAM run launched with the generated fleet. Optional baseline delta plots
require a second scenario.

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
