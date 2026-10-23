---
title: Troubleshooting
summary: Diagnostic triage for startup, configuration, bootstrap, restart, and HPC cluster failures.
---

# Troubleshooting

Inspect the earliest exception in the console or Slurm log first. Downstream model errors often stem from a configuration validation failure, a missing bootstrap input, or an incomplete workspace on restart.

## Startup and Configuration

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError` before config validation | The path passed to `-c/--config` or `-S/--stage` does not exist, or a `base:` parent path cannot be resolved. | Verify the `-c` and `-S` paths relative to your working directory and run `python -m pilates.config.models <config.yaml> --validate`. |
| `ValueError` on `base:` inheritance | Circular `base:` reference chain or missing parent YAML file. | Inspect the `base:` chain in your scenario overlay and ensure each parent file exists. |
| Validation error on `run.end_year` | `run.end_year` is less than `run.start_year`. | Set `run.end_year >= run.start_year` in your scenario YAML. |
| Validation error on `atlas.beamac` | `atlas.beamac > 0` is configured outside an SF Bay TAZ run. | Set `atlas.beamac: 0`, or confirm `run.region: sfbay` and `shared.skims.skims_zone_type: taz`. |
| Validation error on missing model section | A model is enabled in `run.models`, but its top-level configuration section (`urbansim`, `atlas`, `activitysim`, or `beam`) is missing. | Add the required model section or set that model key to `null`/`false` under `run.models`. |
| Validation error on `consist_db_filename` | `run.consist_db_filename` contains directory separators (`/` or `\`). | Provide a basename only, such as `provenance.duckdb`. |

## Bootstrap and Data Layout

| Symptom | Cause | Fix |
| --- | --- | --- |
| Missing base HDF5 datastore during bootstrap | `pilates/urbansim/data/` lacks `custom_mpo_<region_id>_model_data.h5` (or the year-specific `.h5` file). Note that ActivitySim and ATLAS also stage this file when UrbanSim is disabled. | Place the regional HDF5 datastore in `urbansim.local_data_input_folder` before starting the run. |
| Missing ActivitySim config directories | One or more required subdirectories (`configs`, `configs_extended`, `configs_mp`, `configs_sh_compile`) are missing under `activitysim.local_configs_folder`. | Populate all four ActivitySim configuration trees in the source directory or check `activitysim.local_mutable_configs_folder`. |
| Missing canonical zones or initial skims | `activitysim.canonical_zones_source`, `shared.geography.zones.source`, or initial OMX skims are missing on disk. | Verify the zone source path and initial skims file under `pilates/activitysim/data/`. |
| Missing BEAM production inputs | The regional directory (`pilates/beam/production/<region>/`), primary `.conf` file, `r5/` router folder, or warmstart linkstats file is missing. | Check `beam.local_input_folder`, `beam.config_file`, and `beam.warmstart_linkstats` in your YAML file. |
| Inbound `admission` check failure | An input artifact failed `prior_run` or `declared_digest` validation under `*.admission` with `mode: strict`. | Verify that the input file matches the expected provenance digest, seed the run-local DB from the shared DB (`consist_db_seed_from_shared_on_start: true`), or switch `mode` to `warn` while diagnosing. |

## Runtime and Restart

| Symptom | Cause | Fix |
| --- | --- | --- |
| Rewind guardrail blocks restart | `run.start_year` in the YAML config is earlier than `current_year` in the archived `run_state.yaml`. | Pass `--allow-rewind-resume` on the command line if you intend to rewind to an earlier simulation year, or align `run.start_year` with the state file. |
| Restart fails after bootstrap with missing workspace files | `run.restart_strict: true` is enabled and local workspace files required by the next stage were purged or not staged. | Check `run.local_workspace_root` and `run.output_directory`, or disable `restart_strict` if bootstrap and Consist cache hydration can supply the required inputs. |
| Crash during `beam_postprocess` after `beam_run` finished | Postprocessing failed or hit walltime after `beam_run` committed its outputs to `archive_run_dir`. | Rerun with `-S <archive_run_dir>/run_state.yaml`. PILATES cache-hits `beam_run` at the `beam_run_completed -> beam_postprocess` boundary, hydrates outputs from `archive_run_dir` if scratch was wiped, and re-executes `beam_postprocess` into a fresh output store without mutating the input `zarr_skims`. See [Scenario Lifecycle](scenario_lifecycle.md#the-beam_run_completed-beam_postprocess-mid-stage-boundary). |
| Expected intermediate file missing after a Consist cache hit | The file is an undeclared model-local scratch file rather than a declared step output. | Only declared step outputs in `RunResult.outputs` are hydrated on cache hits. Declare the artifact on the step contract if downstream stages require it. |

## HPC and Slurm

| Symptom | Cause | Fix |
| --- | --- | --- |
| `hpc/job_runner.sh` exits before `sbatch` | Missing `-a/--account`, invalid `-p` partition name, or missing `-c` config file. | Pass `-a <slurm_account>` and a valid partition (`lr6`, `lr6_m192`, `lr7`, or `lr8`). |
| `hpc/job.sh` fails during environment bootstrap | Missing cluster modules (`gcc`, `proj`, `python`), broken `PILATES_VENV_PATH`, or unreachable `hpc/requirements-hpc.txt`. | Check `PILATES_DIR`, `PILATES_VENV_PATH`, and the `module load` block in `hpc/job.sh`. |
| `consist` import or install failure on compute node | `CONSIST_SRC_DIR` points to a missing path and compute nodes lack outbound network access to install `CONSIST_PYPI_PACKAGE`. | Pass `--editable-consist /abs/path/to/consist` to `hpc/job_runner.sh` or pre-populate the virtual environment on the login node before submitting. |
| Out-of-memory kill inside BEAM | JVM heap (`BEAM_MEMORY`) exceeds the Slurm allocation (`MEMORY_LIMIT_GB`) or the partition preset is too small for the sample size. | Pass `--high-mem` (`-H`), switch to `-p lr8`, or set explicit `MEMORY_LIMIT_GB` and `BEAM_MEMORY` environment variables before calling `hpc/job_runner.sh`. |

## Related Pages

- Read [Getting Started](../start-here/getting_started.md) for prerequisite and directory checks.
- See [Scenario Lifecycle](scenario_lifecycle.md) for startup, bootstrap, and restart mechanics.
- Consult [HPC Overview](hpc_overview.md) and [Lawrencium](lawrencium.md) for cluster configuration and Slurm settings.
