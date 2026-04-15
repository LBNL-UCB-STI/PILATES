---
title: Troubleshooting
summary: Common startup, runtime, restart, data, and HPC failures in the current PILATES runtime.
---

# Troubleshooting

## Startup And Install

- If `python run.py` fails before the workflow starts, check that the config path passed with `-c/--config` exists and parses.
- If you pass `-S/--stage`, the loader checks that the stage file exists before it loads the config.
- If the config loader raises on `atlas.beamac`, check the region and zone type. The loader only allows `atlas.beamac > 0` for SF Bay with `skims_zone_type = taz`.
- If the config loader raises on `end_year`, check that `end_year >= start_year`.

## Runtime And Restart

- If restart startup complains about missing files, check the archive run directory and the local workspace roots that the launcher resolved.
- If `restart_strict` is true, PILATES can fail after bootstrap when required restart artifacts are still missing.
- If you expected rewind-style resume, make sure `--allow-rewind-resume` was passed on the command line.
- If the launcher prints a restart command on failure, use that command as the next recovery step.

## Data Layout

- If ActivitySim bootstrap complains about missing config directories, check the `configs`, `configs_extended`, `configs_mp`, and `configs_sh_compile` trees under the mutable ActivitySim config root.
- If BEAM restart preflight complains about missing files, check the region subdirectory and the primary BEAM config file under the mutable BEAM input root.
- If bootstrap or restart cannot find the expected UrbanSim or ATLAS inputs, check the paths in the active scenario template.

## HPC

- If `job_runner.sh` stops immediately, it usually means the Slurm account was not provided or the requested partition was invalid.
- If the job script cannot install dependencies, check `hpc/requirements-hpc.txt`, `PILATES_VENV_PATH`, and the module environment that `job.sh` loads.
- If `job.sh` cannot find `consist`, check whether `CONSIST_SRC_DIR` exists or whether the configured PyPI package is reachable.

## What This Page Does Not Guess

This page does not try to diagnose model-science issues or undocumented region-specific assumptions. Those belong in the workflow, model-integration, or analysis pages.

## Adjacent Pages

- Use [Getting Started](../start-here/getting_started.md) for the normal path.
- Use [Restart and Resume](restart_and_resume.md) for replay-specific behavior.
- Use [SQL and DuckDB](../analysis/sql_and_duckdb.md) for post-run DB inspection.
