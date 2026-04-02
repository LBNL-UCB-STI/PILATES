# Troubleshooting

This is the first stop for common user and contributor failures. Each entry is
structured as:

- Symptom
- Likely cause
- How to confirm
- Fix

When in doubt, capture the failing command, the config file path passed via
`-c`, and any restart state file passed via `-S`.

## Installation Issues

### Import errors (`ModuleNotFoundError: ...`)

- Symptom: `ModuleNotFoundError: No module named 'pilates'` or missing runtime deps
  (for example `consist`, `geopandas`, `duckdb`).
- Likely cause: wrong environment activated, or running from outside the repo
  root without the repo on `PYTHONPATH`.
- How to confirm:
  - `which python` does not point at your intended env.
  - `python -c "import consist; import pilates"` fails.
- Fix:
  - Activate the intended environment and run from the repo root directory.
  - For the stub provenance test scripts, prefer the repo-root invocation that
    sets `PYTHONPATH` (see `run_stub_test_with_output.sh`).

### Container runtime not available / permission errors

- Symptom: errors like `docker: command not found`, `Cannot connect to the Docker daemon`,
  `permission denied`, or `singularity: command not found`.
- Likely cause: container runtime not installed/running, or your user lacks
  permissions to use it.
- How to confirm:
  - `docker ps` fails (Docker).
  - `singularity --version` fails (Singularity/Apptainer).
- Fix:
  - Install/enable the runtime and ensure it is running (Docker Desktop, system service, etc.).
  - If permissions are the issue, fix your group membership / runtime config.

## Configuration Issues

### Unknown CLI flags (`unrecognized arguments: ...`)

- Symptom: `error: unrecognized arguments: -v` (or similar).
- Likely cause: the runtime config loader parses only a small set of flags:
  `-c/--config`, `-S/--stage`, and `--allow-rewind-resume`.
- How to confirm:
  - See `parse_args_and_settings()` in `pilates/utils/io.py`.
- Fix:
  - Remove unsupported flags and retry.
  - Use `-c <settings.yaml>` to select a config and `-S <archive_state.yaml>` to restart.

### Config file or stage file missing

- Symptom:
  - `FileNotFoundError: Explicit stage file provided via -S/--stage does not exist: ...`
  - A settings YAML path passed via `-c` does not exist.
- Likely cause: wrong path, or running from a different directory than you expect.
- How to confirm:
  - The path in the error message does not exist on disk.
- Fix:
  - Pass an absolute path, or run from the repo root and pass the correct relative path.

### Pydantic validation errors / missing required fields

- Symptom: config load fails with a Pydantic validation error (missing fields,
  wrong types), or later code fails due to `None` where a required value is expected.
- Likely cause: `settings.yaml` does not match the expected nested config model.
- How to confirm:
  - The log prints `Loading config with Pydantic validation: ...` and then raises.
- Fix:
  - Start from an existing working config (for example `settings.yaml` or one of
    the `settings-*.yaml` files in the repo root) and apply minimal edits.

### Land use enabled with household sampling

- Symptom: ValueError containing:
  `Land use models must be disabled ... to use a non-zero household sample size`.
- Likely cause: `activitysim.household_sample_size > 0` while a land use model is enabled.
- How to confirm:
  - The exception is raised during settings parsing in `pilates/utils/io.py`.
- Fix:
  - Set `activitysim.household_sample_size: 0`, or disable land use.
  - Disabling land use can be done by clearing the land use model name or using
    warm-start mode (depending on your scenario configuration).

### ATLAS beamac constraint

- Symptom: ValueError containing:
  `atlas_beamac must be 0 ... unless region = sfbay and skims_zone_type = taz`.
- Likely cause: `atlas.beamac > 0` with an unsupported region/zone_type combination.
- How to confirm:
  - The exception is raised during settings parsing in `pilates/utils/io.py`.
- Fix:
  - Set `atlas.beamac: 0`, or switch to `run.region: sfbay` and
    `shared.geography.zones.zone_type: taz` if that is actually intended.

## Workflow, Cache, And Restart Issues

### “Run failed. Restart command:” in logs

- Symptom: the runtime prints a restart command and the run terminates.
- Likely cause: some step failed; the launcher prints a restart invocation that
  includes the config and (when available) the archive state file.
- How to confirm:
  - Look for `Run failed. Restart command:` in stdout logs.
- Fix:
  - Re-run using the printed command. It will look like:
    `python run.py -c <config> -S <archive_state_path>`.
  - If you are on HPC, the launcher also prints a `./hpc/job_runner.sh ...` form.

### Resume guardrail: attempting to resume earlier than the archive state

- Symptom: resume fails (or warns) when `-S` points to an archive state later
  than the year you’re attempting to run.
- Likely cause: safety guardrail to prevent accidental rewind-resume.
- How to confirm:
  - Search logs for “rewind” or “resume” messages around restart handling.
- Fix:
  - If you intend to rewind, pass `--allow-rewind-resume`.
  - Otherwise, choose the correct archive state file.

### Step prerequisite errors (“... must complete first”)

- Symptom: runtime raises errors like:
  - `UrbanSim preprocess must complete first`
  - `ActivitySim preprocess must complete first`
  - `BEAM run must complete first`
- Likely cause: the workflow is resuming into a state where upstream step outputs
  were not produced (or not reconstructed during restart hydration).
- How to confirm:
  - The exception message names the missing prerequisite step.
  - The expected output directory for that step is missing under the workspace.
- Fix:
  - If this is a restart (`-S`), confirm tracker-backed restart reconstruction
    is available and the run metadata is queryable.
  - If required local artifacts are missing, copy/restore them into the run
    workspace or restart from a point where bootstrap can re-initialize them.

### Unexpected cache hits / steps skipping work

- Symptom: a step returns quickly and appears to reuse outputs when you expected
  it to re-run.
- Likely cause: Consist cache hit due to unchanged identity inputs/config, or a
  “fast” hashing strategy missing a content change.
- How to confirm:
  - Logs mention cache hits or reconstruction/materialization during startup.
  - Your settings have the same `run.cache_epoch` and equivalent identity inputs.
- Fix:
  - Bump `run.cache_epoch` to invalidate prior caches for the scenario.
  - If you suspect metadata-only hashing is too permissive, switch to
    `run.consist_hashing_strategy: full` (slower but more robust).

### Consist tracker appears disabled when it should be enabled

- Symptom: runtime errors or logs mention Consist returned a noop tracker while
  enabled, or provenance artifacts are missing.
- Likely cause: Consist initialization failed due to DB path issues or API/version
  mismatch.
- How to confirm:
  - Search logs for Consist tracker initialization warnings/errors.
- Fix:
  - Ensure the run can create/write its Consist DB under the run output directory.
  - If you configured a shared DB path, ensure it is writable and consistent
    with your runtime mode (local-run DB vs shared DB).

## Data And Alignment Issues

### Required inputs missing or stale

- Symptom:
  - `FileNotFoundError: Required input '<key>' not found at <path>.`
  - `RuntimeError: Required input '<key>' appears stale at <path>.`
- Likely cause: input paths in config are wrong, files were not staged into the
  expected workspace, or you are pointing at an old/mismatched dataset.
- How to confirm:
  - The error message gives you the key and the resolved path.
- Fix:
  - Fix the configured path or stage the expected input to the referenced location.
  - For “stale” inputs, re-generate the input or point the run at the correct version.

### Canonical zone source file not found

- Symptom: `FileNotFoundError: Canonical zone source file not found. Tried: ...`
- Likely cause: `shared.geography.zones.source_file` (or its fallback candidates)
  does not exist on disk.
- How to confirm:
  - The exception lists all attempted candidate paths.
- Fix:
  - Update the configured canonical zone source path.
  - If you expect the zone source to be copied into the ActivitySim mutable
    config/data tree, confirm it exists under that workspace directory.

### Duplicate canonical zone IDs

- Symptom: `ValueError: Canonical ID column '<id_col>' contains duplicate values.`
- Likely cause: the zone source file’s ID column is not unique.
- How to confirm:
  - Inspect the zone source file and check the configured `canonical_id_col`.
- Fix:
  - Fix the source data (dedupe/correct IDs) or change `canonical_id_col` to the
    intended unique field.

### ActivitySim household/person consistency error

- Symptom: ValueError containing:
  `ActivitySim preprocess produced inconsistent household/person inputs: ...`
- Likely cause: upstream data filtering or joins produced households without
  persons, or persons pointing at missing households.
- How to confirm:
  - The error message includes counts and samples of problematic IDs.
- Fix:
  - Validate the upstream UrbanSim tables and any sampling/filtering behavior.
  - If you are using a non-default sampling configuration, re-check your
    ActivitySim preprocess assumptions.

### Skim Zarr indexing or “preprocessed” attribute issues

- Symptom: downstream models behave as if skim indices are misaligned, or a
  Zarr skim cache fails compatibility checks.
- Likely cause: skims are not 0-based contiguous, or the `preprocessed` marker is missing.
- How to confirm:
  - Inspect `otaz`/`dtaz` coordinate arrays and attributes in the Zarr store.
  - See the logic in `ensure_0_based_and_flag_zarr_skims()` in `pilates/utils/zone_utils.py`.
- Fix:
  - Re-run the workflow path that produces (or rewrites) the Zarr skims cache so
    it is normalized and flagged.
  - See `docs/zone_id_management.md` for the intended end-to-end behavior.

## Database And Documentation Generation Issues

### Consist DB file missing for a run

- Symptom: you cannot find the run’s Consist DB under the run output directory
  (often under a `.consist/` folder).
- Likely cause: shared database mode is disabled, or the run could not create
  the local DB directory due to permissions.
- How to confirm:
  - Check whether `shared.database.enabled` is set in your config.
  - Search the run output directory for `.consist` and `*.duckdb`.
- Fix:
  - Enable shared database mode and ensure the run output directory is writable.
  - If you are in local-run DB mode, ensure snapshot restore is configured if
    you expect restarts to recover the DB.

### “Export database docs” script fails with missing Python file

- Symptom: `./export_database_docs.sh ...` fails with something like:
  `python: can't open file 'pilates/utils/export_data_dictionary.py': [Errno 2] No such file or directory`
- Likely cause: `export_database_docs.sh` currently points at a CLI module that
  is not present in this repository layout.
- How to confirm:
  - The error message references `pilates/utils/export_data_dictionary.py`.
- Fix:
  - Use the preserved-test-output workflow to view schema docs today:
    `./run_stub_test_with_output.sh` then open `test_output/.../documentation/schema.html`.
  - For ERD generation from curated schemas (not from an arbitrary DB), run:
    `python pilates/database/scripts/generate_schema_erd.py --format html`

### ERD HTML opens but shows “Failed to load local Cytoscape asset”

- Symptom: the generated ERD HTML renders an error message about Cytoscape assets.
- Likely cause: the HTML expects `docs/diagrams/node_modules/...` relative to
  the output location, but those assets are not present.
- How to confirm:
  - The message is visible inside the HTML page.
- Fix:
  - Prefer the Mermaid output (`--format mermaid`) or Graphviz dot output (`--format dot`).
  - If you need the interactive HTML, ensure the expected assets are available
    at the referenced relative path.

## Related Docs

- `docs/getting_started.md`
- `docs/workflow_primer.md`
- `docs/zone_id_management.md`
- `docs/test_output_preservation.md`
