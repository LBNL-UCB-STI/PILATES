# CLI Reference

This page documents the public command-line entry points that are currently
supported in this repository.

The main runtime CLI is `python run.py`. A small number of helper scripts are
also relevant for common workflows.

## Primary Entry Point

From the repo root:

```bash
python run.py
```

`run.py` is a thin entrypoint that delegates to the runtime launcher in
`pilates/runtime/launcher.py`.

## Supported `run.py` Arguments

The top-level CLI arguments are defined in `pilates/utils/io.py`.

### `-c`, `--config <path>`

Select the settings file to load.

Example:

```bash
python run.py -c scenarios/seattle/settings-seattle-newconfig-local.yaml
```

Default:

- `settings.yaml`

Notes:

- The config is loaded through `pilates.config.models.load_config(...)`.
- In practice, most users should pass an explicit scenario config rather than
  relying on the default.

### `-S`, `--stage <path>`

Resume or restart from an existing saved run-state file.

Example:

```bash
python run.py -c settings.local.yaml -S /path/to/run/run_state.yaml
```

Notes:

- This must point to an existing file.
- If the file does not exist, startup fails immediately with a `FileNotFoundError`.
- The saved state is used to reconstruct workflow progress and restart context.

### `--allow-rewind-resume`

Allow resuming from an earlier year than the archived run state would normally
permit.

Example:

```bash
python run.py -c settings.local.yaml -S /path/to/run/run_state.yaml --allow-rewind-resume
```

Use this only when you intentionally want to rewind resume behavior. The normal
default is to protect you from accidentally resuming from the wrong state.

## Common Invocation Patterns

### Fresh local run

```bash
python run.py -c settings.local.yaml
```

### Resume a prior run

```bash
python run.py -c settings.local.yaml -S /path/to/output/<run_name>/run_state.yaml
```

### Stub or smoke run

Stub behavior is controlled by config, not by a dedicated CLI flag.

If your config sets:

```yaml
run:
  use_stubs: true
```

then:

```bash
python run.py -c settings.stub.yaml
```

will skip real container execution for model runners that honor stub mode.

## What The CLI Does Not Currently Support

The current top-level parser does not expose a large flag surface.

In particular, there is no supported top-level flag for:

- verbose mode toggling
- disabling individual stages ad hoc
- setting output paths directly from the CLI
- toggling stub mode directly from the CLI

Those behaviors are controlled through the config file instead.

## Wrapper Scripts

### HPC submission wrapper

For Lawrencium-style HPC execution, use:

```bash
./hpc/job_runner.sh -c <settings.yaml> -a <slurm_account>
```

That wrapper handles Slurm submission, generated settings files, memory
templating, and calling `hpc/job.sh`.

See `docs/hpc_execution.md` for details.

### Database-doc export helper

The repo also contains:

```bash
./export_database_docs.sh /path/to/database.duckdb
```

Current caveat:

- this helper references `pilates/utils/export_data_dictionary.py`, which is not
  present in this checkout
- treat it as a partial or broken workflow until that export tool is restored

Use `docs/database_documentation_guide.md` for the currently supported schema
documentation path.

## Relationship Between CLI And Config

The CLI chooses:

- which config file to load
- whether to load a saved run state
- whether rewind-resume is allowed

The config file controls almost everything else, including:

- enabled models
- years and iteration counts
- output directories
- bootstrap/cache/restart settings
- container runtime and images
- database behavior

The practical rule is: CLI selects the run context; config defines the run.

## Failure Modes To Expect

- bad `-c` path:
  config load fails because the file cannot be found or parsed
- bad `-S` path:
  immediate `FileNotFoundError`
- incompatible config:
  Pydantic validation or early runtime checks fail before model execution

For general troubleshooting, see `docs/troubleshooting.md`.

## Related Docs

- `docs/getting_started.md`
- `docs/configuration_reference.md`
- `docs/hpc_execution.md`
- `docs/troubleshooting.md`
