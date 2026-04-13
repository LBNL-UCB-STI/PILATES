---
hide:
  - navigation
  - title
  - toc
---

<section id="home-page" class="home-hero">
  <div class="home-hero__inner">
    <p class="home-hero__eyebrow">Integrated microsimulation platform</p>
    <h1>PILATES couples land use and transportation models into one reproducible workflow.</h1>
    <p class="tagline">Run UrbanSim, ActivitySim, BEAM, and ATLAS in a shared scenario loop with containerized execution, explicit state handoffs, and provenance-aware outputs.</p>
    <div class="home-actions">
      <a class="md-button md-button--primary" href="getting_started.md">Get started</a>
      <a class="md-button" href="workflow_primer.md">See the workflow</a>
    </div>
    <dl class="home-summary">
      <div class="home-summary__item">
        <dt>Model stack</dt>
        <dd>UrbanSim, ActivitySim, BEAM, and ATLAS</dd>
      </div>
      <div class="home-summary__item">
        <dt>Execution style</dt>
        <dd>Containerized steps with explicit state handoffs</dd>
      </div>
      <div class="home-summary__item">
        <dt>Best for</dt>
        <dd>Regional forecasting, scenario analysis, and coupled simulation experiments</dd>
      </div>
    </dl>
  </div>
</section>

## Start Here

Follow this path in order if you are new to PILATES:

1. [Getting Started](getting_started.md)
2. [CLI Reference](cli_reference.md)
3. [Configuration Reference](configuration_reference.md)
4. [Data Bootstrap](data_bootstrap.md)
5. [Workflow Primer](workflow_primer.md)

!!! note

    PILATES expects a container runtime, a configured Python environment, and
    region-specific input data. The [Getting Started](getting_started.md) guide
    covers the minimum local setup before the first scenario run.

## What PILATES Does

PILATES is a workflow runtime for long-horizon regional simulation. It keeps
model adapters decoupled, coordinates their run order, and manages the state
passed between them.

It helps you:

- compose different model stacks for short- and long-horizon scenarios
- keep run structure explicit through preprocess, runner, and postprocess stages
- run locally or on HPC with the same scenario and configuration framing
- trace outputs back to configs, artifacts, and execution context
- evolve database and data-loading paths without rewriting every model adapter

## Choose A Reading Path

=== "Running a scenario"

    - [Getting Started](getting_started.md)
    - [Configuration Reference](configuration_reference.md)
    - [CLI Reference](cli_reference.md)
    - [Data Bootstrap](data_bootstrap.md)
    - [Troubleshooting](troubleshooting.md)

=== "Understanding the workflow"

    - [Workflow Primer](workflow_primer.md)
    - [Architecture](architecture.md)
    - [Provenance and Lineage](provenance_and_lineage.md)
    - [Artifact Flow](artifact_flow.md)
    - [Lineage Map](lineage_map.md)

=== "Extending PILATES"

    - [Model Integration Guide](model_integration_guide.md)
    - [Adding a Model](adding_a_model.md)
    - [Database Setup](database-setup.md)
    - [Database Documentation Guide](database_documentation_guide.md)

## Common Follow-Up Tasks

| I want to... | Go to |
| --- | --- |
| run the current workflow locally | [Getting Started](getting_started.md) |
| understand how stages and steps fit together | [Workflow Primer](workflow_primer.md) |
| inspect the runtime architecture | [Architecture](architecture.md) |
| trace artifact handoffs and lineage | [Artifact Flow](artifact_flow.md) and [Lineage Map](lineage_map.md) |
| integrate or refactor a model adapter | [Model Integration Guide](model_integration_guide.md) |
| add a new step family | [Adding a Model](adding_a_model.md) |
| enable the run-local database and inspect schemas | [Database Setup](database-setup.md) |
| prepare region inputs and validate external data layout | [Data Bootstrap](data_bootstrap.md) |

## Documentation Status

These docs are substantially stronger than the earlier flat Markdown set, but
they still trail the live runtime in some places. When behavior matters, check
the code paths that own execution:

- `run.py`
- `pilates/runtime/launcher.py`
- `workflow_state.py`
- `pilates/workflows/stages/*.py`
- `pilates/workflows/steps/*.py`
- `pilates/workflows/catalog.py`
