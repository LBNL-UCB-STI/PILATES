---
hide:
  - navigation
  - title
  - toc
---

<section id="home-page" class="home-hero">
  <div class="home-hero__inner">
    <p class="home-hero__eyebrow">PILATES documentation</p>
    <h1>Notes on running PILATES.</h1>
    <p class="tagline">PILATES couples UrbanSim, ATLAS, ActivitySim, and BEAM into a single multi-year run. These docs cover how to use it, how it is put together, and how to read the runs it leaves behind.</p>
    <div class="home-actions">
      <a class="md-button md-button--primary" href="start-here/getting_started/">Run a scenario</a>
      <a class="md-button" href="run/troubleshooting/">Diagnose a run</a>
      <a class="md-button" href="extend/adding_a_model/">Add a model</a>
    </div>
    <dl class="home-summary">
      <div class="home-summary__item">
        <dt>Who this is for</dt>
        <dd>Group members and collaborators working with the models, plus anyone curious about BEAM CORE and PILATES capabilities</dd>
      </div>
      <div class="home-summary__item">
        <dt>What it covers</dt>
        <dd>Running scenarios, following the workflow, restarting cleanly, extending models, and opening archived runs</dd>
      </div>
      <div class="home-summary__item">
        <dt>How it runs</dt>
        <dd>Stages and steps with typed outputs, named artifacts, and Consist-backed replay on restart</dd>
      </div>
    </dl>
  </div>
</section>

## Start With The Job In Front Of You

- **Run a scenario:** [Getting Started](start-here/getting_started.md), then [First Run Walkthrough](start-here/first_run_walkthrough.md).
- **Understand what happened:** [Scenario Lifecycle](run/scenario_lifecycle.md), then [Workflow Primer](workflow/workflow_primer.md).
- **Fix a stopped run:** [Troubleshooting](run/troubleshooting.md), then [Restart and Resume](run/restart_and_resume.md).
- **Add a model or step:** [Adding a Model](extend/adding_a_model.md), then [Model Integration Guide](extend/model_integration_guide.md).
- **Inspect archived outputs:** [Analysis Overview](analysis/overview.md), then [Opening Archives](analysis/opening_archives.md).
- **Understand Consist itself:** [Consist documentation](https://lbnl-ucb-sti.github.io/consist/latest/), then [Consist in PILATES](workflow/consist_in_pilates.md).

New contributors should use this order:

1. [Getting Started](start-here/getting_started.md) for local setup and a first runnable scenario.
2. [Architecture](workflow/architecture.md) and [Consist in PILATES](workflow/consist_in_pilates.md) for the Consist-first runtime model.
3. [Adding a Model](extend/adding_a_model.md) for the hands-on contributor path.

If you want the shortest first-run path, use this order:

1. [Getting Started](start-here/getting_started.md)
2. [First Run Walkthrough](start-here/first_run_walkthrough.md)
3. [CLI](run/cli.md)
4. [Configuration Reference](run/configuration_reference.md)
5. [Troubleshooting](run/troubleshooting.md)

!!! note

    These docs are intentionally layered. Start with the short run path first,
    then branch into workflow semantics, model extension, or archived-run
    analysis only when you need them.

## What PILATES Does

PILATES runs land-use and transportation models together over a sequence of
years. Each model still operates as normal; PILATES handles the order they run
in, the inputs they hand to each other, and the bookkeeping needed to restart
a run or open an old one later.

The runtime is intentionally small:

- the launcher sets up state, storage, bootstrap, and the scenario context
- the enabled workflow surface decides which stages and steps are active for
  the current run
- stages handle ordering and the inner-iteration loop
- step factories publish typed outputs that the next step can read by name

These docs cover the things that come up day-to-day:

- running scenarios locally or on the cluster with the same launcher
- following what each stage and artifact actually means, not just where its
  code lives
- adding a new model or step without breaking the existing contracts
- opening archived runs for SQL, datasets, and scenario comparisons
- working out whether a stopped run needs a config tweak, a data fix, a
  restart, or a closer look at the model itself

## What We Use It For

PILATES couples land-use, vehicle ownership, activity-based travel demand, and
network assignment so they can co-evolve over multi-decade horizons. The kinds
of questions we use it for include:

- how population, employment, vehicle fleet, and travel behavior shift
  together under different policy futures
- electrification and fuel-price scenarios with feedback into household
  ownership and activity patterns
- accessibility and equity outcomes across zone systems and demographic
  groups
- joint calibration of land-use, demand, and assignment models against
  regional observations
- counterfactuals on fleet composition, pricing, infrastructure, and
  land-use policy

Current scenarios cover the SF Bay Area and the Seattle region. The runtime is
region-agnostic, so adding a new MPO is a configuration and data exercise
rather than a code rewrite.

## Why Consist

PILATES runs on top of [Consist](https://lbnl-ucb-sti.github.io/consist/latest/),
an open-source cache-aware run tracker. Consist gives scenarios, steps, and
artifacts durable identity, records lineage to a queryable database, and lets a
step be skipped when its inputs match an earlier execution. That is the main
reason PILATES is more than shell-script orchestration: multi-year land-use and
transport coupling runs are long and expensive, so cache-aware restarts and
archived-state recovery are part of the research method, not just operational
plumbing. In practice, restarts pick up from archived state instead of being
rebuilt by hand, repeated scenarios reuse earlier outputs where the identity
matches, and old runs can be opened later without reconstructing what they were.
See [Consist in PILATES](workflow/consist_in_pilates.md) for the boundary
between what Consist owns and what PILATES owns.

## Choose A Reading Path

=== "Running a scenario"

    - [Getting Started](start-here/getting_started.md)
    - [First Run Walkthrough](start-here/first_run_walkthrough.md)
    - [CLI](run/cli.md)
    - [Configuration Reference](run/configuration_reference.md)
    - [Troubleshooting](run/troubleshooting.md)

=== "Understanding the workflow"

    - [Scenario Lifecycle](run/scenario_lifecycle.md)
    - [Workflow Primer](workflow/workflow_primer.md)
    - [Architecture](workflow/architecture.md)
    - [Stages and Steps](workflow/stages_and_steps.md)
    - [Model Boundaries](reference/model_boundaries.md)
    - [Simulation Logic by Stage](workflow/simulation_logic_by_stage.md)
    - [Artifact Flow](workflow/artifact_flow.md)
    - [Consist in PILATES](workflow/consist_in_pilates.md)
    - [Consist documentation](https://lbnl-ucb-sti.github.io/consist/latest/)

=== "Extending PILATES"

    - [Adding a Model](extend/adding_a_model.md)
    - [Model Integration Guide](extend/model_integration_guide.md)
    - [Model Contract Checklist](extend/model_contract_checklist.md)
    - [Step Contracts](workflow/step_contracts.md)
    - [Model Boundaries](reference/model_boundaries.md)
    - [Output Validation](extend/output_validation.md)

=== "Analyzing archived runs"

    - [Analysis Overview](analysis/overview.md)
    - [Consist in Action](analysis/consist_in_action.md)
    - [Opening Archives](analysis/opening_archives.md)
    - [Consist Analysis CLI](analysis/consist_analysis_cli.md)
    - [Run Discovery and Runsets](analysis/run_discovery_and_runsets.md)
    - [SQL and DuckDB](analysis/sql_and_duckdb.md)

=== "Running on Lawrencium"

    - [Scenario Lifecycle](run/scenario_lifecycle.md)
    - [HPC Overview](run/hpc_overview.md)
    - [Lawrencium](run/lawrencium.md)
    - [Restart and Resume](run/restart_and_resume.md)
