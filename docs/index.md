---
hide:
  - navigation
  - title
  - toc
---

<section id="home-page" class="home-hero">
  <div class="home-hero__inner">
    <p class="home-hero__eyebrow">PILATES documentation</p>
    <h1>Coupled land-use and transportation workflow docs.</h1>
    <p class="tagline">For analysts, model developers, and policy teams working with UrbanSim, ATLAS, ActivitySim, BEAM, and archived Consist-backed runs.</p>
    <div class="home-actions">
      <a class="md-button md-button--primary" href="start-here/getting_started/">Run a scenario</a>
      <a class="md-button" href="run/troubleshooting/">Diagnose a run</a>
      <a class="md-button" href="extend/adding_a_model/">Add a model</a>
    </div>
    <dl class="home-summary">
      <div class="home-summary__item">
        <dt>Audience</dt>
        <dd>Academic researchers, public agencies, model maintainers, and software engineers</dd>
      </div>
      <div class="home-summary__item">
        <dt>Purpose</dt>
        <dd>Run scenarios, understand workflow handoffs, debug restarts, and extend model boundaries</dd>
      </div>
      <div class="home-summary__item">
        <dt>Runtime</dt>
        <dd>Layered stages, typed step outputs, explicit artifact keys, and replay-first restart semantics</dd>
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

PILATES is a workflow runtime for coupled regional simulation. It keeps model
adapters decoupled, coordinates their run order across years and inner
iterations, and makes workflow boundaries explicit through typed outputs,
artifact keys, and a Consist-backed execution contract.

The current runtime mental model is intentionally simple:

- the launcher prepares runtime state, storage, bootstrap, and scenario context
- the enabled workflow surface projects the active run shape from settings plus state
- stages decide ordering and loop structure
- step factories publish typed outputs and coupler-visible artifacts

Use these docs to:

- run local and HPC scenarios under one runtime model
- understand what each stage and artifact means logically, not just where code lives
- extend the workflow with new model integrations and explicit step contracts
- reopen archived runs for SQL, datasets, runset comparisons, and scenario analysis
- decide whether a stopped run needs config repair, data repair, restart, or deeper model debugging

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
