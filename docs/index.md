---
hide:
  - navigation
  - title
  - toc
---

<section id="home-page" class="home-hero">
  <div class="home-hero__inner">
    <p class="home-hero__eyebrow">Integrated simulation runtime</p>
    <h1>PILATES coordinates long-horizon land use and transportation workflows.</h1>
    <p class="tagline">Run UrbanSim, ATLAS, ActivitySim, and BEAM in a shared scenario lifecycle with Consist-backed replay, provenance, cache reuse, and post-run analysis over archived artifacts.</p>
    <div class="home-actions">
      <a class="md-button md-button--primary" href="start-here/getting_started.md">Get started</a>
      <a class="md-button" href="workflow/workflow_primer.md">See the workflow</a>
    </div>
    <dl class="home-summary">
      <div class="home-summary__item">
        <dt>Model stack</dt>
        <dd>UrbanSim, ATLAS, ActivitySim, and BEAM</dd>
      </div>
      <div class="home-summary__item">
        <dt>Execution style</dt>
        <dd>Layered workflow runtime with explicit contracts and replay-first restart semantics</dd>
      </div>
      <div class="home-summary__item">
        <dt>Best for</dt>
        <dd>Regional forecasting, scenario analysis, model integration, and archived-run analysis</dd>
      </div>
    </dl>
  </div>
</section>

## Start Here

Follow this path if you want a successful first local run:

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
iterations, and makes the workflow boundary explicit through typed outputs,
artifact keys, and a Consist-backed execution contract.

The current runtime mental model is intentionally simple:

- the launcher initializes runtime flags and workflow state
- the enabled workflow surface projects the active run shape
- stages and step factories execute against that shared projection

It helps you:

- run local and HPC scenarios under one runtime model
- understand what each stage and artifact means logically, not just where code lives
- extend the workflow with new model integrations and explicit step contracts
- reopen archived runs for SQL, datasets, runset comparisons, and scenario analysis
- separate public current-state docs from internal design history and migration notes

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
