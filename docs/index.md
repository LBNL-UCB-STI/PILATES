---
hide:
  - navigation
  - title
  - toc
---

<section id="home-page" class="home-hero">
  <div class="home-hero__inner">
    <p class="home-hero__eyebrow">PILATES documentation</p>
    <h1>Multi-model co-simulation for land use, fleet, and transportation.</h1>
    <p class="tagline">PILATES orchestrates multi-year simulations across land-use, demographic, vehicle-ownership, activity-demand, and network-assignment models—managing execution order, cross-model data contracts, receiving-side admission checks, and Consist-backed provenance.</p>
    <div class="home-actions">
      <a class="md-button md-button--primary" href="start-here/getting_started/">Run a scenario</a>
      <a class="md-button" href="extend/adding_a_model/">Add or integrate a model</a>
      <a class="md-button" href="run/troubleshooting/">Diagnose a run</a>
    </div>
    <dl class="home-summary">
      <div class="home-summary__item">
        <dt>Who this is for</dt>
        <dd>Researchers and multi-lab collaborators running coupled scenarios, integrating models (UrbanSim, ATLAS, ActivitySim, BEAM, DEMOS, SynthFirm, POLARIS), or analyzing archived runs</dd>
      </div>
      <div class="home-summary__item">
        <dt>What it covers</dt>
        <dd>Running scenarios locally or on HPC, restarting from archived state, adding or swapping models via typed contracts, and querying outputs</dd>
      </div>
      <div class="home-summary__item">
        <dt>How it runs</dt>
        <dd>Stage-based orchestration on top of Consist, with explicit input/output contracts, preflight admission checks, and DuckDB provenance</dd>
      </div>
    </dl>
  </div>
</section>

## Start With Your Task

- **Run a scenario:** [Getting Started](start-here/getting_started.md) → [Configuration Basics](start-here/configuration_basics.md) → [CLI](run/cli.md) & [Configuration Reference](run/configuration_reference.md)
- **Integrate or add a model:** [Adding a Model](extend/adding_a_model.md) → [Step Contracts](workflow/step_contracts.md) & [Model Boundaries](reference/model_boundaries.md)
- **Understand the workflow:** [Architecture](workflow/architecture.md) → [Artifact Flow](workflow/artifact_flow.md) → [Lineage Map](workflow/lineage_map.md)
- **Fix or resume a stopped run:** [Troubleshooting](run/troubleshooting.md) → [Scenario Lifecycle & Restart](run/scenario_lifecycle.md)
- **Query archived outputs:** [Analysis Overview](analysis/overview.md) → [Scenario Comparison](analysis/scenario_comparison.md) → [SQL and DuckDB](analysis/sql_and_duckdb.md)
- **Understand Consist:** [Step Contracts](workflow/step_contracts.md) → [Consist documentation](https://lbnl-ucb-sti.github.io/consist/latest/)

## What PILATES Does

PILATES couples land-use, household demographics, vehicle ownership, travel
demand, and network assignment over multi-year horizons. Each component model
runs in its own container or Python environment; PILATES manages step ordering
and timing, translates tables across model boundaries, validates inputs before
expensive steps launch, and tracks provenance so runs can be resumed or
queried later.

At runtime:

- **Bootstrap & Admission:** Resolves scenario YAML inheritance (`base:`),
  stages initial datasets, and runs receiving-side admission checks (digest and
  provenance verification) on external or prior-run inputs.
- **Stage & Loop Orchestration:** Steps through forecast years and inner
  supply-demand iterations (for example, `urbansim` → `atlas` →
  `activitysim` ↔ `beam`, or custom model combinations configured in
  `run.models`).
- **Contract-Driven Steps:** Each model step (`preprocess`, `run`,
  `postprocess`) resolves its declared inputs from the Consist coupler,
  executes inside its container or environment, publishes typed outputs back to
  the coupler, and archives durable outputs.

Because stages communicate strictly through named coupler roles and typed step
contracts rather than hardcoded file paths, individual models can be disabled,
swapped, or fed via pre-admitted external handoffs without rewriting the
orchestrator.

## What We Use It For

Coupling land-use, fleet, demand, and network assignment lets long-term and
short-term choices feed back into one another:

- **Longitudinal policy & infrastructure scenarios:** How population,
  employment, commercial and household fleets, and travel behavior co-evolve
  under pricing, transit, automation, or land-use interventions.
- **Electrification & energy analysis:** Household and commercial vehicle
  adoption, charging, and fuel-price feedback into daily activity and mode
  choice.
- **Accessibility & equity:** Travel times, multi-period skims, and welfare
  metrics across zone systems and demographic groups.
- **Cross-model calibration & comparison:** Aligning land-use, demand, and
  assignment outputs against regional observations across MPOs.

## Why Consist

Multi-year coupled simulations are expensive and often span multiple runs,
clusters, or modeling teams. PILATES runs on top of
[Consist](https://lbnl-ucb-sti.github.io/consist/latest/), an open-source,
cache-aware provenance and workflow tracker backed by DuckDB:

- **Deterministic restarts:** Interrupted runs resume from archived step
  outputs and DuckDB coupler snapshots instead of starting over or relying on
  manual file surgery.
- **Verified cross-run & cross-team handoffs:** External inputs (such as
  pre-computed skims, warm-start linkstats, or upstream model tables) are
  admitted against cryptographic digests and provenance sidecars before a step
  executes.
- **Cache reuse across scenarios:** Branching a scenario or re-running with a
  downstream parameter change reuses upstream steps whose inputs and
  configuration have not changed.
- **Queryable archives:** Every step, configuration facet, and output artifact
  is indexed in DuckDB (`provenance.duckdb`) for inspection via the `consist`
  CLI or `pilates.analysis` Ibis queries.

See [Step Contracts](workflow/step_contracts.md) for how PILATES and Consist
divide responsibilities.

## Reading Paths by Role

=== "Running a scenario"

    - [Getting Started](start-here/getting_started.md)
    - [Configuration Basics](start-here/configuration_basics.md)
    - [CLI](run/cli.md)
    - [Configuration Reference](run/configuration_reference.md)
    - [Scenario Lifecycle & Restart](run/scenario_lifecycle.md)
    - [Troubleshooting](run/troubleshooting.md)

=== "Understanding the workflow"

    - [Architecture](workflow/architecture.md)
    - [Step Contracts](workflow/step_contracts.md)
    - [Model Boundaries](reference/model_boundaries.md)
    - [Artifact Flow](workflow/artifact_flow.md)
    - [Lineage Map](workflow/lineage_map.md)
    - [Zone & Skim Alignment](reference/zone_and_skim_alignment.md)
    - [Consist documentation](https://lbnl-ucb-sti.github.io/consist/latest/)

=== "Extending PILATES"

    - [Adding a Model](extend/adding_a_model.md)
    - [Step Contracts](workflow/step_contracts.md)
    - [Model Boundaries](reference/model_boundaries.md)
    - [Output Validation](extend/output_validation.md)

=== "Analyzing archived runs"

    - [Analysis Overview](analysis/overview.md)
    - [Scenario Comparison](analysis/scenario_comparison.md)
    - [SQL and DuckDB](analysis/sql_and_duckdb.md)
    - [Database & Schemas](reference/database_and_schemas.md)

=== "Running on HPC & Lawrencium"

    - [HPC Overview](run/hpc_overview.md)
    - [Lawrencium](run/lawrencium.md)
    - [Run Promotion](run/run_promotion.md)
    - [Scenario Lifecycle & Restart](run/scenario_lifecycle.md)
