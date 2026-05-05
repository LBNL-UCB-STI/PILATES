---
title: Adding a Model
summary: Hands-on checklist for wiring a new model or model phase into PILATES.
---

# Adding a Model

## Reading Path

- If you want the quickest implementation path, start on this page.
- If you want the architecture picture first, read [Model Integration Guide](model_integration_guide.md).
- Pair either path with [Model Contract Checklist](model_contract_checklist.md).
- Finish with [Testing New Integrations](testing_new_integrations.md).

## The Short Version

Adding a model is not just adding a container command. In the current runtime,
the durable workflow boundary is a tracked step with:

- a model-local adapter implementation under `pilates/<model>/`
- typed outputs that name the files the workflow can trust
- a catalog entry that declares inputs, outputs, dependencies, and stage metadata
- a step factory that binds the model adapter to the live coupler and output holder
- Consist step metadata for cache/facet identity and any model config adapter
- binding rules for runtime source precedence and restart/replay fallbacks
- one stage call site that decides when the step runs
- tests that prove the declared contract and runtime publications stay aligned

Keep that split intact. Model packages should know how to run model-specific
work. Workflow packages should know when the work runs, which keys it publishes,
and how later steps find those keys.

Read [Step Contracts](../workflow/step_contracts.md) for the contract layer and
[Artifact Semantics](../workflow/artifact_semantics.md) for the meaning of the
published keys.

## Practical Order

For most contributors, the least confusing order is:

1. start from `scripts/new_model_scaffold.py`
2. define the typed outputs and public workflow keys
3. add or update the catalog entry
4. add Consist metadata or config-reference policy when the step has cache
   identity beyond scalar settings
5. register binding behavior for any nontrivial input precedence or replay source
6. wire the step factory and holder publication
7. connect the stage that will call the step
8. add validation, guardrail coverage, and docs before calling the integration done

That order is intentional: the scaffold and the workflow catalog own the
contract surface, while model-local modules own implementation details.
If you are wondering "where do I add X?", use this split:

- model-local behavior lives under `pilates/<model>/`
- public workflow keys and typed outputs live in `pilates/<model>/outputs.py`
- execution metadata lives in `pilates/workflows/catalog.py`
- Consist step metadata and config adapters live in `pilates/workflows/step_consist_meta.py`
- runtime source precedence and fallback rules live in `pilates/workflows/binding.py`
- stage sequencing stays in `pilates/workflows/stages/`
- step shells stay in `pilates/workflows/steps/`

## Preferred Starting Point

Use `scripts/new_model_scaffold.py` first unless you are touching an existing model.

The scaffold gives you the current expected shape:

- typed outputs classes
- `WorkflowStepSpec` catalog entries
- `StandardStepSpec` / `build_standard_step()` workflow step shells
- `SCHEMA_STEP_BUILDERS` registration
- stage snippets built around `WorkflowRuntimeContext`, `StageRunner`, and `build_binding_plan(...)`
- a checklist artifact under `docs/checklists/`

That path is intentionally more declarative than wiring a model by hand across multiple modules.

Run it before copying an existing model. Older integrations can contain
historical patterns that still work but are not the clearest path for new code.
The scaffold exists to encode the current step-contract shape.

## What Each File Is For

| File or package | Purpose | Common mistake |
| --- | --- | --- |
| `pilates/<model>/preprocessor.py`, `runner.py`, `postprocessor.py` | Model-specific execution and model-local logging/recovery callbacks. | Rebuilding workflow enablement or restart policy here. |
| `pilates/<model>/outputs.py` | Typed outputs and validation for workflow-visible products. | Returning loose dictionaries for files that downstream steps consume. |
| `pilates/generic/model_factory.py` | Registry from model name or phase alias to concrete components. | Adding a step factory but forgetting component dispatch. |
| `pilates/workflows/catalog.py` | Static contract metadata for step identity, keys, dependencies, and stage placement. | Letting runtime behavior drift from declared keys. |
| `pilates/workflows/step_consist_meta.py` | Lazy Consist step metadata: config adapters, scalar cache identity, facets, and identity inputs. | Logging config directories as artifacts when a config adapter should own fingerprinting. |
| `pilates/workflows/binding.py` | Binding rules, source precedence, and fallback policy for step inputs. | Repeating source-precedence chains in stages or model modules. |
| `pilates/workflows/steps/<model>.py` | Step factory that binds the live coupler, holder, model component, and typed output class. | Putting stage-loop decisions inside the step factory. |
| `pilates/workflows/stages/` | Stage orchestration: ordering, iteration, state updates, and calls to step refs. | Doing ad hoc input precedence instead of using binding. |
| `tests/` | Contract, publication, restart, and stage-boundary coverage. | Testing only that a component runs, not that the workflow can consume its outputs. |

## Architecture Rules for Contributors

- Enablement comes from the enabled workflow surface, not from reconstructing runtime flags or `WorkflowProfile` in model code.
- Runtime source precedence and fallback belong in binding.
- Stages own sequencing only; they should not become miniature input-resolution engines.
- Typed outputs own boundary validation and publication shape.
- Restart-sensitive contract decisions belong in workflow/runtime code, not model-local modules.
- The coupler is the current semantic-role registry between steps. It is not a
  historical version store.
- Snapshot artifacts record semantic facts at model boundaries. Archive and
  recovery roots record byte locations for existing artifacts.

## Step Versus Stage

Use this rule when deciding where code belongs:

- Put code in a **stage** when it decides ordering, looping, state progress, or which already-defined step runs next.
- Put code in a **step factory** when it turns one model phase into one executable Consist step with typed outputs.
- Put code in a **model module** when it prepares files, runs a container or Python model, postprocesses model-local outputs, or implements model-specific logging callbacks.
- Put code in **binding** when it chooses between coupler values, previous holder outputs, configured fallback paths, or restart/replay material.

If a new integration needs one special branch, first decide whether that branch
is workflow policy or model mechanics. Workflow policy should be visible in the
catalog, surface, binding, or stage code; model mechanics should stay in the
model package.

## Consist Metadata, Configs, and Artifacts

Every tracked step runs inside the Consist-backed workflow boundary. Step
factories get the common decoration through `build_standard_step()` and
`pilates/workflows/step_consist_meta.py`; add a model branch there when the
step needs a config adapter, scalar cache identity, query facets, or
`identity_inputs`.

Use config adapters for model configuration trees that should affect cache
identity. For BEAM-style configs with path references, declare reference
policies in the adapter metadata so Consist knows which paths are delegated to
workflow artifacts, ignored because they are dormant examples, or treated as
runtime outputs.

When choosing artifact keys, keep these concepts separate:

- a logged output is a normal step product that downstream code may inspect
- a snapshot artifact records a boundary state that restart/replay may need
- an archive or recovery root is only an alternate byte location for an
  existing artifact

Use [Artifact Semantics](../workflow/artifact_semantics.md) for the public key
meaning before adding new names. Add new keys only when the workflow needs a
new semantic role, not because a file was copied to a new storage location.

## Minimal Wiring Map

For a new model, the normal set of files to touch is:

- `pilates/<model>/outputs.py`
- `pilates/<model>/preprocessor.py`, `runner.py`, and/or `postprocessor.py`
- `pilates/generic/model_factory.py`
- `pilates/workflows/catalog.py`
- `pilates/workflows/step_consist_meta.py` if the step needs Consist config,
  facet, or identity-input metadata
- `pilates/workflows/binding.py` if the step needs binding rules beyond the
  catalog-derived defaults
- `pilates/workflows/steps/<model>.py`
- `pilates/workflows/steps/__init__.py`
- one stage module under `pilates/workflows/stages/`
- focused tests under `tests/`

If your change needs more than that, pause and check whether workflow policy is leaking into the model package.

## One-Hop Checklist

Use this quick checklist after the scaffold is in place:

1. Define the typed outputs and confirm the public artifact keys are explicit.
2. Register the model classes in `pilates/generic/model_factory.py`.
3. Add or update the `WorkflowStepSpec` entry in `pilates/workflows/catalog.py`.
4. Add `step_consist_meta.py` metadata for config adapters, facets, or
   identity inputs when needed.
5. Add `binding.py` rules for custom source precedence, optional fallbacks, or
   restart/replay material.
6. Build the step shell in `pilates/workflows/steps/<model>.py` and export it from `pilates/workflows/steps/__init__.py`.
7. Wire exactly one stage boundary that calls the step.
8. Add focused contract tests for the stage boundary and any restart-sensitive outputs.
9. Update the relevant docs page so the new boundary is discoverable without code spelunking.

## Practical Validation Targets

Use focused tests before trying a full production run:

- `tests/test_generic_modules.py` for `ModelFactory` dispatch and generated component signatures.
- `tests/test_standard_step_builder.py` for shared step-shell behavior.
- `tests/test_step_contract_validator.py` and `tests/test_workflow_catalog_contracts.py` for catalog drift.
- `tests/test_workflow_binding.py` for input source precedence and fallback behavior.
- `tests/test_workflow_step_metadata.py` and model-specific config adapter tests
  for Consist cache/facet metadata.
- `tests/test_coupler_key_invariants.py` for public coupler-key alignment.
- A stage-specific test near the stage you changed for ordering, state progress, and holder publication.

Production-scale model runs are useful later, but they are not the first line
of defense for integration mistakes. The developer contract is smaller than a
full scenario: declared keys, typed outputs, holder publication, coupler
publication, and stage placement.
