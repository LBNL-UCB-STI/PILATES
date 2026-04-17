---
title: Getting Started
summary: Fast path from clone to a first successful PILATES run.
---

# Getting Started

## Read In Order

1. Read [First Run Walkthrough](first_run_walkthrough.md).
2. Use [Configuration Basics](configuration_basics.md) if the settings file shape is unclear.
3. Use [CLI](../run/cli.md) and [Configuration Reference](../run/configuration_reference.md) when you need the exact flags and fields.
4. Use [Data Bootstrap](data_bootstrap.md) if your region inputs are not in place yet.
5. Use [Troubleshooting](../run/troubleshooting.md) when a run stops early.

## If You Are New

For most readers, the shortest path is:

1. copy an active local scenario template
2. fix the machine-specific paths in that copy
3. run `python run.py -c <your-settings>.yaml`
4. switch to troubleshooting only if startup, bootstrap, or validation fails

## What To Expect

PILATES loads one YAML settings file, initializes the runtime flags from that file, restores or creates `WorkflowState`, and then builds the enabled workflow surface that the launcher uses to decide which stages and step contracts are active. The practical path is to copy an active scenario template, edit the file paths for your machine, and run `python run.py -c <your-settings>.yaml`.

## Start Points

- Local users usually begin from one of the active templates under `scenarios/`.
- HPC users should also read [HPC Overview](../run/hpc_overview.md) and [Lawrencium](../run/lawrencium.md).
- If you only need the file layout and not the workflow details, jump to [Configuration Basics](configuration_basics.md).
