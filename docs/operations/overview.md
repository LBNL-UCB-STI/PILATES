---
title: Operations Overview
summary: Operator and contributor utilities that sit beside the main run and analysis paths.
---

# Operations Overview

## Current Utility Surface

The public operations section is intentionally small. It documents utility workflows that sit beside the main run path but are still current enough to be useful to operators and contributors.

At the moment, the main public utility is preserved test output for inspection and review. That workflow is narrow on purpose: it is for capturing the artifacts produced by a test run so they can be inspected without rerunning the scenario.

If a utility starts to overlap with normal run execution, restart behavior, or analysis, it belongs in the run, workflow, or analysis docs instead of becoming a second operations guide.

## Adjacent Pages

- Use [Test Output Preservation](test_output_preservation.md) for the concrete preserved-output workflow.
- Pair this with [Database Documentation Guide](../reference/database_documentation_guide.md) when inspecting outputs.
- Use [Troubleshooting](../run/troubleshooting.md) for failure handling.
