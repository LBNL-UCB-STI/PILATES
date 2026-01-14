# Adding a Model to PILATES

This guide describes the minimum steps to wire a new model into the PILATES
workflow using the current step-based orchestration pattern.

## Quick Checklist (In Order)

1. Create the model package under `pilates/<model_name>/`
   - `__init__.py`
   - `preprocessor.py` (class `ModelPreprocessor`)
   - `runner.py` (class `ModelRunner`)
   - `postprocessor.py` (class `ModelPostprocessor`)
   - `outputs.py` (typed outputs dataclasses)

2. Register components in `pilates/generic/model_factory.py`
   - Import the new preprocessor/runner/postprocessor.
   - Add registry mapping for `<model_name>`.

3. Add typed outputs to `pilates/workflows/steps.py`
   - Add `Optional[...]` attributes to `StepOutputsHolder`.
   - Register classes in `STEP_OUTPUTS_CLASSES`.
   - Add dependencies in `STEP_DEPENDENCIES`.

4. Create step factories in `pilates/workflows/steps.py`
   - `make_<model>_preprocess_step`
   - `make_<model>_run_step`
   - `make_<model>_postprocess_step`
   - Add any custom input/output logging as needed.

5. Add the stage in `run.py`
   - Import step factories.
   - Build `WorkflowStepSpec` list and call `WorkflowStage.run(...)`.
   - Use `WorkflowState.Stage` for gating and completion.

6. Add inputs builder (optional)
   - If you need specialized inputs, add `pilates/<model_name>/inputs.py` and
     `build_<model>_inputs(...)`.
   - Update `run.py` to call it before preprocess.

7. Add or update tests
   - Reuse `tests/fixtures/` where possible.
   - Document any long-running tests in docstrings.

## Minimal File Template

### `pilates/<model_name>/outputs.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, TYPE_CHECKING

from pilates.generic.records import RecordStore
from pilates.utils.coupler_helpers import artifact_to_path
from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    from pilates.workspace import Workspace


@dataclass
class ModelPreprocessOutputs(StepOutputsBase):
    primary_output_attr: ClassVar[str] = "model_mutable_data_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("model_mutable_data_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)

    model_mutable_data_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ModelPreprocessOutputs":
        mapping = record_store.to_mapping() if record_store is not None else {}
        prepared_inputs: Dict[str, Path] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            prepared_inputs[key] = Path(path)
        return cls(
            model_mutable_data_dir=Path(workspace.get_model_mutable_data_dir()),
            prepared_inputs=prepared_inputs,
        )


@dataclass
class ModelRunOutputs(StepOutputsBase):
    primary_output_attr: ClassVar[str] = "model_output_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("model_output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)

    model_output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ModelRunOutputs":
        mapping = record_store.to_mapping() if record_store is not None else {}
        raw_outputs: Dict[str, Path] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
        return cls(
            model_output_dir=Path(workspace.get_model_output_dir()),
            raw_outputs=raw_outputs,
        )


@dataclass
class ModelPostprocessOutputs(StepOutputsBase):
    primary_output_attr: ClassVar[str] = "model_output_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("model_output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)

    model_output_dir: Path
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ModelPostprocessOutputs":
        mapping = record_store.to_mapping() if record_store is not None else {}
        processed_outputs: Dict[str, Path] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            processed_outputs[key] = Path(path)
        return cls(
            model_output_dir=Path(workspace.get_model_output_dir()),
            processed_outputs=processed_outputs,
        )
```

## Common Gotchas

- Forgetting to update `STEP_OUTPUTS_CLASSES` or `STEP_DEPENDENCIES`.
- Missing `StepOutputsHolder` attributes for new steps.
- Not wiring a stage in `run.py` (steps defined but never called).
- Using implicit file paths instead of `Workspace` helpers.
- Missing provenance logging in step factories (inputs/outputs).

## Verification

- Run a single-year scenario with `-v` and check coupler keys.
- Ensure step outputs validate and manifest entries deserialize.
- Confirm outputs are visible in `docs/lineage_map.md` for the new model.
