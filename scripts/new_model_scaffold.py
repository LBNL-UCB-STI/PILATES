#!/usr/bin/env python3
"""
Scaffold a new PILATES model integration.

This generator creates:
- model package boilerplate (preprocessor/runner/postprocessor/outputs)
- workflow step stub module (make_*_step factories)
- central registrations (ModelFactory, step exports, step contracts)
- a model-specific checklist in docs/checklists/

By default this writes directly to the repository. Use --dry-run to preview.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple


@dataclass(frozen=True)
class ScaffoldSpec:
    model: str
    class_prefix: str
    step_module: str
    major_stage: str

    @property
    def holder_prefix(self) -> str:
        return self.model

    @property
    def preprocess_output_class(self) -> str:
        return f"{self.class_prefix}PreprocessOutputs"

    @property
    def run_output_class(self) -> str:
        return f"{self.class_prefix}RunOutputs"

    @property
    def postprocess_output_class(self) -> str:
        return f"{self.class_prefix}PostprocessOutputs"

    @property
    def preprocessor_class(self) -> str:
        return f"{self.class_prefix}Preprocessor"

    @property
    def runner_class(self) -> str:
        return f"{self.class_prefix}Runner"

    @property
    def postprocessor_class(self) -> str:
        return f"{self.class_prefix}Postprocessor"


def _snake_case(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()
    if not cleaned:
        raise ValueError("Model name must contain at least one alphanumeric character")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", cleaned):
        raise ValueError(
            "Model name must start with a letter and contain only letters, numbers, underscores"
        )
    return cleaned


def _pascal_case(value: str) -> str:
    parts = [p for p in re.split(r"[^a-zA-Z0-9]", value) if p]
    if not parts:
        raise ValueError("Cannot derive class prefix from empty value")
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _matching_brace_index(text: str, open_brace_index: int) -> int:
    depth = 0
    for idx in range(open_brace_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError("Unbalanced braces while scanning file")


def _insert_once(text: str, *, anchor: str, snippet: str, dedupe_token: str) -> str:
    if dedupe_token in text:
        return text
    idx = text.find(anchor)
    if idx == -1:
        raise ValueError(f"Anchor not found: {anchor!r}")
    return text[:idx] + snippet + text[idx:]


def _append_import_after_pilates_imports(text: str, new_imports: List[str]) -> str:
    for imp in new_imports:
        if imp in text:
            continue
        lines = text.splitlines(keepends=True)
        insertion_index = -1
        for i, line in enumerate(lines):
            if line.startswith("from pilates."):
                insertion_index = i
        if insertion_index < 0:
            raise ValueError("Could not find PILATES import block")
        lines.insert(insertion_index + 1, imp + "\n")
        text = "".join(lines)
    return text


def _insert_registry_entry(
    text: str,
    *,
    registry_name: str,
    key_token: str,
    entry_block: str,
) -> str:
    if key_token in text:
        return text
    anchor = f"{registry_name} = {{"
    start = text.find(anchor)
    if start == -1:
        raise ValueError(f"Could not find registry anchor {anchor!r}")
    open_brace = text.find("{", start)
    close_brace = _matching_brace_index(text, open_brace)
    return text[:close_brace] + entry_block + text[close_brace:]


def _upsert_model_factory(repo_root: Path, spec: ScaffoldSpec, *, dry_run: bool) -> Tuple[Path, bool]:
    path = repo_root / "pilates/generic/model_factory.py"
    text = _read_text(path)
    original = text

    text = _append_import_after_pilates_imports(
        text,
        [
            f"from pilates.{spec.model}.preprocessor import {spec.preprocessor_class}",
            f"from pilates.{spec.model}.runner import {spec.runner_class}",
            f"from pilates.{spec.model}.postprocessor import {spec.postprocessor_class}",
        ],
    )

    entry = dedent(
        f"""
                "{spec.model}": {{
                    "preprocessor": {spec.preprocessor_class},
                    "runner": {spec.runner_class},
                    "postprocessor": {spec.postprocessor_class},
                }},
        """
    )
    text = _insert_registry_entry(
        text,
        registry_name="_registry",
        key_token=f'        "{spec.model}": {{',
        entry_block=entry,
    )

    changed = text != original
    if changed:
        _write_text(path, text, dry_run=dry_run)
    return path, changed


def _upsert_steps_init(repo_root: Path, spec: ScaffoldSpec, *, dry_run: bool) -> Tuple[Path, bool]:
    path = repo_root / "pilates/workflows/steps/__init__.py"
    text = _read_text(path)
    original = text

    step_import_block = dedent(
        f"""
        from .{spec.step_module} import (  # noqa: F401
            make_{spec.model}_postprocess_step,
            make_{spec.model}_preprocess_step,
            make_{spec.model}_run_step,
        )
        """
    )
    text = _insert_once(
        text,
        anchor="from .postprocessing import make_postprocessing_step  # noqa: F401\n",
        snippet=step_import_block,
        dedupe_token=f"from .{spec.step_module} import (  # noqa: F401",
    )

    module_import_line = re.search(
        r"^from \. import (.+)  # noqa: F401,E402$", text, flags=re.MULTILINE
    )
    if not module_import_line:
        raise ValueError("Could not find steps module re-export line")
    modules = [m.strip() for m in module_import_line.group(1).split(",")]
    if spec.step_module not in modules:
        modules.append(spec.step_module)
        new_line = "from . import " + ", ".join(modules) + "  # noqa: F401,E402"
        text = (
            text[: module_import_line.start()]
            + new_line
            + text[module_import_line.end() :]
        )

    changed = text != original
    if changed:
        _write_text(path, text, dry_run=dry_run)
    return path, changed


def _insert_block_before(text: str, *, anchor: str, block: str, dedupe_token: str) -> str:
    if dedupe_token in text:
        return text
    idx = text.find(anchor)
    if idx == -1:
        raise ValueError(f"Anchor not found: {anchor!r}")
    return text[:idx] + block + text[idx:]


def _upsert_steps_shared(repo_root: Path, spec: ScaffoldSpec, *, dry_run: bool) -> Tuple[Path, bool]:
    path = repo_root / "pilates/workflows/steps/shared.py"
    text = _read_text(path)
    original = text

    output_import_block = dedent(
        f"""
        from pilates.{spec.model}.outputs import (
            {spec.postprocess_output_class},
            {spec.preprocess_output_class},
            {spec.run_output_class},
        )
        """
    )
    text = _insert_block_before(
        text,
        anchor="from pilates.workflows.step_consist_meta import consist_step_meta\n",
        block=output_import_block,
        dedupe_token=f"from pilates.{spec.model}.outputs import (",
    )

    holder_fields_block = dedent(
        f"""
            {spec.holder_prefix}_preprocess: Optional[{spec.preprocess_output_class}] = None
            {spec.holder_prefix}_run: Optional[{spec.run_output_class}] = None
            {spec.holder_prefix}_postprocess: Optional[{spec.postprocess_output_class}] = None
        """
    )
    text = _insert_once(
        text,
        anchor="    def set_attribute(self, step_name: str, outputs: Any) -> None:\n",
        snippet=holder_fields_block,
        dedupe_token=f"{spec.holder_prefix}_preprocess: Optional[{spec.preprocess_output_class}]",
    )

    classes_entry = dedent(
        f"""
            "{spec.holder_prefix}_preprocess": {spec.preprocess_output_class},
            "{spec.holder_prefix}_run": {spec.run_output_class},
            "{spec.holder_prefix}_postprocess": {spec.postprocess_output_class},
        """
    )
    text = _insert_registry_entry(
        text,
        registry_name="STEP_OUTPUTS_CLASSES",
        key_token=f'    "{spec.holder_prefix}_preprocess": {spec.preprocess_output_class},',
        entry_block=classes_entry,
    )

    deps_entry = dedent(
        f"""
            "{spec.holder_prefix}_preprocess": {{
                "depends_on": [],
                "holder_inputs": [],
            }},
            "{spec.holder_prefix}_run": {{
                "depends_on": ["{spec.holder_prefix}_preprocess"],
                "holder_inputs": ["{spec.holder_prefix}_preprocess"],
            }},
            "{spec.holder_prefix}_postprocess": {{
                "depends_on": ["{spec.holder_prefix}_run"],
                "holder_inputs": ["{spec.holder_prefix}_run"],
            }},
        """
    )
    text = _insert_registry_entry(
        text,
        registry_name="STEP_DEPENDENCIES",
        key_token=f'    "{spec.holder_prefix}_preprocess": {{',
        entry_block=deps_entry,
    )

    changed = text != original
    if changed:
        _write_text(path, text, dry_run=dry_run)
    return path, changed


def _render_preprocessor(spec: ScaffoldSpec) -> str:
    return dedent(
        f'''
        from __future__ import annotations

        from typing import TYPE_CHECKING, Tuple

        from pilates.generic.preprocessor import GenericPreprocessor
        from pilates.generic.records import RecordStore

        if TYPE_CHECKING:
            from pilates.config import PilatesConfig
            from pilates.workspace import Workspace


        class {spec.preprocessor_class}(GenericPreprocessor):
            """Preprocess inputs for the {spec.class_prefix} model."""

            def copy_data_to_mutable_location(
                self,
                settings: "PilatesConfig",
                output_dir: str,
            ) -> Tuple[RecordStore, RecordStore]:
                """Copy immutable inputs into the mutable workspace location."""
                return RecordStore(), RecordStore()

            def _preprocess(
                self,
                workspace: "Workspace",
                previous_records: RecordStore = RecordStore(),
            ) -> RecordStore:
                """Build and return {spec.class_prefix} preprocess artifacts."""
                return RecordStore()
        '''
    ).strip() + "\n"


def _render_runner(spec: ScaffoldSpec) -> str:
    return dedent(
        f'''
        from __future__ import annotations

        from typing import TYPE_CHECKING

        from pilates.generic.runner import GenericRunner
        from pilates.generic.records import RecordStore

        if TYPE_CHECKING:
            from pilates.workspace import Workspace


        class {spec.runner_class}(GenericRunner):
            """Execute the {spec.class_prefix} model."""

            def _run(
                self,
                store: RecordStore,
                workspace: "Workspace",
            ) -> RecordStore:
                """Run {spec.class_prefix} and return raw output artifacts."""
                return RecordStore()
        '''
    ).strip() + "\n"


def _render_postprocessor(spec: ScaffoldSpec) -> str:
    return dedent(
        f'''
        from __future__ import annotations

        from typing import Optional, TYPE_CHECKING

        from pilates.generic.postprocessor import GenericPostprocessor
        from pilates.generic.records import RecordStore

        if TYPE_CHECKING:
            from pilates.workspace import Workspace


        class {spec.postprocessor_class}(GenericPostprocessor):
            """Postprocess {spec.class_prefix} outputs for downstream stages."""

            def _postprocess(
                self,
                raw_outputs: RecordStore,
                workspace: "Workspace",
                model_run_hash: Optional[str] = None,
            ) -> RecordStore:
                """Transform raw outputs into publishable artifacts."""
                return raw_outputs
        '''
    ).strip() + "\n"


def _render_outputs(spec: ScaffoldSpec) -> str:
    return dedent(
        f'''
        from __future__ import annotations

        from dataclasses import dataclass, field
        from pathlib import Path
        from typing import ClassVar, Dict, Iterable, Tuple, TYPE_CHECKING

        from pilates.generic.records import RecordStore
        from pilates.utils.coupler_helpers import artifact_to_path
        from pilates.workflows.outputs_base import StepOutputsBase

        if TYPE_CHECKING:
            from pilates.workspace import Workspace


        @dataclass
        class {spec.preprocess_output_class}(StepOutputsBase):
            """Typed outputs from {spec.class_prefix} preprocess."""

            primary_output_attr: ClassVar[str] = "output_root"
            declared_outputs: ClassVar[Tuple[str, ...]] = ()
            required_path_fields: ClassVar[Tuple[str, ...]] = ("output_root",)
            dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)

            output_root: Path
            prepared_inputs: Dict[str, Path] = field(default_factory=dict)

            def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
                for key, path in self.prepared_inputs.items():
                    yield key, path, f"{spec.class_prefix} prepared input: {{key}}"

            @classmethod
            def from_record_store(
                cls, record_store: RecordStore, workspace: "Workspace"
            ) -> "{spec.preprocess_output_class}":
                mapping = record_store.to_mapping() if record_store is not None else {{}}
                prepared_inputs: Dict[str, Path] = {{}}
                for key, value in mapping.items():
                    path = artifact_to_path(value, workspace)
                    if path is None:
                        continue
                    prepared_inputs[key] = Path(path)
                return cls(
                    output_root=Path(workspace.full_path) / "{spec.model}" / "preprocess",
                    prepared_inputs=prepared_inputs,
                )


        @dataclass
        class {spec.run_output_class}(StepOutputsBase):
            """Typed outputs from {spec.class_prefix} run."""

            primary_output_attr: ClassVar[str] = "output_root"
            declared_outputs: ClassVar[Tuple[str, ...]] = ()
            required_path_fields: ClassVar[Tuple[str, ...]] = ("output_root",)
            dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)

            output_root: Path
            raw_outputs: Dict[str, Path] = field(default_factory=dict)

            def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
                for key, path in self.raw_outputs.items():
                    yield key, path, f"{spec.class_prefix} raw output: {{key}}"

            @classmethod
            def from_record_store(
                cls, record_store: RecordStore, workspace: "Workspace"
            ) -> "{spec.run_output_class}":
                mapping = record_store.to_mapping() if record_store is not None else {{}}
                raw_outputs: Dict[str, Path] = {{}}
                for key, value in mapping.items():
                    path = artifact_to_path(value, workspace)
                    if path is None:
                        continue
                    raw_outputs[key] = Path(path)
                return cls(
                    output_root=Path(workspace.full_path) / "{spec.model}" / "run",
                    raw_outputs=raw_outputs,
                )


        @dataclass
        class {spec.postprocess_output_class}(StepOutputsBase):
            """Typed outputs from {spec.class_prefix} postprocess."""

            primary_output_attr: ClassVar[str] = "output_root"
            declared_outputs: ClassVar[Tuple[str, ...]] = ()
            required_path_fields: ClassVar[Tuple[str, ...]] = ("output_root",)
            dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)

            output_root: Path
            processed_outputs: Dict[str, Path] = field(default_factory=dict)

            def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
                for key, path in self.processed_outputs.items():
                    yield key, path, f"{spec.class_prefix} postprocess output: {{key}}"

            @classmethod
            def from_record_store(
                cls, record_store: RecordStore, workspace: "Workspace"
            ) -> "{spec.postprocess_output_class}":
                mapping = record_store.to_mapping() if record_store is not None else {{}}
                processed_outputs: Dict[str, Path] = {{}}
                for key, value in mapping.items():
                    path = artifact_to_path(value, workspace)
                    if path is None:
                        continue
                    processed_outputs[key] = Path(path)
                return cls(
                    output_root=Path(workspace.full_path) / "{spec.model}" / "postprocess",
                    processed_outputs=processed_outputs,
                )
        '''
    ).strip() + "\n"


def _render_step_module(spec: ScaffoldSpec) -> str:
    return dedent(
        f'''
        from __future__ import annotations

        from typing import Any, Callable, Dict

        from pilates.{spec.model}.outputs import (
            {spec.postprocess_output_class},
            {spec.preprocess_output_class},
            {spec.run_output_class},
        )

        from .shared import (
            CouplerProtocol,
            PilatesConfig,
            StepOutputsHolder,
            Workspace,
            WorkflowState,
            _execute_postprocess,
            _execute_preprocess,
            _execute_run,
            _make_generic_step_function,
            log_and_set_output,
        )


        def make_{spec.model}_preprocess_step(
            *,
            coupler: CouplerProtocol,
            outputs_holder: StepOutputsHolder,
        ) -> Callable[..., None]:
            """Build the {spec.class_prefix} preprocess workflow step."""

            def _log_outputs(
                outputs: {spec.preprocess_output_class},
                settings: PilatesConfig,
                state: WorkflowState,
                workspace: Workspace,
                holder: StepOutputsHolder,
            ) -> None:
                for short_name, path, description in outputs._iter_record_items():
                    log_and_set_output(
                        key=short_name,
                        path=str(path),
                        description=description,
                        coupler=coupler,
                    )

            return _make_generic_step_function(
                coupler=coupler,
                outputs_holder=outputs_holder,
                model_name="{spec.model}",
                phase="preprocess",
                outputs_class={spec.preprocess_output_class},
                component_getter=lambda factory, state: factory.get_preprocessor(
                    "{spec.model}",
                    state,
                    WorkflowState.Stage.{spec.major_stage},
                ),
                component_executor=_execute_preprocess,
                outputs_holder_setter=lambda holder, outputs: setattr(
                    holder, "{spec.holder_prefix}_preprocess", outputs
                ),
                output_logger=_log_outputs,
            )


        def make_{spec.model}_run_step(
            *,
            coupler: CouplerProtocol,
            outputs_holder: StepOutputsHolder,
        ) -> Callable[..., None]:
            """Build the {spec.class_prefix} run workflow step."""

            def _log_outputs(
                outputs: {spec.run_output_class},
                settings: PilatesConfig,
                state: WorkflowState,
                workspace: Workspace,
                holder: StepOutputsHolder,
            ) -> None:
                for short_name, path, description in outputs._iter_record_items():
                    log_and_set_output(
                        key=short_name,
                        path=str(path),
                        description=description,
                        coupler=coupler,
                    )

            return _make_generic_step_function(
                coupler=coupler,
                outputs_holder=outputs_holder,
                model_name="{spec.model}",
                phase="run",
                outputs_class={spec.run_output_class},
                component_getter=lambda factory, state: factory.get_runner(
                    "{spec.model}",
                    state,
                    WorkflowState.Stage.{spec.major_stage},
                ),
                component_executor=_execute_run,
                outputs_holder_setter=lambda holder, outputs: setattr(
                    holder, "{spec.holder_prefix}_run", outputs
                ),
                output_logger=_log_outputs,
            )


        def make_{spec.model}_postprocess_step(
            *,
            coupler: CouplerProtocol,
            outputs_holder: StepOutputsHolder,
        ) -> Callable[..., None]:
            """Build the {spec.class_prefix} postprocess workflow step."""

            def _log_outputs(
                outputs: {spec.postprocess_output_class},
                settings: PilatesConfig,
                state: WorkflowState,
                workspace: Workspace,
                holder: StepOutputsHolder,
            ) -> None:
                for short_name, path, description in outputs._iter_record_items():
                    log_and_set_output(
                        key=short_name,
                        path=str(path),
                        description=description,
                        coupler=coupler,
                    )

            return _make_generic_step_function(
                coupler=coupler,
                outputs_holder=outputs_holder,
                model_name="{spec.model}",
                phase="postprocess",
                outputs_class={spec.postprocess_output_class},
                component_getter=lambda factory, state: factory.get_postprocessor(
                    "{spec.model}",
                    state,
                    WorkflowState.Stage.{spec.major_stage},
                ),
                component_executor=_execute_postprocess,
                outputs_holder_setter=lambda holder, outputs: setattr(
                    holder, "{spec.holder_prefix}_postprocess", outputs
                ),
                output_logger=_log_outputs,
            )
        '''
    ).strip() + "\n"


def _render_checklist(spec: ScaffoldSpec) -> str:
    return dedent(
        f"""
        # Model Integration Checklist: `{spec.model}`

        This checklist was scaffolded by `scripts/new_model_scaffold.py`.

        ## Generated Artifacts

        - [ ] `pilates/{spec.model}/preprocessor.py`
        - [ ] `pilates/{spec.model}/runner.py`
        - [ ] `pilates/{spec.model}/postprocessor.py`
        - [ ] `pilates/{spec.model}/outputs.py`
        - [ ] `pilates/workflows/steps/{spec.step_module}.py`

        ## Registrations Applied

        - [ ] `pilates/generic/model_factory.py`
        - [ ] `pilates/workflows/steps/__init__.py`
        - [ ] `pilates/workflows/steps/shared.py`

        ## Required Follow-up Wiring

        - [ ] Add stage assembly `StepRef(...)` calls in the appropriate stage module:
              `pilates/workflows/stages/*.py`
        - [ ] Prefer declaring expected outputs on the step metadata path (for example
              `@define_step(outputs=[...])`) and rely on orchestration's strict
              inferred defaults (`output_missing=\"error\"`, `output_mismatch=\"error\"`).
              Use explicit `outputs` (or `StepRef.required_outputs` alias) /
              `StepRef.output_*` only as overrides.
        - [ ] Set `declared_outputs` on generated `StepOutputs` classes for stable output keys
              so decorator metadata and runtime fallback use one canonical contract.
        - [ ] Add the new `make_{spec.model}_*_step` call(s) into `run.py::_build_schema_steps()` so
              startup contract validation includes this model.
        - [ ] Add/extend coupler keys in `pilates/workflows/artifact_keys.py`.
        - [ ] Add/extend coupler schema entries in `pilates/workflows/coupler_schema.py`.
        - [ ] Add/extend Consist config hashing/facets in `pilates/utils/consist_config.py`.
        - [ ] Add model-specific expected input/output declarations (component classes + stage wiring).

        ## Test Plan

        - [ ] Add step contract tests for `{spec.model}` in `tests/test_stage_contracts.py` or a model-specific test.
        - [ ] Add expected input/output contract tests (`tests/test_expected_inputs_contracts.py`).
        - [ ] Add facet/coupler/manifest invariants if this model introduces new cross-step artifacts.
        - [ ] Run targeted tests and record commands/results in your PR.

        ## Documentation

        - [ ] Update `docs/adding_a_model.md` with model-specific notes if the integration pattern differs.
        - [ ] Update `docs/consist_migration_checklist.md` if this touches migration work items.
        """
    ).strip() + "\n"


def _create_model_files(repo_root: Path, spec: ScaffoldSpec, *, dry_run: bool, force: bool) -> List[Tuple[Path, bool]]:
    model_dir = repo_root / "pilates" / spec.model
    files = {
        model_dir / "__init__.py": f"# {spec.model} subpackage\n",
        model_dir / "preprocessor.py": _render_preprocessor(spec),
        model_dir / "runner.py": _render_runner(spec),
        model_dir / "postprocessor.py": _render_postprocessor(spec),
        model_dir / "outputs.py": _render_outputs(spec),
        repo_root / "pilates" / "workflows" / "steps" / f"{spec.step_module}.py": _render_step_module(spec),
        repo_root / "docs" / "checklists" / f"add_model_{spec.model}.md": _render_checklist(spec),
    }

    results: List[Tuple[Path, bool]] = []
    for path, content in files.items():
        existed = path.exists()
        if existed and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file without --force: {path}"
            )
        _write_text(path, content, dry_run=dry_run)
        results.append((path, not existed or force))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model slug (e.g., freight, emissions)")
    parser.add_argument(
        "--class-prefix",
        help="Class prefix override (default: PascalCase(model))",
    )
    parser.add_argument(
        "--step-module",
        help="Workflow step module filename under pilates/workflows/steps/ (default: model slug)",
    )
    parser.add_argument(
        "--major-stage",
        default="supply_demand_loop",
        choices=[
            "land_use",
            "vehicle_ownership_model",
            "supply_demand_loop",
            "postprocessing",
        ],
        help="WorkflowState.Stage enum used in generated step factory component_getter lambdas",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (advanced; useful for tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite scaffold target files if they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model = _snake_case(args.model)
    class_prefix = args.class_prefix or _pascal_case(model)
    step_module = _snake_case(args.step_module or model)

    spec = ScaffoldSpec(
        model=model,
        class_prefix=class_prefix,
        step_module=step_module,
        major_stage=args.major_stage,
    )

    repo_root = args.repo_root.resolve()

    actions: List[Tuple[str, Path, bool]] = []
    try:
        for path, changed in _create_model_files(
            repo_root,
            spec,
            dry_run=args.dry_run,
            force=args.force,
        ):
            actions.append(("create", path, changed))

        path, changed = _upsert_model_factory(repo_root, spec, dry_run=args.dry_run)
        actions.append(("update", path, changed))

        path, changed = _upsert_steps_init(repo_root, spec, dry_run=args.dry_run)
        actions.append(("update", path, changed))

        path, changed = _upsert_steps_shared(repo_root, spec, dry_run=args.dry_run)
        actions.append(("update", path, changed))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"[{mode}] scaffold model={spec.model} class_prefix={spec.class_prefix}")
    for action, path, changed in actions:
        status = "changed" if changed else "no-op"
        print(f"  - {action}: {path} ({status})")

    print("\nNext step: complete the generated checklist:")
    print(f"  {repo_root / 'docs' / 'checklists' / f'add_model_{spec.model}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
