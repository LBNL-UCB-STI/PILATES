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
from typing import Dict, List, Optional, Sequence, Tuple


STAGE_PATTERN_LINEAR = "linear"
STAGE_PATTERN_ITERATIVE = "iterative"
DEFAULT_STAGE_PATTERNS: Tuple[str, ...] = (
    STAGE_PATTERN_LINEAR,
    STAGE_PATTERN_ITERATIVE,
)

_MAJOR_STAGE_TO_CATALOG_STAGE: Dict[str, str] = {
    "land_use": "land_use",
    "vehicle_ownership_model": "vehicle_ownership_model",
    "supply_demand_loop": "traffic_assignment",
    "postprocessing": "postprocessing",
}

_MAJOR_STAGE_TO_STAGE_MODULE: Dict[str, str] = {
    "land_use": "pilates/workflows/stages/land_use.py",
    "vehicle_ownership_model": "pilates/workflows/stages/vehicle_ownership.py",
    "supply_demand_loop": "pilates/workflows/stages/supply_demand.py",
    "postprocessing": "pilates/workflows/stages/postprocessing.py",
}

_MAJOR_STAGE_TO_STAGE_FUNCTION: Dict[str, str] = {
    "land_use": "run_land_use_stage",
    "vehicle_ownership_model": "run_vehicle_ownership_stage",
    "supply_demand_loop": "run_supply_demand_stage",
    "postprocessing": "run_postprocessing_stage",
}

_CATALOG_STAGE_ENABLEMENT_DEFAULTS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "land_use": ("land_use_enabled", "land_use"),
    "vehicle_ownership_model": (
        "vehicle_ownership_model_enabled",
        "vehicle_ownership",
    ),
    "activity_demand": ("activity_demand_enabled", "activity_demand"),
    "traffic_assignment": ("traffic_assignment_enabled", "travel"),
}


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
    def preprocess_step_name(self) -> str:
        return f"{self.model}_preprocess"

    @property
    def run_step_name(self) -> str:
        return f"{self.model}_run"

    @property
    def postprocess_step_name(self) -> str:
        return f"{self.model}_postprocess"

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

    @property
    def provenance_constant_name(self) -> str:
        return f"_{self.model.upper()}_PROVENANCE"


@dataclass(frozen=True)
class CatalogScaffoldOptions:
    stage_name: str
    order_start: Optional[int]
    enabled_flag_attr: Optional[str]
    enabled_model_attr: Optional[str]
    provenance_builder_key: Optional[str]


@dataclass(frozen=True)
class StagePatchPlanOptions:
    target_module: Path
    stage_function: str


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


def _normalize_optional_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "false", "off"}:
        return None
    return cleaned


def _resolve_catalog_options(
    *,
    major_stage: str,
    catalog_stage_arg: Optional[str],
    enabled_flag_attr_arg: Optional[str],
    enabled_model_attr_arg: Optional[str],
    provenance_builder_key_arg: Optional[str],
    order_start_arg: Optional[int],
) -> CatalogScaffoldOptions:
    default_catalog_stage = _MAJOR_STAGE_TO_CATALOG_STAGE.get(major_stage, major_stage)
    stage_name = _normalize_optional_arg(catalog_stage_arg) or default_catalog_stage
    default_enablement = _CATALOG_STAGE_ENABLEMENT_DEFAULTS.get(stage_name, (None, None))
    enabled_flag_attr = _normalize_optional_arg(enabled_flag_attr_arg)
    enabled_model_attr = _normalize_optional_arg(enabled_model_attr_arg)
    if enabled_flag_attr is None and enabled_flag_attr_arg is None:
        enabled_flag_attr = default_enablement[0]
    if enabled_model_attr is None and enabled_model_attr_arg is None:
        enabled_model_attr = default_enablement[1]

    return CatalogScaffoldOptions(
        stage_name=stage_name,
        order_start=order_start_arg,
        enabled_flag_attr=enabled_flag_attr,
        enabled_model_attr=enabled_model_attr,
        provenance_builder_key=_normalize_optional_arg(provenance_builder_key_arg),
    )


def _resolve_stage_patterns(raw_patterns: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not raw_patterns:
        return DEFAULT_STAGE_PATTERNS
    ordered: List[str] = []
    for pattern in raw_patterns:
        if pattern not in ordered:
            ordered.append(pattern)
    return tuple(ordered)


def _resolve_stage_patch_plan_options(
    *,
    major_stage: str,
    enabled: bool,
    stage_target_module_arg: Optional[str],
    stage_target_function_arg: Optional[str],
) -> Optional[StagePatchPlanOptions]:
    if not enabled:
        return None

    default_target_module = _MAJOR_STAGE_TO_STAGE_MODULE.get(
        major_stage,
        f"pilates/workflows/stages/{major_stage}.py",
    )
    target_module = Path(
        _normalize_optional_arg(stage_target_module_arg) or default_target_module
    )
    stage_function = _normalize_optional_arg(stage_target_function_arg) or _MAJOR_STAGE_TO_STAGE_FUNCTION.get(
        major_stage,
        f"run_{major_stage}_stage",
    )

    return StagePatchPlanOptions(
        target_module=target_module,
        stage_function=stage_function,
    )


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _matching_delimiter_index(
    text: str,
    open_index: int,
    *,
    open_char: str,
    close_char: str,
) -> int:
    depth = 0
    for idx in range(open_index, len(text)):
        char = text[idx]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(
        f"Unbalanced delimiters while scanning file: {open_char!r} {close_char!r}"
    )


def _matching_brace_index(text: str, open_brace_index: int) -> int:
    return _matching_delimiter_index(
        text,
        open_brace_index,
        open_char="{",
        close_char="}",
    )


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
        insertion_index = _find_pilates_import_insertion_index(lines)
        lines.insert(insertion_index + 1, imp + "\n")
        text = "".join(lines)
    return text


def _find_pilates_import_insertion_index(lines: Sequence[str]) -> int:
    insertion_index = -1
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if not line.startswith("from pilates."):
            line_index += 1
            continue

        end_index = line_index
        delimiter_depth = line.count("(") - line.count(")")
        while delimiter_depth > 0:
            end_index += 1
            if end_index >= len(lines):
                raise ValueError("Unbalanced multiline PILATES import block")
            delimiter_depth += lines[end_index].count("(") - lines[end_index].count(")")

        insertion_index = end_index
        line_index = end_index + 1

    if insertion_index < 0:
        raise ValueError("Could not find PILATES import block")
    return insertion_index


def _insert_import_block_after_pilates_imports(
    text: str,
    *,
    import_block: str,
    dedupe_token: str,
) -> str:
    if dedupe_token in text:
        return text
    lines = text.splitlines(keepends=True)
    insertion_index = _find_pilates_import_insertion_index(lines)
    block = import_block if import_block.endswith("\n") else import_block + "\n"
    lines.insert(insertion_index + 1, block)
    return "".join(lines)


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


def _insert_tuple_entry(
    text: str,
    *,
    tuple_anchor: str,
    dedupe_token: str,
    entry_block: str,
) -> str:
    if dedupe_token in text:
        return text
    start = text.find(tuple_anchor)
    if start == -1:
        raise ValueError(f"Could not find tuple anchor {tuple_anchor!r}")
    open_paren = text.find("(", start)
    if open_paren == -1:
        raise ValueError(f"Could not find opening tuple paren after {tuple_anchor!r}")
    close_paren = _matching_delimiter_index(
        text,
        open_paren,
        open_char="(",
        close_char=")",
    )
    return text[:close_paren] + entry_block + text[close_paren:]


def _next_catalog_order(text: str) -> int:
    matches = [int(value) for value in re.findall(r"\border=(\d+)\b", text)]
    if not matches:
        return 10
    return max(matches) + 10


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

    registry_entry_block = dedent(
        f"""
            "{spec.preprocess_step_name}": make_{spec.model}_preprocess_step,
            "{spec.run_step_name}": make_{spec.model}_run_step,
            "{spec.postprocess_step_name}": make_{spec.model}_postprocess_step,
        """
    )
    text = _insert_once(
        text,
        anchor="}\n\n\ndef schema_step_builder_registry() -> Dict[str, Callable[..., Any]]:\n",
        snippet=registry_entry_block,
        dedupe_token=f'    "{spec.preprocess_step_name}": make_{spec.model}_preprocess_step,',
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

    changed = text != original
    if changed:
        _write_text(path, text, dry_run=dry_run)
    return path, changed


def _tuple_literal(values: Sequence[str]) -> str:
    if not values:
        return "()"
    if len(values) == 1:
        return f'("{values[0]}",)'
    inner = ", ".join(f'"{value}"' for value in values)
    return f"({inner})"


def _workflow_spec_optional_lines(
    catalog_options: CatalogScaffoldOptions,
    *,
    provenance_ref: Optional[str],
) -> str:
    lines: List[str] = []
    if catalog_options.enabled_flag_attr is not None:
        lines.append(f'        enabled_flag_attr="{catalog_options.enabled_flag_attr}",')
    if catalog_options.enabled_model_attr is not None:
        lines.append(f'        enabled_model_attr="{catalog_options.enabled_model_attr}",')
    if provenance_ref is not None:
        lines.append(f"        provenance={provenance_ref},")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _upsert_workflow_catalog(
    repo_root: Path,
    spec: ScaffoldSpec,
    *,
    catalog_options: CatalogScaffoldOptions,
    dry_run: bool,
) -> Tuple[Path, bool]:
    path = repo_root / "pilates/workflows/catalog.py"
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
    text = _insert_import_block_after_pilates_imports(
        text,
        import_block=output_import_block,
        dedupe_token=f"from pilates.{spec.model}.outputs import (",
    )

    provenance_ref: Optional[str] = None
    if catalog_options.provenance_builder_key is not None:
        provenance_ref = spec.provenance_constant_name
        provenance_block = (
            f'{provenance_ref} = WorkflowStepProvenanceSpec('
            f'builder_key="{catalog_options.provenance_builder_key}")\n\n'
        )
        text = _insert_once(
            text,
            anchor="WORKFLOW_STEP_SPECS: Tuple[WorkflowStepSpec, ...] = (\n",
            snippet=provenance_block,
            dedupe_token=f"{provenance_ref} = WorkflowStepProvenanceSpec(",
        )

    order_start = (
        catalog_options.order_start
        if catalog_options.order_start is not None
        else _next_catalog_order(text)
    )
    optional_lines = _workflow_spec_optional_lines(
        catalog_options,
        provenance_ref=provenance_ref,
    )
    entry_block = dedent(
        f"""
            WorkflowStepSpec(
                step_name="{spec.preprocess_step_name}",
                model_name="{spec.preprocess_step_name}",
                phase="preprocess",
                stage_name="{catalog_options.stage_name}",
                order={order_start},
                outputs_class={spec.preprocess_output_class},
                depends_on=(),
                holder_inputs=(),
{optional_lines}    ),
            WorkflowStepSpec(
                step_name="{spec.run_step_name}",
                model_name="{spec.run_step_name}",
                phase="run",
                stage_name="{catalog_options.stage_name}",
                order={order_start + 10},
                outputs_class={spec.run_output_class},
                depends_on={_tuple_literal((spec.preprocess_step_name,))},
                holder_inputs={_tuple_literal((spec.preprocess_step_name,))},
{optional_lines}    ),
            WorkflowStepSpec(
                step_name="{spec.postprocess_step_name}",
                model_name="{spec.postprocess_step_name}",
                phase="postprocess",
                stage_name="{catalog_options.stage_name}",
                order={order_start + 20},
                outputs_class={spec.postprocess_output_class},
                depends_on={_tuple_literal((spec.run_step_name,))},
                holder_inputs={_tuple_literal((spec.run_step_name,))},
{optional_lines}    ),
        """
    )
    text = _insert_tuple_entry(
        text,
        tuple_anchor="WORKFLOW_STEP_SPECS: Tuple[WorkflowStepSpec, ...] = (",
        dedupe_token=f'step_name="{spec.preprocess_step_name}"',
        entry_block=entry_block,
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

        from typing import Any, Callable

        from pilates.{spec.model}.outputs import (
            {spec.postprocess_output_class},
            {spec.preprocess_output_class},
            {spec.run_output_class},
        )

        from .shared import (
            CouplerProtocol,
            PilatesConfig,
            StandardStepSpec,
            StepOutputsHolder,
            Workspace,
            WorkflowState,
            build_standard_step,
            log_and_set_output,
        )


        def _execute_{spec.model}_preprocess(
            component: Any,
            workspace: Workspace,
            outputs_holder: StepOutputsHolder,
            **kwargs: Any,
        ) -> {spec.preprocess_output_class}:
            """Adapt this scaffolded preprocess executor to the model component."""
            raise NotImplementedError("Adapt preprocess execution for {spec.model}")


        def _execute_{spec.model}_run(
            component: Any,
            workspace: Workspace,
            outputs_holder: StepOutputsHolder,
            **kwargs: Any,
        ) -> {spec.run_output_class}:
            """Adapt this scaffolded run executor to the model component."""
            raise NotImplementedError("Adapt run execution for {spec.model}")


        def _execute_{spec.model}_postprocess(
            component: Any,
            workspace: Workspace,
            outputs_holder: StepOutputsHolder,
            **kwargs: Any,
        ) -> {spec.postprocess_output_class}:
            """Adapt this scaffolded postprocess executor to the model component."""
            raise NotImplementedError("Adapt postprocess execution for {spec.model}")


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

            return build_standard_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
                spec=StandardStepSpec(
                    step_name="{spec.preprocess_step_name}",
                    model_name="{spec.model}",
                    phase="preprocess",
                    outputs_class={spec.preprocess_output_class},
                    component_getter=lambda factory, state: factory.get_preprocessor(
                        "{spec.model}", state
                    ),
                    component_executor=_execute_{spec.model}_preprocess,
                    outputs_holder_key="{spec.holder_prefix}_preprocess",
                    output_logger=_log_outputs,
                    step_description="{spec.class_prefix} preprocess",
                    tags=["{spec.model}", "preprocess"],
                ),
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

            return build_standard_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
                spec=StandardStepSpec(
                    step_name="{spec.run_step_name}",
                    model_name="{spec.model}",
                    phase="run",
                    outputs_class={spec.run_output_class},
                    component_getter=lambda factory, state: factory.get_runner(
                        "{spec.model}", state
                    ),
                    component_executor=_execute_{spec.model}_run,
                    outputs_holder_key="{spec.holder_prefix}_run",
                    output_logger=_log_outputs,
                    step_description="{spec.class_prefix} run",
                    tags=["{spec.model}", "run"],
                ),
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

            return build_standard_step(
                coupler=coupler,
                outputs_holder=outputs_holder,
                spec=StandardStepSpec(
                    step_name="{spec.postprocess_step_name}",
                    model_name="{spec.model}",
                    phase="postprocess",
                    outputs_class={spec.postprocess_output_class},
                    component_getter=lambda factory, state: factory.get_postprocessor(
                        "{spec.model}", state
                    ),
                    component_executor=_execute_{spec.model}_postprocess,
                    outputs_holder_key="{spec.holder_prefix}_postprocess",
                    output_logger=_log_outputs,
                    step_description="{spec.class_prefix} postprocess",
                    tags=["{spec.model}", "postprocess"],
                ),
            )
        '''
    ).strip() + "\n"


def _stage_template_relative_path(spec: ScaffoldSpec, pattern: str) -> Path:
    return (
        Path("docs")
        / "checklists"
        / "stage_templates"
        / f"add_model_{spec.model}.{pattern}.py.snippet"
    )


def _stage_patch_plan_relative_path(spec: ScaffoldSpec) -> Path:
    return (
        Path("docs")
        / "checklists"
        / "stage_templates"
        / f"add_model_{spec.model}.stage_patch.md"
    )


def _resolve_stage_patch_target_path(
    repo_root: Path,
    target_module: Path,
) -> Path:
    return target_module if target_module.is_absolute() else repo_root / target_module


def _first_matching_anchor_line(
    text: str,
    candidates: Sequence[str],
) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        for candidate in candidates:
            if candidate in stripped:
                return stripped
    return None


def _first_matching_anchor_regex_line(
    text: str,
    patterns: Sequence[str],
) -> Optional[str]:
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    for line in text.splitlines():
        stripped = line.strip()
        for pattern in compiled_patterns:
            if pattern.search(stripped):
                return stripped
    return None


def _extract_top_level_function_block(
    text: str,
    function_name: str,
) -> Optional[str]:
    lines = text.splitlines(keepends=True)
    function_pattern = re.compile(rf"^\s*def {re.escape(function_name)}\s*\(")
    start_index: Optional[int] = None
    for index, line in enumerate(lines):
        if function_pattern.match(line):
            start_index = index
            break
    if start_index is None:
        return None

    def_indent = len(lines[start_index]) - len(lines[start_index].lstrip())
    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= def_indent and line.lstrip() == line and not stripped.startswith("#"):
            end_index = index
            break
    return "".join(lines[start_index:end_index])


def _render_stage_patch_import_block(spec: ScaffoldSpec) -> str:
    return dedent(
        f"""
            from pilates.runtime.context import WorkflowRuntimeContext
            from pilates.workflows.binding import build_binding_plan
            from pilates.workflows.orchestration import StageRunner, StepRef, run_workflow
            make_{spec.model}_postprocess_step,
            make_{spec.model}_preprocess_step,
            make_{spec.model}_run_step,
        """
    ).strip()


def _render_stage_patch_stepref_block(
    spec: ScaffoldSpec,
    *,
    pattern: str,
) -> str:
    if pattern == STAGE_PATTERN_LINEAR:
        return dedent(
            f"""
            runtime_context = WorkflowRuntimeContext.from_parts(
                settings=settings,
                state=state,
                workspace=workspace,
                surface=surface,
            )
            stage_runner = StageRunner(
                stage_name="<stage_name>",
                scenario=scenario,
                state=runtime_context.state,
                settings=runtime_context.settings,
                workspace=runtime_context.workspace,
                coupler=coupler,
                outputs_holder=outputs_holder_year,
                name_suffix=str(year),
                run_workflow_fn=run_workflow,
            )
            {spec.model}_preprocess_binding = build_binding_plan(
                step_name="{spec.preprocess_step_name}",
                coupler=coupler,
                explicit_inputs={{...}},
                fallback_inputs={{...}},
                required_keys=[...],
                surface=runtime_context.surface,
            )
            stage_runner.run_step(
                step=StepRef(
                    name="{spec.preprocess_step_name}",
                    step_func=make_{spec.model}_preprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder_year,
                    ),
                    binding={spec.model}_preprocess_binding,
                    year=year,
                )
            )
            """
        ).strip()
    if pattern == STAGE_PATTERN_ITERATIVE:
        return dedent(
            f"""
            runtime_context = WorkflowRuntimeContext.from_parts(
                settings=settings,
                state=state,
                workspace=workspace,
                surface=surface,
            )
            stage_runner = StageRunner(
                stage_name="<stage_name>",
                scenario=scenario,
                state=runtime_context.state,
                settings=runtime_context.settings,
                workspace=runtime_context.workspace,
                coupler=coupler,
                outputs_holder=outputs_holder_iteration,
                name_suffix=f"{{year}}_iter{{i}}",
                iteration=i,
                run_workflow_fn=run_workflow,
            )
            {spec.model}_preprocess_binding = build_binding_plan(
                step_name="{spec.preprocess_step_name}",
                coupler=coupler,
                explicit_inputs={{...}},
                fallback_inputs={{...}},
                required_keys=[...],
                surface=runtime_context.surface,
            )
            stage_runner.run_step(
                step=StepRef(
                    name="{spec.preprocess_step_name}",
                    step_func=make_{spec.model}_preprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder_iteration,
                    ),
                    binding={spec.model}_preprocess_binding,
                    year=year,
                )
            )
            """
        ).strip()
    raise ValueError(f"Unsupported stage patch pattern: {pattern}")


def _anchor_candidates_for_pattern(pattern: str) -> Tuple[str, ...]:
    if pattern == STAGE_PATTERN_LINEAR:
        return ("steps = [", "run_workflow(")
    if pattern == STAGE_PATTERN_ITERATIVE:
        return ("for i in range(", "for iteration in range(", "run_workflow(")
    raise ValueError(f"Unsupported stage patch pattern: {pattern}")


def _anchor_regexes_for_pattern(pattern: str) -> Tuple[str, ...]:
    if pattern == STAGE_PATTERN_LINEAR:
        return (
            r"^[A-Za-z_][A-Za-z0-9_]*(?:\s*:\s*[^=]+)?\s*=\s*\[",
            r"^[A-Za-z_][A-Za-z0-9_]*\.extend\(",
            r"^run_workflow\(",
        )
    if pattern == STAGE_PATTERN_ITERATIVE:
        return (
            r"^for\s+.+\s+in\s+range\(",
            r"^for\s+.+\s+in\s+.+:$",
            r"^while\s+.+:$",
            r"^[A-Za-z_][A-Za-z0-9_]*(?:\s*:\s*[^=]+)?\s*=\s*\[",
            r"^run_workflow\(",
        )
    raise ValueError(f"Unsupported stage patch pattern: {pattern}")


def _first_matching_pattern_anchor(
    text: str,
    *,
    pattern: str,
) -> Optional[str]:
    return (
        _first_matching_anchor_regex_line(text, _anchor_regexes_for_pattern(pattern))
        or _first_matching_anchor_line(text, _anchor_candidates_for_pattern(pattern))
    )


def _render_stage_patch_plan(
    spec: ScaffoldSpec,
    *,
    catalog_stage: str,
    repo_root: Path,
    patch_options: StagePatchPlanOptions,
    stage_patterns: Sequence[str],
) -> str:
    target_path = _resolve_stage_patch_target_path(repo_root, patch_options.target_module)
    import_anchor = "from pilates.workflows.steps import ("
    function_anchor = f"def {patch_options.stage_function}("
    per_pattern_anchors: Dict[str, str] = {}

    if target_path.exists():
        target_text = _read_text(target_path)
        function_text = _extract_top_level_function_block(
            target_text,
            patch_options.stage_function,
        )
        import_anchor = (
            _first_matching_anchor_regex_line(
                target_text,
                (r"^from\s+pilates\.workflows\.steps\s+import\b",),
            )
            or import_anchor
        )
        function_anchor = (
            _first_matching_anchor_regex_line(
                target_text,
                (rf"^def\s+{re.escape(patch_options.stage_function)}\s*\(",),
            )
            or function_anchor
        )
        anchor_search_text = function_text or target_text
        for pattern in stage_patterns:
            detected_anchor = _first_matching_pattern_anchor(
                anchor_search_text,
                pattern=pattern,
            )
            if detected_anchor is None and function_text is not None:
                detected_anchor = _first_matching_pattern_anchor(
                    target_text,
                    pattern=pattern,
                )
            per_pattern_anchors[pattern] = (
                detected_anchor or _anchor_candidates_for_pattern(pattern)[0]
            )
    else:
        for pattern in stage_patterns:
            per_pattern_anchors[pattern] = _anchor_candidates_for_pattern(pattern)[0]

    pattern_sections = "\n\n".join(
        dedent(
            f"""
            ### `{pattern}` `StepRef` block

            Suggested insertion anchor in function body: `{per_pattern_anchors[pattern]}`

            ```python
            {_render_stage_patch_stepref_block(spec, pattern=pattern)}
            ```
            """
        ).strip()
        for pattern in stage_patterns
    )

    return dedent(
        f"""
        # Stage Patch Plan: `{spec.model}`

        This patch plan was scaffolded by `scripts/new_model_scaffold.py`.
        Insertion anchors are heuristic/advisory and should be adapted to the target stage module.

        ## Target Stage Wiring

        - Stage module to edit: `{patch_options.target_module.as_posix()}`
        - Stage function to edit: `{patch_options.stage_function}`
        - Catalog stage metadata: `stage_name="{catalog_stage}"`
        - Generated step names:
          - `{spec.preprocess_step_name}`
          - `{spec.run_step_name}`
          - `{spec.postprocess_step_name}`

        ## Insertion Anchors

        - Import anchor: `{import_anchor}`
        - Stage function anchor: `{function_anchor}`

        ## Ready-to-Paste Snippets

        ### Import block for `from pilates.workflows.steps import (...)`

        ```python
        {_render_stage_patch_import_block(spec)}
        ```

        {pattern_sections}

        ## Required Variable Adaptation Checklist

        - [ ] Confirm the stage function has `coupler`, `scenario`, `state`, `settings`, `workspace`, `surface`, and `year`, or adapt the snippet to an existing `WorkflowRuntimeContext`.
        - [ ] Replace `outputs_holder_year` and/or `outputs_holder_iteration` with the target function's holder variable names.
        - [ ] Replace placeholder `<stage_name>` and adapt `StageRunner(...)` naming/iteration metadata to the target stage.
        - [ ] Update `build_binding_plan(...)` explicit/fallback/required mappings for the new model's true dependencies.
        - [ ] Keep `stage_name="{catalog_stage}"` aligned between stage wiring and catalog entries.
        """
    ).strip() + "\n"


def _render_linear_stage_template(spec: ScaffoldSpec, *, catalog_stage: str) -> str:
    return dedent(
        f"""
        # Linear stage template for `{spec.model}`
        #
        # Copy/adapt this snippet into the target stage module under
        # pilates/workflows/stages/.

        from pilates.runtime.context import WorkflowRuntimeContext
        from pilates.workflows.binding import build_binding_plan
        from pilates.workflows.orchestration import StageRunner, StepRef, run_workflow
        from pilates.workflows.steps import (
            make_{spec.model}_postprocess_step,
            make_{spec.model}_preprocess_step,
            make_{spec.model}_run_step,
        )

        runtime_context = WorkflowRuntimeContext.from_parts(
            settings=settings,
            state=state,
            workspace=workspace,
            surface=surface,
        )
        stage_runner = StageRunner(
            stage_name="{catalog_stage}",
            scenario=scenario,
            state=runtime_context.state,
            settings=runtime_context.settings,
            workspace=runtime_context.workspace,
            coupler=coupler,
            outputs_holder=outputs_holder_year,
            name_suffix=str(year),
            run_workflow_fn=run_workflow,
        )

        preprocess_binding = build_binding_plan(
            step_name="{spec.preprocess_step_name}",
            coupler=coupler,
            explicit_inputs={{...}},
            fallback_inputs={{...}},
            required_keys=[...],
            surface=runtime_context.surface,
        )
        stage_runner.run_step(
            step=StepRef(
                name="{spec.preprocess_step_name}",
                step_func=make_{spec.model}_preprocess_step(
                    coupler=coupler,
                    outputs_holder=outputs_holder_year,
                ),
                binding=preprocess_binding,
                year=year,
            )
        )
        """
    ).strip() + "\n"


def _render_iterative_stage_template(spec: ScaffoldSpec, *, catalog_stage: str) -> str:
    return dedent(
        f"""
        # Iterative stage template for `{spec.model}`
        #
        # Use this when the model should run per-iteration (for example within
        # supply-demand loops). Adjust input resolution to your stage.

        from pilates.runtime.context import WorkflowRuntimeContext
        from pilates.workflows.binding import build_binding_plan
        from pilates.workflows.orchestration import StageRunner, StepRef, run_workflow
        from pilates.workflows.steps import (
            StepOutputsHolder,
            make_{spec.model}_postprocess_step,
            make_{spec.model}_preprocess_step,
            make_{spec.model}_run_step,
        )

        for iteration in range(settings.run.supply_demand_iters):
            outputs_holder_iteration = StepOutputsHolder()
            runtime_context = WorkflowRuntimeContext.from_parts(
                settings=settings,
                state=state,
                workspace=workspace,
                surface=surface,
            )
            stage_runner = StageRunner(
                stage_name="{catalog_stage}",
                scenario=scenario,
                state=runtime_context.state,
                settings=runtime_context.settings,
                workspace=runtime_context.workspace,
                coupler=coupler,
                outputs_holder=outputs_holder_iteration,
                name_suffix=f"{{year}}_iter{{iteration}}",
                iteration=iteration,
                run_workflow_fn=run_workflow,
            )

            preprocess_binding = build_binding_plan(
                step_name="{spec.preprocess_step_name}",
                coupler=coupler,
                explicit_inputs={{...}},
                fallback_inputs={{...}},
                required_keys=[...],
                surface=runtime_context.surface,
            )
            stage_runner.run_step(
                step=StepRef(
                    name="{spec.preprocess_step_name}",
                    step_func=make_{spec.model}_preprocess_step(
                        coupler=coupler,
                        outputs_holder=outputs_holder_iteration,
                    ),
                    binding=preprocess_binding,
                    year=year,
                )
            )
        """
    ).strip() + "\n"


def _render_stage_template(
    spec: ScaffoldSpec,
    *,
    catalog_stage: str,
    pattern: str,
) -> str:
    if pattern == STAGE_PATTERN_LINEAR:
        return _render_linear_stage_template(spec, catalog_stage=catalog_stage)
    if pattern == STAGE_PATTERN_ITERATIVE:
        return _render_iterative_stage_template(spec, catalog_stage=catalog_stage)
    raise ValueError(f"Unsupported stage template pattern: {pattern}")


def _render_checklist(
    spec: ScaffoldSpec,
    *,
    catalog_options: CatalogScaffoldOptions,
    stage_template_paths: Sequence[Path],
    stage_patch_plan_path: Optional[Path],
) -> str:
    template_lines = "\n".join(
        f"        - [ ] `{path.as_posix()}`" for path in stage_template_paths
    )
    stage_patch_line = (
        f"\n        - [ ] `{stage_patch_plan_path.as_posix()}`"
        if stage_patch_plan_path is not None
        else ""
    )
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
{template_lines}
{stage_patch_line}

        ## Registrations Applied

        - [ ] `pilates/generic/model_factory.py`
        - [ ] `pilates/workflows/steps/__init__.py`
        - [ ] `pilates/workflows/steps/__init__.py` (`SCHEMA_STEP_BUILDERS`)
        - [ ] `pilates/workflows/steps/shared.py`
        - [ ] `pilates/workflows/catalog.py`

        ## Required Follow-up Wiring

        - [ ] Start from one of the generated stage template snippets and wire
              the new model into the target stage module under
              `pilates/workflows/stages/*.py` using `WorkflowRuntimeContext`,
              `StageRunner`, and `build_binding_plan(...)`.
        - [ ] If generated, use the stage patch plan artifact in
              `docs/checklists/stage_templates/` to apply import + stage-runner /
              binding wiring with the recommended insertion anchors.
        - [ ] Confirm catalog stage metadata for `{spec.model}`:
              `stage_name="{catalog_options.stage_name}"`, `order={catalog_options.order_start or "auto"}`
              and enablement attrs (`enabled_flag_attr`, `enabled_model_attr`) are
              correct for your model.
        - [ ] Keep model enablement on the surface path. Do not derive run shape
              from raw settings or `WorkflowProfile` inside the model module.
        - [ ] Keep runtime source precedence and fallback policy in binding, not
              in stage-local `coupler.get(...)` chains.
        - [ ] Prefer declaring expected outputs on the step metadata path (for example
              `@define_step(outputs=[...])`) and rely on orchestration's strict
              inferred defaults (`output_missing=\"error\"`, `output_mismatch=\"error\"`).
              Use explicit `outputs` (or `StepRef.required_outputs` alias) /
              `StepRef.output_*` only as overrides.
        - [ ] Set `declared_outputs` on generated `StepOutputs` classes for stable output keys
              so decorator metadata and runtime fallback use one canonical contract.
        - [ ] Use `StandardStepSpec` / `build_standard_step()` for the normal
              typed-step shell unless the model truly needs custom execution wiring.
        - [ ] If this model needs provenance hashing/facets, confirm the generated
              catalog `provenance` metadata and add builder support in
              `pilates/utils/consist_config.py` as needed.
        - [ ] Add/extend coupler keys in `pilates/workflows/artifact_keys.py`.
        - [ ] Add/extend coupler schema entries in `pilates/workflows/coupler_schema.py`.
        - [ ] Add model-specific expected input/output declarations (component classes + stage wiring).

        ## Test Plan

        - [ ] Add step contract tests for `{spec.model}` in `tests/test_stage_contracts.py` or a model-specific test.
        - [ ] Add expected input/output contract tests (`tests/test_expected_inputs_contracts.py`).
        - [ ] Add facet/coupler/manifest invariants if this model introduces new cross-step artifacts.
        - [ ] Add or update architecture guardrail tests if the integration adds a new allowed seam.
        - [ ] Run targeted tests and record commands/results in your PR.

        ## Documentation

        - [ ] Update `docs/extend/adding_a_model.md` and related contributor docs if the integration pattern differs.
        - [ ] Update `docs-internal/consist_migration_checklist.md` if this touches migration work items.
        """
    ).strip() + "\n"


def _create_model_files(
    repo_root: Path,
    spec: ScaffoldSpec,
    *,
    catalog_options: CatalogScaffoldOptions,
    stage_patterns: Sequence[str],
    stage_patch_plan_options: Optional[StagePatchPlanOptions],
    dry_run: bool,
    force: bool,
) -> List[Tuple[Path, bool]]:
    model_dir = repo_root / "pilates" / spec.model
    template_relative_paths = [
        _stage_template_relative_path(spec, pattern) for pattern in stage_patterns
    ]
    files = {
        model_dir / "__init__.py": f"# {spec.model} subpackage\n",
        model_dir / "preprocessor.py": _render_preprocessor(spec),
        model_dir / "runner.py": _render_runner(spec),
        model_dir / "postprocessor.py": _render_postprocessor(spec),
        model_dir / "outputs.py": _render_outputs(spec),
        repo_root / "pilates" / "workflows" / "steps" / f"{spec.step_module}.py": _render_step_module(spec),
        repo_root
        / "docs"
        / "checklists"
        / f"add_model_{spec.model}.md": _render_checklist(
            spec,
            catalog_options=catalog_options,
            stage_template_paths=template_relative_paths,
            stage_patch_plan_path=(
                _stage_patch_plan_relative_path(spec)
                if stage_patch_plan_options is not None
                else None
            ),
        ),
    }
    for pattern, relpath in zip(stage_patterns, template_relative_paths):
        files[repo_root / relpath] = _render_stage_template(
            spec,
            catalog_stage=catalog_options.stage_name,
            pattern=pattern,
        )
    if stage_patch_plan_options is not None:
        stage_patch_plan_relpath = _stage_patch_plan_relative_path(spec)
        files[repo_root / stage_patch_plan_relpath] = _render_stage_patch_plan(
            spec,
            catalog_stage=catalog_options.stage_name,
            repo_root=repo_root,
            patch_options=stage_patch_plan_options,
            stage_patterns=stage_patterns,
        )

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
        help=(
            "Major workflow stage used to infer default catalog metadata and "
            "stage-template guidance"
        ),
    )
    parser.add_argument(
        "--catalog-stage",
        help=(
            "Catalog stage_name for generated WorkflowStepSpec entries "
            "(default: inferred from --major-stage)"
        ),
    )
    parser.add_argument(
        "--catalog-order-start",
        type=int,
        help=(
            "Base order for catalog entries; preprocess uses this value, run/postprocess "
            "use +10/+20 (default: append after current max order)"
        ),
    )
    parser.add_argument(
        "--enabled-flag-attr",
        help=(
            "Workflow settings attribute gating this model in schema filtering "
            "(use 'none' to omit)"
        ),
    )
    parser.add_argument(
        "--enabled-model-attr",
        help=(
            "settings.run.models attribute paired with --enabled-flag-attr "
            "(use 'none' to omit)"
        ),
    )
    parser.add_argument(
        "--provenance-builder-key",
        help=(
            "Optional catalog provenance builder key for generated steps "
            "(for pilates/utils/consist_config.py dispatch)"
        ),
    )
    parser.add_argument(
        "--stage-pattern",
        dest="stage_patterns",
        action="append",
        choices=[STAGE_PATTERN_LINEAR, STAGE_PATTERN_ITERATIVE],
        help=(
            "Stage-template snippet pattern to generate. "
            "Repeat for multiple patterns (default: both linear + iterative)."
        ),
    )
    parser.add_argument(
        "--stage-patch-plan",
        action="store_true",
        help=(
            "Generate a guided stage patch artifact under "
            "docs/checklists/stage_templates/ with insertion anchors and "
            "ready-to-paste snippets."
        ),
    )
    parser.add_argument(
        "--stage-target-module",
        help=(
            "Target stage module path for the stage patch plan "
            "(default: inferred from --major-stage)"
        ),
    )
    parser.add_argument(
        "--stage-target-function",
        help=(
            "Target stage function identifier for the stage patch plan "
            "(default: inferred from --major-stage)"
        ),
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
    catalog_options = _resolve_catalog_options(
        major_stage=args.major_stage,
        catalog_stage_arg=args.catalog_stage,
        enabled_flag_attr_arg=args.enabled_flag_attr,
        enabled_model_attr_arg=args.enabled_model_attr,
        provenance_builder_key_arg=args.provenance_builder_key,
        order_start_arg=args.catalog_order_start,
    )
    stage_patterns = _resolve_stage_patterns(args.stage_patterns)
    stage_patch_plan_options = _resolve_stage_patch_plan_options(
        major_stage=args.major_stage,
        enabled=args.stage_patch_plan,
        stage_target_module_arg=args.stage_target_module,
        stage_target_function_arg=args.stage_target_function,
    )

    repo_root = args.repo_root.resolve()

    actions: List[Tuple[str, Path, bool]] = []
    try:
        for path, changed in _create_model_files(
            repo_root,
            spec,
            catalog_options=catalog_options,
            stage_patterns=stage_patterns,
            stage_patch_plan_options=stage_patch_plan_options,
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

        path, changed = _upsert_workflow_catalog(
            repo_root,
            spec,
            catalog_options=catalog_options,
            dry_run=args.dry_run,
        )
        actions.append(("update", path, changed))

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(
        f"[{mode}] scaffold model={spec.model} class_prefix={spec.class_prefix} "
        f"catalog_stage={catalog_options.stage_name} "
        f"stage_patterns={','.join(stage_patterns)}"
    )
    for action, path, changed in actions:
        status = "changed" if changed else "no-op"
        print(f"  - {action}: {path} ({status})")

    print("\nNext step: complete the generated checklist:")
    print(f"  {repo_root / 'docs' / 'checklists' / f'add_model_{spec.model}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
