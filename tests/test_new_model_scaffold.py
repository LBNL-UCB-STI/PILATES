from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


def _seed_minimal_repo(repo_root: Path) -> None:
    (repo_root / "pilates/generic").mkdir(parents=True, exist_ok=True)
    (repo_root / "pilates/workflows/steps").mkdir(parents=True, exist_ok=True)
    (repo_root / "pilates/workflows").mkdir(parents=True, exist_ok=True)

    (repo_root / "pilates/generic/model_factory.py").write_text(
        dedent(
            """
            from pilates.existing.preprocessor import ExistingPreprocessor
            from pilates.existing.runner import ExistingRunner
            from pilates.existing.postprocessor import ExistingPostprocessor


            class ModelFactory:
                _registry = {
                    "existing": {
                        "preprocessor": ExistingPreprocessor,
                        "runner": ExistingRunner,
                        "postprocessor": ExistingPostprocessor,
                    },
                }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "pilates/workflows/steps/__init__.py").write_text(
        dedent(
            """
            from .postprocessing import make_postprocessing_step  # noqa: F401
            from . import postprocessing, shared  # noqa: F401,E402
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "pilates/workflows/steps/shared.py").write_text(
        dedent(
            """
            from dataclasses import dataclass
            from typing import Any, Optional

            from pilates.existing.outputs import ExistingOutputs
            from pilates.workflows.step_consist_meta import consist_step_meta


            @dataclass
            class StepOutputsHolder:
                existing_preprocess: Optional[ExistingOutputs] = None

                def set_attribute(self, step_name: str, outputs: Any) -> None:
                    attr = step_name.replace("-", "_")
                    setattr(self, attr, outputs)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "pilates/workflows/catalog.py").write_text(
        dedent(
            """
            from dataclasses import dataclass
            from typing import Any, Optional, Tuple, Type

            from pilates.existing.outputs import (
                ExistingPreprocessOutputs,
            )


            @dataclass(frozen=True)
            class WorkflowStepProvenanceSpec:
                builder_key: str


            @dataclass(frozen=True)
            class WorkflowStepSpec:
                step_name: str
                model_name: str
                phase: str
                stage_name: str
                order: int
                outputs_class: Optional[Type[Any]] = None
                depends_on: Tuple[str, ...] = ()
                holder_inputs: Tuple[str, ...] = ()
                enabled_flag_attr: Optional[str] = None
                enabled_model_attr: Optional[str] = None
                provenance: Optional[WorkflowStepProvenanceSpec] = None


            WORKFLOW_STEP_SPECS: Tuple[WorkflowStepSpec, ...] = (
                WorkflowStepSpec(
                    step_name="existing_preprocess",
                    model_name="existing_preprocess",
                    phase="preprocess",
                    stage_name="land_use",
                    order=10,
                    outputs_class=ExistingPreprocessOutputs,
                ),
            )
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "run.py").write_text(
        dedent(
            """
            from typing import Any, Callable, Dict, List

            from pilates.workflows.catalog import schema_step_names
            from pilates.workflows.steps import (
                StepOutputsHolder,
                make_existing_preprocess_step,
                validate_workflow_step_contracts,
            )


            def _build_schema_steps() -> List[Callable[..., Any]]:
                coupler = object()
                outputs_holder = StepOutputsHolder()
                step_factories: Dict[str, Callable[..., Any]] = {
                    "existing_preprocess": make_existing_preprocess_step,
                }
                ordered_steps = schema_step_names()
                return [
                    step_factories[step_name](coupler=coupler, outputs_holder=outputs_holder)
                    for step_name in ordered_steps
                ]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_scaffold_generates_catalog_era_wiring_and_templates(tmp_path: Path) -> None:
    _seed_minimal_repo(tmp_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts/new_model_scaffold.py"

    subprocess.run(
        [sys.executable, str(script_path), "freight", "--repo-root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    checklist_path = tmp_path / "docs/checklists/add_model_freight.md"
    checklist_text = checklist_path.read_text(encoding="utf-8")
    assert "STEP_OUTPUTS_CLASSES" not in checklist_text
    assert "docs/checklists/stage_templates/add_model_freight.linear.py.snippet" in checklist_text
    assert "docs/checklists/stage_templates/add_model_freight.iterative.py.snippet" in checklist_text

    assert (
        tmp_path
        / "docs/checklists/stage_templates/add_model_freight.linear.py.snippet"
    ).exists()
    assert (
        tmp_path
        / "docs/checklists/stage_templates/add_model_freight.iterative.py.snippet"
    ).exists()

    shared_text = (tmp_path / "pilates/workflows/steps/shared.py").read_text(encoding="utf-8")
    assert "from pilates.freight.outputs import (" in shared_text
    assert "freight_preprocess: Optional[FreightPreprocessOutputs] = None" in shared_text
    assert '"freight_preprocess": FreightPreprocessOutputs' not in shared_text

    catalog_text = (tmp_path / "pilates/workflows/catalog.py").read_text(encoding="utf-8")
    ast.parse(catalog_text)
    assert (
        "from pilates.existing.outputs import (\n"
        "    ExistingPreprocessOutputs,\n"
        ")\n\n"
        "from pilates.freight.outputs import ("
    ) in catalog_text
    assert 'step_name="freight_preprocess"' in catalog_text
    assert 'step_name="freight_run"' in catalog_text
    assert 'step_name="freight_postprocess"' in catalog_text
    assert 'stage_name="traffic_assignment"' in catalog_text
    assert 'enabled_flag_attr="traffic_assignment_enabled"' in catalog_text
    assert 'enabled_model_attr="travel"' in catalog_text
    assert 'depends_on=("freight_preprocess",)' in catalog_text
    assert 'holder_inputs=("freight_run",)' in catalog_text

    run_text = (tmp_path / "run.py").read_text(encoding="utf-8")
    assert "make_freight_preprocess_step" in run_text
    assert '"freight_preprocess": make_freight_preprocess_step' in run_text


@pytest.mark.parametrize(
    ("model", "major_stage", "extra_args", "expected_stage", "expected_flag", "expected_model"),
    [
        (
            "housing_supply",
            "land_use",
            [],
            "land_use",
            "land_use_enabled",
            "land_use",
        ),
        (
            "auto_ownership",
            "vehicle_ownership_model",
            [],
            "vehicle_ownership_model",
            "vehicle_ownership_model_enabled",
            "vehicle_ownership",
        ),
        (
            "network_ops",
            "supply_demand_loop",
            [],
            "traffic_assignment",
            "traffic_assignment_enabled",
            "travel",
        ),
        (
            "demand_tuning",
            "supply_demand_loop",
            ["--catalog-stage", "activity_demand"],
            "activity_demand",
            "activity_demand_enabled",
            "activity_demand",
        ),
    ],
)
def test_scaffold_catalog_defaults_apply_enablement_mapping(
    tmp_path: Path,
    model: str,
    major_stage: str,
    extra_args: list[str],
    expected_stage: str,
    expected_flag: str,
    expected_model: str,
) -> None:
    _seed_minimal_repo(tmp_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts/new_model_scaffold.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            model,
            "--repo-root",
            str(tmp_path),
            "--major-stage",
            major_stage,
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    catalog_text = (tmp_path / "pilates/workflows/catalog.py").read_text(encoding="utf-8")
    assert f'step_name="{model}_preprocess"' in catalog_text
    assert f'stage_name="{expected_stage}"' in catalog_text
    assert f'enabled_flag_attr="{expected_flag}"' in catalog_text
    assert f'enabled_model_attr="{expected_model}"' in catalog_text


def test_scaffold_dry_run_reports_actions_without_writing(tmp_path: Path) -> None:
    _seed_minimal_repo(tmp_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts/new_model_scaffold.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "freight",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            "--stage-patch-plan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "[DRY-RUN] scaffold model=freight class_prefix=Freight "
        "catalog_stage=traffic_assignment stage_patterns=linear,iterative"
    ) in result.stdout
    assert "- create:" in result.stdout
    assert "- update:" in result.stdout
    assert "docs/checklists/stage_templates/add_model_freight.stage_patch.md" in result.stdout
    assert not (tmp_path / "docs/checklists/add_model_freight.md").exists()
    assert not (
        tmp_path / "docs/checklists/stage_templates/add_model_freight.stage_patch.md"
    ).exists()


def test_scaffold_generates_stage_patch_plan_artifact(tmp_path: Path) -> None:
    _seed_minimal_repo(tmp_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts/new_model_scaffold.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "freight",
            "--repo-root",
            str(tmp_path),
            "--stage-pattern",
            "iterative",
            "--stage-patch-plan",
            "--stage-target-module",
            "pilates/workflows/stages/custom_supply.py",
            "--stage-target-function",
            "run_custom_supply_stage",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stage_patch_path = (
        tmp_path / "docs/checklists/stage_templates/add_model_freight.stage_patch.md"
    )
    assert stage_patch_path.exists()
    stage_patch_text = stage_patch_path.read_text(encoding="utf-8")

    assert (
        "Stage module to edit: `pilates/workflows/stages/custom_supply.py`"
        in stage_patch_text
    )
    assert "Stage function to edit: `run_custom_supply_stage`" in stage_patch_text
    assert 'Catalog stage metadata: `stage_name="traffic_assignment"`' in stage_patch_text
    assert "`freight_preprocess`" in stage_patch_text
    assert "`freight_run`" in stage_patch_text
    assert "`freight_postprocess`" in stage_patch_text
    assert "### `iterative` `StepRef` block" in stage_patch_text
    assert "make_freight_preprocess_step" in stage_patch_text
    assert "outputs_holder=outputs_holder_iteration" in stage_patch_text
    assert "Suggested insertion anchor in function body:" in stage_patch_text

    checklist_text = (tmp_path / "docs/checklists/add_model_freight.md").read_text(
        encoding="utf-8"
    )
    assert "docs/checklists/stage_templates/add_model_freight.stage_patch.md" in checklist_text
