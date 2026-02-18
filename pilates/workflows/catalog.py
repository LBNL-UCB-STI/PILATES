from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type

from pilates.activitysim.outputs import (
    ActivitySimPostprocessOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
)
from pilates.atlas.outputs import (
    AtlasPostprocessOutputs,
    AtlasPreprocessOutputs,
    AtlasRunOutputs,
)
from pilates.beam.outputs import (
    BeamFullSkimOutputs,
    BeamPostprocessOutputs,
    BeamPreprocessOutputs,
    BeamRunOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
)


@dataclass(frozen=True)
class WorkflowStepSpec:
    step_name: str
    model_name: str
    phase: str
    stage_name: str
    order: int
    outputs_class: Optional[Type[Any]] = None
    optional: bool = False
    tracked: bool = True
    include_in_schema: bool = True
    depends_on: Tuple[str, ...] = ()
    holder_inputs: Tuple[str, ...] = ()
    enabled_flag_attr: Optional[str] = None
    enabled_model_attr: Optional[str] = None


WORKFLOW_STEP_SPECS: Tuple[WorkflowStepSpec, ...] = (
    WorkflowStepSpec(
        step_name="urbansim_preprocess",
        model_name="urbansim_preprocess",
        phase="preprocess",
        stage_name="land_use",
        order=10,
        outputs_class=UrbanSimPreprocessOutputs,
        depends_on=(),
        holder_inputs=(),
        enabled_flag_attr="land_use_enabled",
        enabled_model_attr="land_use",
    ),
    WorkflowStepSpec(
        step_name="urbansim_run",
        model_name="urbansim_run",
        phase="run",
        stage_name="land_use",
        order=20,
        outputs_class=UrbanSimRunOutputs,
        depends_on=("urbansim_preprocess",),
        holder_inputs=("urbansim_preprocess",),
        enabled_flag_attr="land_use_enabled",
        enabled_model_attr="land_use",
    ),
    WorkflowStepSpec(
        step_name="urbansim_postprocess",
        model_name="urbansim_postprocess",
        phase="postprocess",
        stage_name="land_use",
        order=30,
        outputs_class=UrbanSimPostprocessOutputs,
        depends_on=("urbansim_run",),
        holder_inputs=("urbansim_run",),
        enabled_flag_attr="land_use_enabled",
        enabled_model_attr="land_use",
    ),
    WorkflowStepSpec(
        step_name="atlas_preprocess",
        model_name="atlas_preprocess",
        phase="preprocess",
        stage_name="vehicle_ownership_model",
        order=40,
        outputs_class=AtlasPreprocessOutputs,
        depends_on=(),
        holder_inputs=(),
        enabled_flag_attr="vehicle_ownership_model_enabled",
        enabled_model_attr="vehicle_ownership",
    ),
    WorkflowStepSpec(
        step_name="atlas_run",
        model_name="atlas_run",
        phase="run",
        stage_name="vehicle_ownership_model",
        order=50,
        outputs_class=AtlasRunOutputs,
        depends_on=("atlas_preprocess",),
        holder_inputs=("atlas_preprocess",),
        enabled_flag_attr="vehicle_ownership_model_enabled",
        enabled_model_attr="vehicle_ownership",
    ),
    WorkflowStepSpec(
        step_name="atlas_postprocess",
        model_name="atlas_postprocess",
        phase="postprocess",
        stage_name="vehicle_ownership_model",
        order=60,
        outputs_class=AtlasPostprocessOutputs,
        depends_on=("atlas_run",),
        holder_inputs=("atlas_run",),
        enabled_flag_attr="vehicle_ownership_model_enabled",
        enabled_model_attr="vehicle_ownership",
    ),
    WorkflowStepSpec(
        step_name="activitysim_preprocess",
        model_name="activitysim_preprocess",
        phase="preprocess",
        stage_name="activity_demand",
        order=70,
        outputs_class=ActivitySimPreprocessOutputs,
        depends_on=(),
        holder_inputs=(),
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
    ),
    WorkflowStepSpec(
        step_name="activitysim_compile",
        model_name="activitysim_compile",
        phase="compile",
        stage_name="activity_demand",
        order=80,
        outputs_class=None,
        tracked=False,
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
    ),
    WorkflowStepSpec(
        step_name="activitysim_run",
        model_name="activitysim_run",
        phase="run",
        stage_name="activity_demand",
        order=90,
        outputs_class=ActivitySimRunOutputs,
        depends_on=("activitysim_preprocess",),
        holder_inputs=("activitysim_preprocess",),
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
    ),
    WorkflowStepSpec(
        step_name="activitysim_postprocess",
        model_name="activitysim_postprocess",
        phase="postprocess",
        stage_name="activity_demand",
        order=100,
        outputs_class=ActivitySimPostprocessOutputs,
        depends_on=("activitysim_run",),
        holder_inputs=("activitysim_run",),
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
    ),
    WorkflowStepSpec(
        step_name="beam_preprocess",
        model_name="beam_preprocess",
        phase="preprocess",
        stage_name="traffic_assignment",
        order=110,
        outputs_class=BeamPreprocessOutputs,
        depends_on=("activitysim_postprocess",),
        holder_inputs=("activitysim_postprocess",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
    ),
    WorkflowStepSpec(
        step_name="beam_run",
        model_name="beam_run",
        phase="run",
        stage_name="traffic_assignment",
        order=120,
        outputs_class=BeamRunOutputs,
        depends_on=("beam_preprocess",),
        holder_inputs=("beam_preprocess",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
    ),
    WorkflowStepSpec(
        step_name="beam_postprocess",
        model_name="beam_postprocess",
        phase="postprocess",
        stage_name="traffic_assignment",
        order=130,
        outputs_class=BeamPostprocessOutputs,
        depends_on=("beam_run",),
        holder_inputs=("beam_run",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
    ),
    WorkflowStepSpec(
        step_name="beam_full_skim",
        model_name="beam_full_skim",
        phase="run",
        stage_name="traffic_assignment",
        order=140,
        outputs_class=BeamFullSkimOutputs,
        optional=True,
        depends_on=("beam_preprocess",),
        holder_inputs=("beam_preprocess",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
    ),
    WorkflowStepSpec(
        step_name="postprocessing",
        model_name="postprocessing",
        phase="postprocess",
        stage_name="postprocessing",
        order=150,
        outputs_class=None,
        tracked=False,
        include_in_schema=False,
    ),
)


def schema_step_names() -> Tuple[str, ...]:
    return tuple(
        spec.step_name
        for spec in sorted(WORKFLOW_STEP_SPECS, key=lambda item: item.order)
        if spec.include_in_schema
    )


def tracked_step_specs() -> Tuple[WorkflowStepSpec, ...]:
    return tuple(spec for spec in WORKFLOW_STEP_SPECS if spec.tracked)


def schema_step_specs(*, include_optional: bool = True) -> Tuple[WorkflowStepSpec, ...]:
    specs = [
        spec
        for spec in sorted(WORKFLOW_STEP_SPECS, key=lambda item: item.order)
        if spec.include_in_schema and (include_optional or not spec.optional)
    ]
    return tuple(specs)


def enabled_schema_step_models(
    settings: Any,
    *,
    is_model_enabled: Callable[..., bool],
    include_optional: bool = True,
) -> Set[str]:
    """
    Resolve enabled schema-step model identifiers for runtime settings.

    Parameters
    ----------
    settings : Any
        Runtime settings object.
    is_model_enabled : callable
        Callback that accepts keyword arguments ``flag_attr`` and ``model_attr``.
    include_optional : bool, default True
        Whether optional schema steps should be included.
    """
    enabled_models: Set[str] = set()
    for spec in schema_step_specs(include_optional=include_optional):
        flag_attr = spec.enabled_flag_attr
        model_attr = spec.enabled_model_attr
        if flag_attr is None or model_attr is None:
            enabled_models.add(spec.model_name)
            continue
        if is_model_enabled(
            settings,
            flag_attr=flag_attr,
            model_attr=model_attr,
        ):
            enabled_models.add(spec.model_name)
    return enabled_models


def step_outputs_classes_from_catalog() -> Dict[str, Type[Any]]:
    outputs: Dict[str, Type[Any]] = {}
    for spec in tracked_step_specs():
        if spec.outputs_class is None:
            raise RuntimeError(
                f"Tracked step {spec.step_name!r} is missing outputs_class."
            )
        outputs[spec.step_name] = spec.outputs_class
    return outputs


def step_dependencies_from_catalog() -> Dict[str, Dict[str, Sequence[str]]]:
    dependencies: Dict[str, Dict[str, Sequence[str]]] = {}
    for spec in tracked_step_specs():
        dependencies[spec.step_name] = {
            "depends_on": list(spec.depends_on),
            "holder_inputs": list(spec.holder_inputs),
        }
    return dependencies
