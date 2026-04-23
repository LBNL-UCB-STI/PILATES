from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type

from pilates.workflows.coupler_namespace import canonical_artifact_key_from_raw_key

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
from pilates.impacts.outputs import (
    ImpactsPostprocessOutputs,
    ImpactsPreprocessOutputs,
    ImpactsRunOutputs,
)
from pilates.urbansim.outputs import (
    UrbanSimPostprocessOutputs,
    UrbanSimPreprocessOutputs,
    UrbanSimRunOutputs,
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
    input_keys: Tuple[str, ...] = ()
    optional_input_keys: Tuple[str, ...] = ()
    output_keys: Tuple[str, ...] = ()
    optional_output_keys: Tuple[str, ...] = ()
    dynamic_input_families: Tuple[str, ...] = ()
    dynamic_output_families: Tuple[str, ...] = ()
    optional: bool = False
    tracked: bool = True
    include_in_schema: bool = True
    depends_on: Tuple[str, ...] = ()
    holder_inputs: Tuple[str, ...] = ()
    upstream_step_inputs: Tuple[str, ...] = ()
    enabled_flag_attr: Optional[str] = None
    enabled_model_attr: Optional[str] = None
    provenance: Optional[WorkflowStepProvenanceSpec] = None


@dataclass(frozen=True)
class WorkflowStepKeyMatch:
    step_name: str
    direction: str
    raw_key: str
    canonical_key: str
    declared: bool
    matched_via: Optional[str] = None
    matched_family: Optional[str] = None
    used_alias: bool = False

    @property
    def alias_note(self) -> str:
        if not self.used_alias:
            return ""
        return f" (canonicalized to '{self.canonical_key}')"


@dataclass(frozen=True)
class RestartProducerCandidate:
    key: str
    step_name: str
    stage_name: str
    phase: Optional[str]


_URBANSIM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="urbansim")
_ATLAS_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="atlas")
_ACTIVITYSIM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="activitysim")
_BEAM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="beam")
_IMPACTS_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="impacts")


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
        provenance=_URBANSIM_PROVENANCE,
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
        provenance=_URBANSIM_PROVENANCE,
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
        provenance=_URBANSIM_PROVENANCE,
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
        provenance=_ATLAS_PROVENANCE,
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
        provenance=_ATLAS_PROVENANCE,
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
        provenance=_ATLAS_PROVENANCE,
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
        provenance=_ACTIVITYSIM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="activitysim_compile",
        model_name="activitysim_compile",
        phase="compile",
        stage_name="activity_demand",
        order=80,
        outputs_class=None,
        tracked=False,
        depends_on=("activitysim_preprocess",),
        holder_inputs=("activitysim_preprocess",),
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
        provenance=_ACTIVITYSIM_PROVENANCE,
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
        provenance=_ACTIVITYSIM_PROVENANCE,
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
        provenance=_BEAM_PROVENANCE,
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
        provenance=_BEAM_PROVENANCE,
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
        provenance=_BEAM_PROVENANCE,
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
        provenance=_BEAM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="impacts_preprocess",
        model_name="impacts_preprocess",
        phase="preprocess",
        stage_name="postprocessing",
        order=145,
        outputs_class=ImpactsPreprocessOutputs,
        depends_on=(),
        holder_inputs=(),
        enabled_flag_attr="impacts_enabled",
        enabled_model_attr="impacts",
        provenance=_IMPACTS_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="impacts_run",
        model_name="impacts_run",
        phase="run",
        stage_name="postprocessing",
        order=146,
        outputs_class=ImpactsRunOutputs,
        depends_on=("impacts_preprocess",),
        holder_inputs=("impacts_preprocess",),
        enabled_flag_attr="impacts_enabled",
        enabled_model_attr="impacts",
        provenance=_IMPACTS_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="impacts_postprocess",
        model_name="impacts_postprocess",
        phase="postprocess",
        stage_name="postprocessing",
        order=147,
        outputs_class=ImpactsPostprocessOutputs,
        depends_on=("impacts_run",),
        holder_inputs=("impacts_run",),
        enabled_flag_attr="impacts_enabled",
        enabled_model_attr="impacts",
        provenance=_IMPACTS_PROVENANCE,
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


_STEP_SPECS_BY_STEP_NAME: Dict[str, WorkflowStepSpec] = {
    spec.step_name: spec for spec in WORKFLOW_STEP_SPECS
}
_STEP_SPECS_BY_MODEL_NAME: Dict[str, WorkflowStepSpec] = {
    spec.model_name: spec for spec in WORKFLOW_STEP_SPECS
}


def workflow_step_spec_for_step_name(step_name: str) -> Optional[WorkflowStepSpec]:
    return _STEP_SPECS_BY_STEP_NAME.get(step_name)


def workflow_step_spec_for_model_name(model_name: str) -> Optional[WorkflowStepSpec]:
    return _STEP_SPECS_BY_MODEL_NAME.get(model_name)


def workflow_step_key_match(
    step_name: str,
    key: str,
    *,
    direction: str,
) -> WorkflowStepKeyMatch:
    spec = workflow_step_spec_for_step_name(step_name)
    canonical_key = canonical_artifact_key_from_raw_key(key)
    if spec is None:
        return WorkflowStepKeyMatch(
            step_name=step_name,
            direction=direction,
            raw_key=key,
            canonical_key=canonical_key,
            declared=False,
            used_alias=(canonical_key != key),
        )

    if direction == "input":
        declared_keys = set(spec.input_keys) | set(spec.optional_input_keys)
        dynamic_families = spec.dynamic_input_families
    elif direction == "output":
        declared_keys = set(spec.output_keys) | set(spec.optional_output_keys)
        dynamic_families = spec.dynamic_output_families
    else:
        raise ValueError(f"Unsupported direction {direction!r}")

    if canonical_key in declared_keys:
        return WorkflowStepKeyMatch(
            step_name=step_name,
            direction=direction,
            raw_key=key,
            canonical_key=canonical_key,
            declared=True,
            matched_via="declared",
            used_alias=(canonical_key != key),
        )

    for family in dynamic_families:
        if family and "{" not in family and canonical_key.startswith(family):
            return WorkflowStepKeyMatch(
                step_name=step_name,
                direction=direction,
                raw_key=key,
                canonical_key=canonical_key,
                declared=True,
                matched_via="dynamic_family",
                matched_family=family,
                used_alias=(canonical_key != key),
            )

    return WorkflowStepKeyMatch(
        step_name=step_name,
        direction=direction,
        raw_key=key,
        canonical_key=canonical_key,
        declared=False,
        used_alias=(canonical_key != key),
    )


def workflow_step_key_is_declared(
    step_name: str,
    key: str,
    *,
    direction: str,
) -> bool:
    return workflow_step_key_match(step_name, key, direction=direction).declared


def workflow_step_declared_input_keys(step_name: str) -> Tuple[str, ...]:
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is None:
        return ()
    return tuple(dict.fromkeys((*spec.input_keys, *spec.optional_input_keys)))


def workflow_step_declared_output_keys(step_name: str) -> Tuple[str, ...]:
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is None:
        return ()
    return tuple(dict.fromkeys((*spec.output_keys, *spec.optional_output_keys)))


def workflow_step_contracts_by_name(
    settings: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    del settings
    contracts: Dict[str, Dict[str, Any]] = {}
    for spec in WORKFLOW_STEP_SPECS:
        contracts[spec.step_name] = {
            "step_name": spec.step_name,
            "stage_name": spec.stage_name,
            "phase": spec.phase,
            "depends_on": list(spec.depends_on),
            "input_keys": list(spec.input_keys),
            "optional_input_keys": list(spec.optional_input_keys),
            "upstream_step_inputs": list(spec.upstream_step_inputs),
            "output_keys": list(spec.output_keys),
            "optional_output_keys": list(spec.optional_output_keys),
            "dynamic_input_families": list(spec.dynamic_input_families),
            "dynamic_output_families": list(spec.dynamic_output_families),
            "optional": spec.optional,
        }
    return contracts


def restart_query_scope_for_step(step_name: str) -> Dict[str, Optional[str]]:
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is None:
        raise KeyError(f"Unknown restart query step name: {step_name}")
    if spec.stage_name == "activity_demand":
        return {
            "model": spec.step_name,
            "stage": f"activity_demand_{spec.phase}",
            "phase": spec.phase,
        }
    if spec.stage_name == "land_use":
        return {"model": spec.step_name, "stage": "land_use", "phase": spec.phase}
    if spec.stage_name == "vehicle_ownership_model":
        return {"model": spec.step_name, "stage": "atlas", "phase": spec.phase}
    if spec.stage_name == "traffic_assignment":
        return {"model": spec.step_name, "stage": "beam", "phase": spec.phase}
    if spec.stage_name == "postprocessing":
        return {"model": spec.step_name, "stage": "postprocessing", "phase": None}
    return {"model": spec.step_name, "stage": spec.stage_name, "phase": spec.phase}


def restart_artifact_producers(
    *,
    frontier_stage: Optional[str] = None,
    enabled_models: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[RestartProducerCandidate, ...]]:
    del frontier_stage, enabled_models
    producers: Dict[str, list[RestartProducerCandidate]] = {}
    for spec in sorted(WORKFLOW_STEP_SPECS, key=lambda item: item.order):
        for key in workflow_step_declared_output_keys(spec.step_name):
            producers.setdefault(key, []).append(
                RestartProducerCandidate(
                    key=key,
                    step_name=spec.step_name,
                    stage_name=spec.stage_name,
                    phase=spec.phase,
                )
            )
    return {key: tuple(candidates) for key, candidates in producers.items()}


def provenance_builder_key_for_step_name(step_name: str) -> Optional[str]:
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is None or spec.provenance is None:
        return None
    return spec.provenance.builder_key


def provenance_builder_key_for_model_name(model_name: str) -> Optional[str]:
    spec = workflow_step_spec_for_model_name(model_name)
    if spec is None or spec.provenance is None:
        return None
    return spec.provenance.builder_key


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


def runtime_step_dependencies_from_catalog() -> Dict[str, Dict[str, Sequence[str]]]:
    """
    Return runtime dependency specs for all declared workflow steps.

    Unlike ``step_dependencies_from_catalog()``, this includes intentional
    untracked steps such as ``activitysim_compile`` that still need startup
    ordering checks but do not participate in typed output reconstruction.
    """
    dependencies: Dict[str, Dict[str, Sequence[str]]] = {}
    for spec in WORKFLOW_STEP_SPECS:
        dependencies[spec.step_name] = {
            "depends_on": list(spec.depends_on),
            "holder_inputs": list(spec.holder_inputs),
        }
    return dependencies
