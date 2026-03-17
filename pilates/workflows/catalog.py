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
from pilates.activitysim.outputs import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
)
from pilates.workflows.artifact_keys import (
    ASIM_MUTABLE_DATA_DIR,
    ASIM_OUTPUT_DIR,
    BEAM_CONFIG_FILE,
    BEAM_EXPERIENCED_PLANS_XML,
    BEAM_FULL_SKIMS,
    BEAM_HOUSEHOLDS_IN,
    BEAM_MUTABLE_DATA_DIR,
    BEAM_OUTPUT_DIR,
    BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
    BEAM_OUTPUT_PLANS_XML,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    BEAM_PLANS_OUT,
    FINAL_SKIMS_OMX,
    ATLAS_OUTPUT_DIR,
    ASIM_SHARROW_CACHE_DIR,
    LINKSTATS,
    LINKSTATS_WARMSTART,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_DATASTORE_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
    USIM_INPUT_NEXT,
    USIM_MUTABLE_DATA_DIR,
    ZARR_SKIMS,
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
    output_keys: Tuple[str, ...] = ()
    optional_input_keys: Tuple[str, ...] = ()
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


_URBANSIM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="urbansim")
_ATLAS_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="atlas")
_ACTIVITYSIM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="activitysim")
_BEAM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="beam")


def _ordered_unique(*groups: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(key for group in groups for key in group))


_ACTIVITYSIM_RUN_OUTPUT_KEYS = ActivitySimRunOutputs.declared_output_keys()
_BEAM_RUN_OUTPUT_KEYS = BeamRunOutputs.declared_output_keys()
_ACTIVITYSIM_POSTPROCESS_OUTPUT_KEYS = _ordered_unique(
    _ACTIVITYSIM_RUN_OUTPUT_KEYS,
    (USIM_INPUT_NEXT, USIM_DATASTORE_H5),
)
_BEAM_POSTPROCESS_OUTPUT_KEYS = _ordered_unique(
    _BEAM_RUN_OUTPUT_KEYS,
    (BEAM_OUTPUT_PLANS_XML, BEAM_OUTPUT_EXPERIENCED_PLANS_XML, BEAM_EXPERIENCED_PLANS_XML, ZARR_SKIMS, FINAL_SKIMS_OMX),
)


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
        input_keys=(USIM_H5_UPDATED,),
        optional_input_keys=(USIM_DATASTORE_CURRENT_H5, USIM_DATASTORE_BASE_H5),
        output_keys=(
            ASIM_MUTABLE_DATA_DIR,
            ASIM_LAND_USE_IN,
            ASIM_HOUSEHOLDS_IN,
            ASIM_PERSONS_IN,
            ASIM_OMX_SKIMS,
        ),
        depends_on=(),
        holder_inputs=(),
        upstream_step_inputs=(),
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
        input_keys=(),
        output_keys=(ZARR_SKIMS, ASIM_SHARROW_CACHE_DIR),
        tracked=False,
        depends_on=("activitysim_preprocess",),
        holder_inputs=("activitysim_preprocess",),
        upstream_step_inputs=("activitysim_preprocess",),
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
        input_keys=(ZARR_SKIMS,),
        output_keys=(
            ASIM_OUTPUT_DIR,
            *_ACTIVITYSIM_RUN_OUTPUT_KEYS,
        ),
        depends_on=("activitysim_preprocess",),
        holder_inputs=("activitysim_preprocess",),
        upstream_step_inputs=("activitysim_preprocess",),
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
        input_keys=(
            ASIM_HOUSEHOLDS_IN,
            ASIM_PERSONS_IN,
            ASIM_LAND_USE_IN,
            ASIM_OMX_SKIMS,
            ZARR_SKIMS,
            USIM_DATASTORE_CURRENT_H5,
            USIM_FORECAST_OUTPUT,
        ),
        output_keys=(
            ASIM_OUTPUT_DIR,
            *_ACTIVITYSIM_POSTPROCESS_OUTPUT_KEYS,
        ),
        depends_on=("activitysim_run",),
        holder_inputs=("activitysim_run",),
        upstream_step_inputs=("activitysim_run",),
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
        input_keys=(BEAM_CONFIG_FILE,),
        output_keys=(
            BEAM_MUTABLE_DATA_DIR,
            *BeamPreprocessOutputs.declared_output_keys(),
            LINKSTATS_WARMSTART,
        ),
        depends_on=("activitysim_postprocess",),
        holder_inputs=("activitysim_postprocess",),
        upstream_step_inputs=("activitysim_postprocess",),
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
        input_keys=(BEAM_CONFIG_FILE,),
        optional_input_keys=(
            LINKSTATS_WARMSTART,
            BEAM_OUTPUT_PLANS_XML,
            BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            BEAM_EXPERIENCED_PLANS_XML,
        ),
        output_keys=(
            BEAM_OUTPUT_DIR,
            *_BEAM_RUN_OUTPUT_KEYS,
            BEAM_OUTPUT_PLANS_XML,
            BEAM_OUTPUT_EXPERIENCED_PLANS_XML,
            BEAM_EXPERIENCED_PLANS_XML,
        ),
        dynamic_output_families=(
            "linkstats_{year}_{iteration}",
            "linkstats_parquet_{year}_{iteration}",
            "linkstats_unmodified_{year}_{iteration}",
            "linkstats_unmodified_parquet_{year}_{iteration}",
            "events_{year}_{iteration}",
            "events_parquet_{year}_{iteration}",
            "raw_od_skims_{year}_{iteration}",
            "raw_od_skims_zarr_{year}_{iteration}",
            "beam_plans_{year}_{iteration}",
            "beam_experienced_plans_{year}_{iteration}",
            "beam_output_*",
        ),
        depends_on=("beam_preprocess",),
        holder_inputs=("beam_preprocess",),
        upstream_step_inputs=("beam_preprocess",),
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
        input_keys=(),
        output_keys=(
            BEAM_OUTPUT_DIR,
            *_BEAM_POSTPROCESS_OUTPUT_KEYS,
        ),
        dynamic_output_families=(
            "events_parquet_{year}_{iteration}",
            "path_traversal_links_{year}_{iteration}",
        ),
        depends_on=("beam_run",),
        holder_inputs=("beam_run",),
        upstream_step_inputs=("beam_run",),
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
        input_keys=(),
        output_keys=(BEAM_FULL_SKIMS,),
        optional=True,
        depends_on=("beam_preprocess",),
        holder_inputs=("beam_preprocess",),
        upstream_step_inputs=("beam_preprocess",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
        provenance=_BEAM_PROVENANCE,
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
