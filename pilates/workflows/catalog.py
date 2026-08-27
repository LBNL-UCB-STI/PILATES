from __future__ import annotations

from dataclasses import dataclass
import re
from fnmatch import fnmatchcase
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from pilates.activitysim.outputs import ASIM_OPTIONAL_RUN_OUTPUT_KEYS
from pilates.workflows.artifact_keys import (
    BEAM_VEHICLES_IN,
    FINAL_SKIMS_OMX,
    LINKSTATS_WARMSTART,
)
from pilates.workflows.coupler_namespace import canonical_artifact_key_from_raw_key

if TYPE_CHECKING:
    from pilates.workflows.step_definition import StepDefinition


@dataclass(frozen=True)
class WorkflowStepProvenanceSpec:
    builder_key: str


_OPTIONAL_OUTPUT_KEYS_BY_STEP: Dict[str, Tuple[str, ...]] = {
    "urbansim_run": ("usim_forecast_output",),
    "atlas_preprocess": ("atlas_accessibility_csv",),
    "activitysim_run": ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
    "activitysim_postprocess": ASIM_OPTIONAL_RUN_OUTPUT_KEYS,
    "beam_preprocess": (LINKSTATS_WARMSTART, BEAM_VEHICLES_IN),
    "beam_postprocess": (FINAL_SKIMS_OMX,),
}


@dataclass(frozen=True)
class WorkflowStepSpec:
    """
    Static catalog entry for a workflow step.

    ``step_name`` is the canonical workflow-step identity in the catalog and
    matches the Consist ``model=...`` value used by the decorated step
    functions.

    The catalog owns PILATES policy only: placement, order, enablement,
    dependencies, optionality, provenance policy, and open-ended semantic key
    families. Static Consist input/output/schema metadata belongs to the
    committed native ``StepDefinition`` and is projected below rather than
    duplicated here.
    """

    step_name: str
    phase: str
    stage_name: str
    order: int
    dynamic_input_families: Tuple[str, ...] = ()
    dynamic_output_families: Tuple[str, ...] = ()
    optional: bool = False
    tracked: bool = True
    include_in_schema: bool = True
    depends_on: Tuple[str, ...] = ()
    enabled_flag_attr: Optional[str] = None
    enabled_model_attr: Optional[str] = None
    provenance: Optional[WorkflowStepProvenanceSpec] = None

    @property
    def definition(self) -> "StepDefinition[Any]":
        """Return this catalog policy entry's committed native definition.

        ``steps.shared`` still imports catalog dependency policy while the
        native definition package initializes. Importing the package here,
        after that initialization boundary, avoids an import cycle without a
        second registry or any late registration: ``STEP_DEFINITIONS`` is the
        committed definition registry established by the native step package.
        """
        from pilates.workflows.steps import STEP_DEFINITIONS

        return STEP_DEFINITIONS[self.step_name]

    @property
    def _consist_metadata(self) -> Any:
        return self.definition.function.__consist_step__

    @property
    def input_keys(self) -> Tuple[str, ...]:
        inputs = self._consist_metadata.inputs or ()
        if isinstance(inputs, Mapping):
            return tuple(inputs)
        return tuple(inputs)

    @property
    def optional_input_keys(self) -> Tuple[str, ...]:
        return tuple(self._consist_metadata.optional_input_keys or ())

    @property
    def schema_output_keys(self) -> Tuple[str, ...]:
        return tuple(self._consist_metadata.schema_outputs or ())

    @property
    def output_keys(self) -> Tuple[str, ...]:
        optional_keys = set(self.optional_output_keys)
        return tuple(
            key for key in self.schema_output_keys if key not in optional_keys
        )

    @property
    def optional_output_keys(self) -> Tuple[str, ...]:
        return _OPTIONAL_OUTPUT_KEYS_BY_STEP.get(self.step_name, ())


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


@dataclass(frozen=True)
class RestartProducerOverride:
    key: str
    producer_step: str
    frontier_stages: FrozenSet[str] = frozenset()
    required_models: FrozenSet[str] = frozenset()
    priority: int = 0


_URBANSIM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="urbansim")
_ATLAS_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="atlas")
_ACTIVITYSIM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="activitysim")
_BEAM_PROVENANCE = WorkflowStepProvenanceSpec(builder_key="beam")


WORKFLOW_STEP_SPECS: Tuple[WorkflowStepSpec, ...] = (
    WorkflowStepSpec(
        step_name="urbansim_run",
        phase="run",
        stage_name="land_use",
        order=10,
        depends_on=(),
        enabled_flag_attr="land_use_enabled",
        enabled_model_attr="land_use",
        provenance=_URBANSIM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="urbansim_postprocess",
        phase="postprocess",
        stage_name="land_use",
        order=20,
        dynamic_output_families=(
            "usim_input_archive_{year}",
            "usim_input_merged_{year}",
        ),
        depends_on=("urbansim_run",),
        enabled_flag_attr="land_use_enabled",
        enabled_model_attr="land_use",
        provenance=_URBANSIM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="atlas_preprocess",
        phase="preprocess",
        stage_name="vehicle_ownership_model",
        order=40,
        depends_on=(),
        enabled_flag_attr="vehicle_ownership_model_enabled",
        enabled_model_attr="vehicle_ownership",
        provenance=_ATLAS_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="atlas_run",
        phase="run",
        stage_name="vehicle_ownership_model",
        order=50,
        dynamic_output_families=("householdv_{year}", "vehicles_{year}"),
        depends_on=("atlas_preprocess",),
        enabled_flag_attr="vehicle_ownership_model_enabled",
        enabled_model_attr="vehicle_ownership",
        provenance=_ATLAS_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="atlas_postprocess",
        phase="postprocess",
        stage_name="vehicle_ownership_model",
        order=60,
        dynamic_input_families=("householdv_{year}", "vehicles_{year}"),
        depends_on=("atlas_run",),
        enabled_flag_attr="vehicle_ownership_model_enabled",
        enabled_model_attr="vehicle_ownership",
        provenance=_ATLAS_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="activitysim_preprocess",
        phase="preprocess",
        stage_name="activity_demand",
        order=70,
        depends_on=(),
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
        provenance=_ACTIVITYSIM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="activitysim_run",
        phase="run",
        stage_name="activity_demand",
        order=90,
        depends_on=("activitysim_preprocess",),
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
        provenance=_ACTIVITYSIM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="activitysim_postprocess",
        phase="postprocess",
        stage_name="activity_demand",
        order=100,
        depends_on=("activitysim_run",),
        enabled_flag_attr="activity_demand_enabled",
        enabled_model_attr="activity_demand",
        provenance=_ACTIVITYSIM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="beam_preprocess",
        phase="preprocess",
        stage_name="traffic_assignment",
        order=110,
        depends_on=("activitysim_postprocess",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
        provenance=_BEAM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="beam_run",
        phase="run",
        stage_name="traffic_assignment",
        order=120,
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
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
        provenance=_BEAM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="beam_postprocess",
        phase="postprocess",
        stage_name="traffic_assignment",
        order=130,
        dynamic_input_families=(
            "events_parquet_{year}_{iteration}",
            "raw_od_skims_{year}_{iteration}",
            "raw_od_skims_zarr_{year}_{iteration}",
        ),
        dynamic_output_families=(
            "events_parquet_{year}_{iteration}",
            "path_traversal_links_{year}_{iteration}",
        ),
        depends_on=("beam_run",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
        provenance=_BEAM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="beam_full_skim",
        phase="run",
        stage_name="traffic_assignment",
        order=140,
        optional=True,
        depends_on=("beam_preprocess",),
        enabled_flag_attr="traffic_assignment_enabled",
        enabled_model_attr="travel",
        provenance=_BEAM_PROVENANCE,
    ),
    WorkflowStepSpec(
        step_name="postprocessing",
        phase="postprocess",
        stage_name="postprocessing",
        order=150,
        tracked=False,
        include_in_schema=False,
    ),
)


_STEP_SPECS_BY_STEP_NAME: Dict[str, WorkflowStepSpec] = {
    spec.step_name: spec for spec in WORKFLOW_STEP_SPECS
}


def workflow_step_spec_for_step_name(step_name: str) -> Optional[WorkflowStepSpec]:
    return _STEP_SPECS_BY_STEP_NAME.get(step_name)


def workflow_step_contracts_by_name(
    settings: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return a plain serializable catalog view for static inspection tools.

    ``settings`` remains accepted for callers that project this view alongside
    runtime policy. Static inputs and schema outputs themselves are read from
    the native decorated Consist contract, not specialized from a catalog copy.
    """
    contracts: Dict[str, Dict[str, Any]] = {}
    for spec in WORKFLOW_STEP_SPECS:
        contract = {
            "step_name": spec.step_name,
            "stage_name": spec.stage_name,
            "phase": spec.phase,
            "depends_on": list(spec.depends_on),
            "input_keys": list(spec.input_keys),
            "optional_input_keys": list(spec.optional_input_keys),
            "output_keys": list(spec.output_keys),
            "optional_output_keys": list(spec.optional_output_keys),
            "schema_outputs": list(spec.schema_output_keys),
            "dynamic_input_families": list(spec.dynamic_input_families),
            "dynamic_output_families": list(spec.dynamic_output_families),
            "optional": spec.optional,
        }
        contracts[spec.step_name] = contract
    return contracts


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


def restart_query_scope_for_step(step_name: str) -> Mapping[str, Optional[str]]:
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
        return {
            "model": spec.step_name,
            "stage": "land_use",
            "phase": spec.phase,
        }
    if spec.stage_name == "vehicle_ownership_model":
        return {
            "model": spec.step_name,
            "stage": "vehicle_ownership",
            "phase": spec.phase,
        }
    if spec.stage_name == "traffic_assignment":
        return {
            "model": spec.step_name,
            "stage": "beam",
            "phase": spec.phase,
        }
    if spec.stage_name == "postprocessing":
        return {
            "model": spec.step_name,
            "stage": "postprocessing",
            "phase": None,
        }
    return {
        "model": spec.step_name,
        "stage": spec.stage_name,
        "phase": spec.phase,
    }


def _restart_producer_candidates_by_key() -> Dict[
    str, Tuple[RestartProducerCandidate, ...]
]:
    candidates_by_key: Dict[str, List[RestartProducerCandidate]] = {}
    for spec in sorted(WORKFLOW_STEP_SPECS, key=lambda item: item.order):
        for key in workflow_step_declared_output_keys(spec.step_name):
            candidates_by_key.setdefault(key, []).append(
                RestartProducerCandidate(
                    key=key,
                    step_name=spec.step_name,
                    stage_name=spec.stage_name,
                    phase=spec.phase,
                )
            )
    return {key: tuple(candidates) for key, candidates in candidates_by_key.items()}


_RESTART_PRODUCER_OVERRIDES: Tuple[RestartProducerOverride, ...] = (
    RestartProducerOverride(
        key="beam_plans_asim_out",
        producer_step="activitysim_postprocess",
        frontier_stages=frozenset({"traffic_assignment"}),
        required_models=frozenset({"activitysim", "beam"}),
        priority=100,
    ),
    RestartProducerOverride(
        key="households_asim_out",
        producer_step="activitysim_postprocess",
        frontier_stages=frozenset({"traffic_assignment"}),
        required_models=frozenset({"activitysim", "beam"}),
        priority=100,
    ),
    RestartProducerOverride(
        key="persons_asim_out",
        producer_step="activitysim_postprocess",
        frontier_stages=frozenset({"traffic_assignment"}),
        required_models=frozenset({"activitysim", "beam"}),
        priority=100,
    ),
)


def restart_artifact_producers(
    *,
    frontier_stage: Optional[str] = None,
    enabled_models: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[RestartProducerCandidate, ...]]:
    """Return static restart producer candidates for inspection tooling."""
    enabled_model_set = frozenset(
        str(model)
        for model in (enabled_models or ())
        if model is not None and str(model).strip()
    )
    producers = _restart_producer_candidates_by_key()
    ordered: Dict[str, Tuple[RestartProducerCandidate, ...]] = {}

    for key, candidates in producers.items():
        priorities = {candidate.step_name: 0 for candidate in candidates}
        for override in _RESTART_PRODUCER_OVERRIDES:
            if override.key != key:
                continue
            if (
                override.frontier_stages
                and frontier_stage not in override.frontier_stages
            ):
                continue
            if override.required_models and not override.required_models.issubset(
                enabled_model_set
            ):
                continue
            if override.producer_step not in priorities:
                continue
            priorities[override.producer_step] = max(
                priorities[override.producer_step],
                override.priority,
            )

        ordered[key] = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -priorities.get(candidate.step_name, 0),
                    next(
                        (
                            spec.order
                            for spec in WORKFLOW_STEP_SPECS
                            if spec.step_name == candidate.step_name
                        ),
                        0,
                    ),
                    candidate.step_name,
                ),
            )
        )
    return ordered


def _family_pattern_matches_key(family: str, key: str) -> bool:
    pattern = re.sub(r"\{[^{}]+\}", "*", family)
    return bool(pattern) and fnmatchcase(key, pattern)


def workflow_step_key_match(
    step_name: str,
    key: str,
    *,
    direction: str,
) -> WorkflowStepKeyMatch:
    canonical_key = canonical_artifact_key_from_raw_key(key)
    used_alias = canonical_key != key
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is None:
        return WorkflowStepKeyMatch(
            step_name=step_name,
            direction=direction,
            raw_key=key,
            canonical_key=canonical_key,
            declared=False,
            used_alias=used_alias,
        )
    if direction == "input":
        if canonical_key in spec.input_keys:
            return WorkflowStepKeyMatch(
                step_name=step_name,
                direction=direction,
                raw_key=key,
                canonical_key=canonical_key,
                declared=True,
                matched_via="input_keys",
                used_alias=used_alias,
            )
        if canonical_key in spec.optional_input_keys:
            return WorkflowStepKeyMatch(
                step_name=step_name,
                direction=direction,
                raw_key=key,
                canonical_key=canonical_key,
                declared=True,
                matched_via="optional_input_keys",
                used_alias=used_alias,
            )
        for family in spec.dynamic_input_families:
            if _family_pattern_matches_key(family, canonical_key):
                return WorkflowStepKeyMatch(
                    step_name=step_name,
                    direction=direction,
                    raw_key=key,
                    canonical_key=canonical_key,
                    declared=True,
                    matched_via="dynamic_input_families",
                    matched_family=family,
                    used_alias=used_alias,
                )
        return WorkflowStepKeyMatch(
            step_name=step_name,
            direction=direction,
            raw_key=key,
            canonical_key=canonical_key,
            declared=False,
            used_alias=used_alias,
        )
    if direction == "output":
        if canonical_key in spec.output_keys:
            return WorkflowStepKeyMatch(
                step_name=step_name,
                direction=direction,
                raw_key=key,
                canonical_key=canonical_key,
                declared=True,
                matched_via="output_keys",
                used_alias=used_alias,
            )
        if canonical_key in spec.optional_output_keys:
            return WorkflowStepKeyMatch(
                step_name=step_name,
                direction=direction,
                raw_key=key,
                canonical_key=canonical_key,
                declared=True,
                matched_via="optional_output_keys",
                used_alias=used_alias,
            )
        for family in spec.dynamic_output_families:
            if _family_pattern_matches_key(family, canonical_key):
                return WorkflowStepKeyMatch(
                    step_name=step_name,
                    direction=direction,
                    raw_key=key,
                    canonical_key=canonical_key,
                    declared=True,
                    matched_via="dynamic_output_families",
                    matched_family=family,
                    used_alias=used_alias,
                )
        return WorkflowStepKeyMatch(
            step_name=step_name,
            direction=direction,
            raw_key=key,
            canonical_key=canonical_key,
            declared=False,
            used_alias=used_alias,
        )
    raise ValueError("direction must be 'input' or 'output'")


def workflow_step_key_is_declared(
    step_name: str,
    key: str,
    *,
    direction: str,
) -> bool:
    return workflow_step_key_match(step_name, key, direction=direction).declared


def provenance_builder_key_for_step_name(step_name: str) -> Optional[str]:
    spec = workflow_step_spec_for_step_name(step_name)
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
    Resolve enabled schema-step identifiers for runtime settings.

    The function name is retained for compatibility with existing callers, but
    the returned identifiers are the canonical schema ``step_name`` values that
    Consist sees as step models.

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
            enabled_models.add(spec.step_name)
            continue
        if is_model_enabled(
            settings,
            flag_attr=flag_attr,
            model_attr=model_attr,
        ):
            enabled_models.add(spec.step_name)
    return enabled_models


def step_dependencies_from_catalog() -> Dict[str, Dict[str, Sequence[str]]]:
    dependencies: Dict[str, Dict[str, Sequence[str]]] = {}
    for spec in tracked_step_specs():
        dependencies[spec.step_name] = {
            "depends_on": list(spec.depends_on),
        }
    return dependencies


def runtime_step_dependencies_from_catalog() -> Dict[str, Dict[str, Sequence[str]]]:
    """
    Return runtime dependency specs for all declared workflow steps.

    Unlike ``step_dependencies_from_catalog()``, this preserves all catalog
    dependencies used by startup ordering checks and typed output reconstruction.
    """
    dependencies: Dict[str, Dict[str, Sequence[str]]] = {}
    for spec in WORKFLOW_STEP_SPECS:
        dependencies[spec.step_name] = {
            "depends_on": list(spec.depends_on),
        }
    return dependencies
