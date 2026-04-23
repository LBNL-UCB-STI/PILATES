from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Set

from consist.core.step_context import StepContext
from consist.types import BindingResult, CacheOptions, ExecutionOptions, OutputPolicyOptions

from pilates.runtime.cache_recovery import (
    cache_miss_audit_fields,
    log_cache_miss_explanation,
    run_with_cache_recovery,
)
from pilates.runtime.consist_audit import emit_consist_audit_event
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import (
    artifact_to_existing_path,
    artifact_to_path,
    record_published_coupler_keys,
    resolve_existing_path,
    resolve_artifact_from_value,
    set_coupler_from_artifact,
)
from pilates.workflows.catalog import (
    workflow_step_contracts_by_name,
    workflow_step_declared_output_keys,
    workflow_step_key_is_declared,
    workflow_step_key_match,
    workflow_step_spec_for_step_name,
)
from pilates.beam.outputs import (
    BeamPreprocessOutputs,
)
from pilates.workflows.coupler_namespace import canonical_artifact_key_from_raw_key
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
    LINKSTATS_WARMSTART,
)
from pilates.utils.consist_types import CouplerProtocol
from pilates.utils.state_access import current_year as state_current_year, iteration_index
from pilates.utils.step_manifest import load_step_manifest, save_step_manifest
from pilates.workflows.outputs_base import (
    ValidationContext,
    declared_outputs_for_step_outputs_class,
    deserialize_step_outputs,
    required_outputs_for_step_outputs_class,
    step_output_mapping,
    serialize_step_outputs,
)
from pilates.workflows.step_io import expected_inputs_for_step
from pilates.workflows.step_runner import common_runtime_kwargs
from pilates.workflows.tracker_outputs import (
    load_tracker_run_outputs,
    merge_canonical_output_mappings,
)
from pilates.workflows.steps import (
    STEP_OUTPUTS_CLASSES,
    shared as steps_shared,
    StepOutputsHolder,
    validate_step_ready,
    validate_workflow_step_contracts,
)

logger = logging.getLogger(__name__)
_STEP_CONTRACT_WARNING_SIGNATURES: set[tuple[Any, ...]] = set()


@dataclass(frozen=True)
class StepRef:
    """
    Declarative reference to a workflow step invocation.
    """

    name: str
    step_func: Callable[..., None]
    input_keys: Optional[Sequence[str]] = None
    inputs: Optional[Dict[str, Any]] = None
    binding: Optional[Any] = None
    output_paths: Optional[Dict[str, Any]] = None
    output_paths_provider: Optional[Callable[..., Optional[Mapping[str, Any]]]] = None
    output_replayer: Optional[Callable[..., None]] = None
    output_recoverer: Optional[Callable[..., Optional[Any]]] = None
    cache_hydration: Optional[str] = None
    cache_mode: Optional[str] = None
    load_inputs: Optional[bool] = None
    input_binding: Optional[str] = None
    input_paths: Optional[Mapping[str, Any]] = None
    input_materialization: Optional[str] = None
    output_missing: Optional[Literal["warn", "error", "ignore"]] = None
    output_mismatch: Optional[Literal["warn", "error", "ignore"]] = None
    model: Optional[str] = None
    year: Optional[int] = None
    iteration: Optional[int] = None
    phase: Optional[str] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        self._normalize_callable_hook(
            "output_paths_provider",
            getattr(self.step_func, "pilates_output_paths_provider", None),
        )
        self._normalize_callable_hook(
            "output_replayer",
            getattr(self.step_func, "pilates_output_replayer", None),
        )
        self._normalize_callable_hook(
            "output_recoverer",
            getattr(self.step_func, "pilates_output_recoverer", None),
        )

    def _normalize_callable_hook(self, field_name: str, fallback: Any) -> None:
        value = getattr(self, field_name)
        if value is None:
            value = fallback
            object.__setattr__(self, field_name, value)
        if value is not None and not callable(value):
            raise TypeError(
                f"Step '{self.name}' {field_name} must be callable or None."
            )


def _infer_phase(step_name: str) -> Optional[str]:
    if "_" not in step_name:
        return None
    return step_name.rsplit("_", 1)[-1] or None


def _warn_once(signature: tuple[Any, ...], message: str, *args: Any) -> None:
    if signature in _STEP_CONTRACT_WARNING_SIGNATURES:
        return
    _STEP_CONTRACT_WARNING_SIGNATURES.add(signature)
    logger.warning(message, *args)


def _normalize_key_iter(values: Optional[Any]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(key) for key in values]


def _resolved_input_keys_for_step(
    *,
    step: StepRef,
    binding: Optional[BindingResult],
) -> Optional[Set[str]]:
    if binding is not None:
        if isinstance(binding.inputs, Mapping):
            return {str(key) for key in binding.inputs.keys()}
        return None
    if step.inputs is not None:
        return {str(key) for key in step.inputs.keys()}
    return None


def _explicit_input_keys_for_step(
    *,
    step: StepRef,
    binding: Optional[BindingResult],
) -> Set[str]:
    explicit_keys: Set[str] = set()
    if binding is not None and isinstance(binding.inputs, Mapping):
        explicit_keys.update(str(key) for key in binding.inputs.keys())
    if step.inputs is not None:
        explicit_keys.update(str(key) for key in step.inputs.keys())
    return explicit_keys


def _warn_for_undeclared_step_inputs(
    *,
    step_name: str,
    input_keys: Optional[Sequence[str]],
    inputs: Optional[Mapping[str, Any]],
    binding: Optional[Any],
    settings: Any,
    state: Any,
    workspace: Any,
) -> None:
    declared_inputs: list[str] = []
    if binding is not None:
        if binding.inputs:
            declared_inputs.extend(str(key) for key in binding.inputs.keys())
        declared_inputs.extend(_normalize_key_iter(getattr(binding, "input_keys", None)))
        declared_inputs.extend(
            _normalize_key_iter(getattr(binding, "optional_input_keys", None))
        )
    else:
        if inputs:
            declared_inputs.extend(str(key) for key in inputs.keys())
        if input_keys:
            declared_inputs.extend(str(key) for key in input_keys)
    component_expected_inputs = expected_inputs_for_step(
        step_name,
        settings,
        state,
        workspace,
    )
    component_expected_keys = set(str(key) for key in component_expected_inputs.keys())
    spec = workflow_step_spec_for_step_name(step_name)
    dynamic_families = tuple(spec.dynamic_input_families) if spec is not None else ()
    for key in dict.fromkeys(declared_inputs):
        if key in component_expected_keys:
            continue
        match = workflow_step_key_match(step_name, key, direction="input")
        if match.declared:
            continue
        if dynamic_families:
            message = (
                "[CONTRACT-ENFORCEMENT][%s] Step '%s' received undeclared input "
                "key '%s'%s from step resolution; it matches no declared input "
                "key and no dynamic input family %s."
            )
            args = (
                step_name,
                step_name,
                key,
                match.alias_note,
                dynamic_families,
            )
        else:
            message = (
                "[CONTRACT-ENFORCEMENT][%s] Step '%s' received undeclared input "
                "key '%s'%s from step resolution; the step declares no dynamic "
                "input families."
            )
            args = (
                step_name,
                step_name,
                key,
                match.alias_note,
            )
        _warn_once(
            ("undeclared_step_input", step_name, key),
            message,
            *args,
        )


def _resolved_step_epoch_identity(
    *,
    step: StepRef,
    state: Any,
    default_iteration: int,
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    model_name = step.model
    if model_name is None:
        step_meta = getattr(step.step_func, "__consist_step__", None)
        raw_model = getattr(step_meta, "model", None)
        if raw_model is not None:
            model_name = str(raw_model)
    year = step.year
    if year is None:
        year = getattr(state, "year", None)
    iteration = step.iteration if step.iteration is not None else default_iteration
    return model_name, year, iteration


def _step_scope_fields(
    *,
    stage_name: str,
    step_name: str,
    state: Any,
    run_id: Optional[str] = None,
    cache_hit: Optional[bool] = None,
) -> Dict[str, Any]:
    current_year = state_current_year(state)
    forecast_year = getattr(state, "forecast_year", None)
    atlas_year = getattr(state, "atlas_year", None)

    target_year = current_year
    if step_name.startswith("activitysim_") or step_name.startswith("beam_"):
        target_year = forecast_year if forecast_year is not None else current_year
    elif atlas_year is not None:
        target_year = atlas_year

    return {
        "stage_name": stage_name,
        "step_name": step_name,
        "year": target_year,
        "simulation_year": current_year,
        "forecast_year": forecast_year,
        "iteration": iteration_index(state),
        "atlas_year": atlas_year,
        "run_id": run_id,
        "cache_hit": cache_hit,
    }


def _resolved_output_keys(outputs: Any) -> list[str]:
    if outputs is None:
        return []
    try:
        return sorted(step_output_mapping(outputs, warn_lossy=False).keys())
    except Exception:
        return []


def _declared_required_and_optional_output_keys(
    step_name: str,
    *,
    settings: Any = None,
    state: Any = None,
) -> tuple[list[str], list[str], list[str]]:
    spec = workflow_step_spec_for_step_name(step_name)
    if spec is not None:
        contract = workflow_step_contracts_by_name(settings=settings).get(step_name, {})
        declared = sorted(dict.fromkeys(contract.get("output_keys", spec.output_keys)))
        optional = sorted(
            dict.fromkeys(
                contract.get("optional_output_keys", spec.optional_output_keys)
            )
        )
        required = list(declared)
        outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
        if outputs_class is not None:
            if not optional:
                optional = list(getattr(outputs_class, "optional_output_keys", lambda: ())())
            required = list(
                required_outputs_for_step_outputs_class(outputs_class, state=state)
            )
        return declared, sorted(dict.fromkeys(required)), optional

    outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
    declared = list(workflow_step_declared_output_keys(step_name))
    if not declared and outputs_class is not None:
        declared = list(declared_outputs_for_step_outputs_class(outputs_class))
    required: list[str] = []
    if outputs_class is not None:
        required = list(required_outputs_for_step_outputs_class(outputs_class, state=state))
    optional: list[str] = []
    if outputs_class is not None:
        optional = list(getattr(outputs_class, "optional_output_keys", lambda: ())())
    return sorted(dict.fromkeys(declared)), sorted(dict.fromkeys(required)), sorted(
        dict.fromkeys(optional)
    )


def _emit_output_hydration_audit(
    *,
    workspace: Any,
    settings: Any,
    stage_name: str,
    step_name: str,
    state: Any,
    resolution_mode: str,
    outputs: Any,
    run_id: Optional[str],
    cache_hit: Optional[bool],
    recovery_meta: Optional[Mapping[str, Any]] = None,
) -> None:
    if outputs is None:
        return
    (
        declared_outputs,
        required_outputs,
        optional_outputs,
    ) = _declared_required_and_optional_output_keys(
        step_name,
        settings=settings,
        state=state,
    )
    resolved_outputs = _resolved_output_keys(outputs)
    missing_required_outputs = [
        key for key in required_outputs if key not in resolved_outputs
    ]
    missing_declared_outputs = [
        key for key in declared_outputs if key not in resolved_outputs
    ]
    missing_optional_outputs = [
        key for key in optional_outputs if key not in resolved_outputs
    ]
    recovery_meta = recovery_meta or {}
    emit_consist_audit_event(
        workspace=workspace,
        event_type="output_hydration_check",
        **_step_scope_fields(
            stage_name=stage_name,
            step_name=step_name,
            state=state,
            run_id=run_id,
            cache_hit=cache_hit,
        ),
        resolution_mode=resolution_mode,
        declared_outputs=declared_outputs,
        required_outputs=required_outputs,
        optional_outputs=optional_outputs,
        resolved_outputs=resolved_outputs,
        missing_required_outputs=missing_required_outputs,
        missing_declared_outputs=missing_declared_outputs,
        missing_optional_outputs=missing_optional_outputs,
        typed_output_rebuilt=bool(outputs is not None),
        hydration_complete=not missing_required_outputs,
        used_manifest_restore=bool(recovery_meta.get("used_manifest_restore", False)),
        used_output_replayer=bool(recovery_meta.get("used_output_replayer", False)),
        used_output_recoverer=bool(recovery_meta.get("used_output_recoverer", False)),
        used_tracker_output_lookup=bool(
            recovery_meta.get("used_tracker_output_lookup", False)
        ),
        used_compatibility_fallback=bool(
            recovery_meta.get("used_compatibility_fallback", False)
        ),
        overwrite_rerun=bool(recovery_meta.get("overwrite_rerun", False)),
    )


def _emit_step_resolution_audit(
    *,
    workspace: Any,
    stage_name: str,
    step_name: str,
    state: Any,
    resolution_mode: str,
    run_id: Optional[str],
    cache_hit: Optional[bool],
    recovery_meta: Optional[Mapping[str, Any]] = None,
) -> None:
    recovery_meta = recovery_meta or {}
    emit_consist_audit_event(
        workspace=workspace,
        event_type="step_resolution",
        **_step_scope_fields(
            stage_name=stage_name,
            step_name=step_name,
            state=state,
            run_id=run_id,
            cache_hit=cache_hit,
        ),
        resolution_mode=resolution_mode,
        used_manifest_restore=bool(recovery_meta.get("used_manifest_restore", False)),
        used_output_replayer=bool(recovery_meta.get("used_output_replayer", False)),
        used_output_recoverer=bool(recovery_meta.get("used_output_recoverer", False)),
        used_tracker_output_lookup=bool(
            recovery_meta.get("used_tracker_output_lookup", False)
        ),
        used_compatibility_fallback=bool(
            recovery_meta.get("used_compatibility_fallback", False)
        ),
        overwrite_rerun=bool(recovery_meta.get("overwrite_rerun", False)),
        initial_cache_hit=bool(recovery_meta.get("initial_cache_hit", False)),
        recovery_attempts=int(recovery_meta.get("recovery_attempts", 0) or 0),
        **cache_miss_audit_fields(recovery_meta.get("cache_miss_explanation")),
    )


def _build_step_run_kwargs(
    *,
    step: StepRef,
    settings: Any,
    state: Any,
    workspace: Any,
    runtime_kwargs: Mapping[str, Any],
    stage_name: str,
    default_iteration: int,
) -> Dict[str, Any]:
    if not hasattr(step.step_func, "__consist_step__"):
        raise TypeError(
            f"Step '{step.name}' must be decorated with @define_step metadata."
        )

    run_kwargs: Dict[str, Any] = {"fn": step.step_func}
    step_meta = getattr(step.step_func, "__consist_step__", None)
    resolved_year = step.year
    if resolved_year is None:
        resolved_year = getattr(state, "year", None)
    if resolved_year is not None:
        run_kwargs["year"] = resolved_year

    resolved_iteration = step.iteration
    if resolved_iteration is None:
        resolved_iteration = default_iteration
    if resolved_iteration is not None:
        run_kwargs["iteration"] = resolved_iteration

    resolved_phase = step.phase or _infer_phase(step.name)
    if resolved_phase:
        run_kwargs["phase"] = resolved_phase
    resolved_stage = step.stage or stage_name
    if resolved_stage:
        run_kwargs["stage"] = resolved_stage

    if step.inputs is not None:
        if step.binding is not None:
            raise ValueError(
                f"Step '{step.name}' cannot set both StepRef.binding and StepRef.inputs."
            )
        run_kwargs["inputs"] = step.inputs
    if step.input_keys is not None:
        if step.binding is not None:
            raise ValueError(
                f"Step '{step.name}' cannot set both StepRef.binding and StepRef.input_keys."
            )
        run_kwargs["input_keys"] = step.input_keys
    resolved_binding: Optional[BindingResult] = None
    if step.binding is not None:
        binding = step.binding
        if not isinstance(binding, BindingResult):
            to_binding_result = getattr(binding, "to_binding_result", None)
            if callable(to_binding_result):
                binding = to_binding_result()
        if not isinstance(binding, BindingResult):
            raise TypeError(
                f"Step '{step.name}' binding must be a consist.BindingResult or a plan with to_binding_result()."
            )
        resolved_binding = binding
        run_kwargs["binding"] = binding
    resolved_output_paths = _resolved_step_output_paths(
        step,
        settings=settings,
        state=state,
        workspace=workspace,
    )
    if resolved_output_paths is not None:
        run_kwargs["output_paths"] = dict(resolved_output_paths)
    resolved_load_inputs = step.load_inputs
    if resolved_load_inputs is None and step.binding is None:
        resolved_load_inputs = _resolve_step_metadata_value(
            getattr(step_meta, "load_inputs", None),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
    step_meta_extra = getattr(step_meta, "extra", None)
    if not isinstance(step_meta_extra, Mapping):
        step_meta_extra = {}
    resolved_input_binding = step.input_binding
    if resolved_input_binding is None:
        resolved_input_binding = _resolve_step_metadata_value(
            getattr(step_meta, "input_binding", None),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
    resolved_input_paths = step.input_paths
    if resolved_input_paths is None:
        resolved_input_paths = _resolve_step_metadata_value(
            step_meta_extra.get("input_paths"),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
    if resolved_input_paths is not None and not isinstance(
        resolved_input_paths, Mapping
    ):
        raise TypeError(
            f"Step '{step.name}' input_paths must resolve to a mapping or None."
        )
    resolved_input_materialization = step.input_materialization
    if resolved_input_materialization is None:
        resolved_input_materialization = _resolve_step_metadata_value(
            step_meta_extra.get("input_materialization"),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
    # Canonicalize legacy load_inputs into the modern input_binding contract so
    # Consist sees a single binding mode regardless of whether the step still
    # publishes older metadata.
    if resolved_input_binding is None and resolved_load_inputs is True:
        resolved_input_binding = "loaded"
        resolved_load_inputs = None
    elif resolved_input_binding is None and resolved_load_inputs is False:
        resolved_input_binding = "none"
        resolved_load_inputs = None
    # Consist rejects simultaneous explicit input_binding and legacy
    # load_inputs hints.
    if resolved_input_binding is not None:
        resolved_load_inputs = None
    if (
        resolved_input_paths is not None
        and resolved_input_materialization == "requested"
    ):
        resolved_input_keys = _resolved_input_keys_for_step(
            step=step,
            binding=resolved_binding,
        )
        explicit_input_keys = _explicit_input_keys_for_step(
            step=step,
            binding=resolved_binding,
        )
        if resolved_input_keys is None:
            # Preserve declared requested destinations for metadata-only steps
            # that do not provide an explicit resolved input mapping. When a
            # binding is present but does not resolve concrete inputs, stage
            # nothing rather than guessing which optional keys should exist.
            if resolved_binding is not None or step.inputs is not None:
                resolved_input_paths = {}
        else:
            requested_input_paths = {
                str(key): value
                for key, value in resolved_input_paths.items()
                if str(key) in resolved_input_keys
            }
            skipped_input_keys = sorted(
                str(key)
                for key in resolved_input_paths.keys()
                if str(key) not in requested_input_paths
            )
            if skipped_input_keys:
                logger.debug(
                    "Step '%s' skipped requested input staging for unresolved keys: %s",
                    step.name,
                    skipped_input_keys,
                )
            resolved_input_paths = requested_input_paths
        if resolved_input_paths:
            explicit_requested_keys = sorted(
                key for key in resolved_input_paths.keys() if str(key) in explicit_input_keys
            )
            if explicit_requested_keys:
                logger.debug(
                    "Step '%s' skipped requested input staging for explicit input values: %s",
                    step.name,
                    explicit_requested_keys,
                )
                resolved_input_paths = {
                    key: value
                    for key, value in resolved_input_paths.items()
                    if str(key) not in explicit_input_keys
                }
    run_kwargs["execution_options"] = ExecutionOptions(
        runtime_kwargs=runtime_kwargs,
        load_inputs=resolved_load_inputs,
        input_binding=resolved_input_binding,
        input_paths=(
            dict(resolved_input_paths)
            if resolved_input_paths is not None
            else None
        ),
        input_materialization=resolved_input_materialization,
    )

    run_cfg = getattr(settings, "run", None)
    code_identity = getattr(run_cfg, "consist_code_identity", None)
    resolved_cache_hydration = step.cache_hydration
    if resolved_cache_hydration is None:
        resolved_cache_hydration = _resolve_step_metadata_value(
            getattr(step_meta, "cache_hydration", None),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
    resolved_cache_mode = step.cache_mode
    if resolved_cache_mode is None:
        resolved_cache_mode = _resolve_step_metadata_value(
            getattr(step_meta, "cache_mode", None),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
    if (
        resolved_cache_hydration is not None
        or resolved_cache_mode is not None
        or code_identity is not None
    ):
        run_kwargs["cache_options"] = CacheOptions(
            cache_hydration=resolved_cache_hydration,
            cache_mode=resolved_cache_mode,
            code_identity=code_identity,
        )

    def _normalize_output_keys(values: Any) -> Optional[list[str]]:
        if not isinstance(values, Sequence) or isinstance(values, str):
            return None
        return [output for output in values if isinstance(output, str)]

    outputs_class = STEP_OUTPUTS_CLASSES.get(step.name)
    required_outputs: list[str] = []
    if outputs_class is not None:
        required_outputs = list(
            required_outputs_for_step_outputs_class(outputs_class, state=state)
        )

    resolved_required_outputs: Optional[Sequence[str]] = None
    optional_declared_outputs: list[str] = []
    if outputs_class is not None:
        # Tracked steps use StepOutputs required_outputs as the strict runtime
        # output contract. declared_outputs remains available for schema/catalog
        # publication without forcing every optional artifact to materialize.
        resolved_required_outputs = required_outputs or None
        optional_declared_outputs = list(
            getattr(outputs_class, "optional_output_keys", lambda: ())()
        )
    elif step_meta is not None:
        # Metadata-only steps remain supported for non-catalog call sites.
        resolved_required_outputs = _normalize_output_keys(
            getattr(step_meta, "outputs", None)
        )

    if resolved_required_outputs:
        run_kwargs["outputs"] = list(resolved_required_outputs)

    # Apply strict defaults for declared-output steps unless explicitly overridden.
    output_missing = step.output_missing
    output_mismatch = step.output_mismatch
    if output_missing is None and resolved_required_outputs:
        # Consist applies missing-output policy to every declared output_path,
        # including optional tracked outputs. Let tracked-step hydration enforce
        # required outputs after execution while tolerating absent optional paths.
        output_missing = "ignore" if optional_declared_outputs else "error"
    if output_mismatch is None and resolved_required_outputs:
        output_mismatch = "error"
    if output_missing is not None or output_mismatch is not None:
        run_kwargs["output_policy"] = OutputPolicyOptions(
            output_missing=output_missing,
            output_mismatch=output_mismatch,
        )

    if step.model is not None:
        run_kwargs["model"] = step.model
    return run_kwargs


@dataclass
class WorkflowStage:
    """
    Orchestrates a sequence of related steps.
    """

    name: str
    stage_type: Any
    steps: Sequence[StepRef]

    def run(
        self,
        *,
        scenario: Any,
        state: Any,
        settings: Any,
        workspace: Any,
        coupler: CouplerProtocol,
        outputs_holder: StepOutputsHolder,
        name_suffix: str,
        iteration: int = 0,
        runtime_kwargs_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute all steps in sequence for this stage.
        """
        run_workflow(
            stage_name=self.name,
            steps=self.steps,
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            name_suffix=name_suffix,
            iteration=iteration,
            runtime_kwargs_extra=runtime_kwargs_extra,
        )


@dataclass(frozen=True)
class ManifestConfig:
    """
    Configuration for manifest-based step checkpointing.
    """

    path: Path


@dataclass(frozen=True)
class StageRunner:
    """
    Thin helper for running multiple step slices inside one stage context.

    This keeps stage modules focused on sequencing and binding decisions while
    preserving their existing `run_workflow(...)` patch points.
    """

    stage_name: str
    scenario: Any
    state: Any
    settings: Any
    workspace: Any
    coupler: CouplerProtocol
    outputs_holder: StepOutputsHolder
    name_suffix: str
    iteration: int = 0
    manifest_config: Optional[ManifestConfig] = None
    runtime_kwargs_extra: Optional[Mapping[str, Any]] = None
    run_workflow_fn: Optional[Callable[..., None]] = None

    def run(
        self,
        *,
        steps: Sequence[StepRef],
        stage_name: Optional[str] = None,
        runtime_kwargs_extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged_runtime_kwargs: Dict[str, Any] = dict(self.runtime_kwargs_extra or {})
        if runtime_kwargs_extra:
            merged_runtime_kwargs.update(runtime_kwargs_extra)
        runner = self.run_workflow_fn or run_workflow
        resolved_stage_name = stage_name or self.stage_name
        logger.info(
            "[stage:%s] starting name_suffix=%s iteration=%s run_id=%s",
            resolved_stage_name,
            self.name_suffix,
            self.iteration,
            cr.current_run_id(),
        )
        runner(
            stage_name=resolved_stage_name,
            steps=steps,
            scenario=self.scenario,
            state=self.state,
            settings=self.settings,
            workspace=self.workspace,
            coupler=self.coupler,
            outputs_holder=self.outputs_holder,
            name_suffix=self.name_suffix,
            iteration=self.iteration,
            runtime_kwargs_extra=merged_runtime_kwargs or None,
            manifest_config=self.manifest_config,
        )

    def run_step(
        self,
        *,
        step: StepRef,
        stage_name: Optional[str] = None,
        runtime_kwargs_extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Execute one step without an intermediate single-item steps list."""
        self.run(
            steps=[step],
            stage_name=stage_name,
            runtime_kwargs_extra=runtime_kwargs_extra,
        )


def _resolved_step_inputs(step: StepRef) -> Optional[Mapping[str, Any]]:
    if step.binding is not None:
        return getattr(step.binding, "inputs", None)
    return step.inputs


def _resolved_step_output_paths(
    step: StepRef,
    *,
    settings: Any,
    state: Any,
    workspace: Any,
) -> Optional[Mapping[str, Any]]:
    if step.output_paths is not None:
        return step.output_paths
    if step.output_paths_provider is None:
        step_meta = getattr(step.step_func, "__consist_step__", None)
        metadata_output_paths = _resolve_step_metadata_value(
            getattr(step_meta, "output_paths", None),
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs={
                "settings": settings,
                "state": state,
                "workspace": workspace,
            },
        )
        if metadata_output_paths is None:
            return None
        output_paths = metadata_output_paths
    else:
        output_paths = step.output_paths_provider(
            settings=settings,
            state=state,
            workspace=workspace,
        )
    if output_paths is None:
        return None
    if not isinstance(output_paths, Mapping):
        raise TypeError(
            f"Step '{step.name}' output-path provider must return a mapping or None."
        )
    return output_paths


def _build_step_metadata_context(
    *,
    step: StepRef,
    settings: Any,
    state: Any,
    workspace: Any,
    runtime_kwargs: Mapping[str, Any],
) -> StepContext:
    step_meta = getattr(step.step_func, "__consist_step__", None)
    step_model = getattr(step_meta, "model", None) or step.name
    signature = inspect.signature(StepContext)
    kwargs: Dict[str, Any] = {
        "func_name": getattr(step.step_func, "__name__", step.name),
        "model": step_model,
        "runtime_kwargs": dict(runtime_kwargs),
    }
    if "settings" in signature.parameters:
        kwargs["settings"] = settings
    if "runtime_settings" in signature.parameters:
        kwargs["runtime_settings"] = settings
    if "runtime_workspace" in signature.parameters:
        kwargs["runtime_workspace"] = workspace
    if "consist_settings" in signature.parameters:
        kwargs["consist_settings"] = SimpleNamespace()
    if "consist_workspace" in signature.parameters:
        kwargs["consist_workspace"] = Path(getattr(workspace, "full_path", "."))
    return StepContext(**kwargs)


def _resolve_step_metadata_value(
    value: Any,
    *,
    step: StepRef,
    settings: Any,
    state: Any,
    workspace: Any,
    runtime_kwargs: Mapping[str, Any],
) -> Any:
    if value is None or not callable(value):
        return value

    try:
        parameters = inspect.signature(value).parameters
    except (TypeError, ValueError):
        ctx = _build_step_metadata_context(
            step=step,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
        )
        return value(ctx)

    if not parameters:
        return value()

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return value(settings=settings, state=state, workspace=workspace)

    names = list(parameters)
    if len(parameters) == 1:
        name = names[0]
        if name in {"ctx", "context", "step_context"}:
            ctx = _build_step_metadata_context(
                step=step,
                settings=settings,
                state=state,
                workspace=workspace,
                runtime_kwargs=runtime_kwargs,
            )
            return value(ctx)

    kwargs: Dict[str, Any] = {}
    if "settings" in parameters:
        kwargs["settings"] = settings
    if "state" in parameters:
        kwargs["state"] = state
    if "workspace" in parameters:
        kwargs["workspace"] = workspace
    if kwargs:
        return value(**kwargs)

    ctx = _build_step_metadata_context(
        step=step,
        settings=settings,
        state=state,
        workspace=workspace,
        runtime_kwargs=runtime_kwargs,
    )
    return value(ctx)


def _publish_recovered_outputs(
    *,
    step: StepRef,
    outputs: Any,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    cached_outputs: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    tracker_outputs: Optional[Mapping[str, Any]] = None
    if run_id:
        try:
            tracker_outputs = load_tracker_run_outputs(
                run_id,
                logger=logger,
                log_context="workflow recovered output publication",
            )
        except RuntimeError:
            logger.debug(
                "Skipping tracker output lookup while republishing recovered outputs for %s; "
                "no active tracker is available.",
                step.name,
                exc_info=True,
            )
    cached_mapping = merge_canonical_output_mappings(
        cached_outputs,
        tracker_outputs,
    )
    replayer = step.output_replayer
    if replayer is not None:
        with record_published_coupler_keys() as replayed_keys:
            if cr.current_run() is not None:
                replayer(outputs, settings, state, workspace, outputs_holder)
            else:
                previous_enabled = getattr(cr, "_enabled_override", None)
                cr.set_enabled(False)
                try:
                    replayer(outputs, settings, state, workspace, outputs_holder)
                finally:
                    cr.set_enabled(previous_enabled)
        if cached_mapping:
            filtered_cached_mapping = {
                key: value
                for key, value in cached_mapping.items()
                if key in replayed_keys
            }
        else:
            filtered_cached_mapping = {}
        if filtered_cached_mapping:
            _update_coupler_from_mapping(
                filtered_cached_mapping,
                coupler=coupler,
                workspace=workspace,
            )
            logger.debug(
                "Republished %d cached artifact outputs into the coupler after replay for %s.",
                len(filtered_cached_mapping),
                step.name,
            )
            return "output_replayer_with_cached_artifacts"
        return "output_replayer"
    _update_coupler_from_outputs(outputs, coupler=coupler, workspace=workspace)
    return "typed_outputs"


def _merge_output_recovery_meta(outputs: Any, audit_meta: Dict[str, Any]) -> None:
    if bool(getattr(outputs, "_compatibility_fallback_used", False)):
        audit_meta["used_compatibility_fallback"] = True


def _finalize_recovered_step_outputs(
    *,
    step: StepRef,
    step_name: str,
    outputs: Any,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    publish_outputs: bool,
    cached_outputs: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    audit_meta: Optional[Dict[str, Any]] = None,
) -> Any:
    if audit_meta is None:
        audit_meta = {}
    validate = getattr(outputs, "validate", None)
    if callable(validate):
        validate()
    outputs_holder.set_attribute(step_name, outputs)
    if publish_outputs:
        publication_mode = _publish_recovered_outputs(
            step=step,
            outputs=outputs,
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            cached_outputs=cached_outputs,
            run_id=run_id,
        )
        audit_meta["output_publication_mode"] = publication_mode
    return outputs


def _recover_step_outputs(
    *,
    step: StepRef,
    step_name: str,
    outputs_holder: StepOutputsHolder,
    settings: Any,
    state: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    step_inputs: Optional[Mapping[str, Any]] = None,
    cached_outputs: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    publish_outputs: bool = True,
    audit_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    if audit_meta is None:
        audit_meta = {}
    recoverer = step.output_recoverer
    if recoverer is None:
        return None
    outputs = recoverer(
        settings=settings,
        state=state,
        workspace=workspace,
        coupler=coupler,
        outputs_holder=outputs_holder,
        step_inputs=step_inputs,
        cached_outputs=cached_outputs,
        run_id=run_id,
    )
    if outputs is not None:
        audit_meta["used_output_recoverer"] = True
        audit_meta["used_tracker_output_lookup"] = True
        audit_meta["used_output_replayer"] = bool(
            publish_outputs and step.output_replayer is not None
        )
        outputs = _finalize_recovered_step_outputs(
            step=step,
            step_name=step_name,
            outputs=outputs,
            settings=settings,
            state=state,
            workspace=workspace,
            coupler=coupler,
            outputs_holder=outputs_holder,
            publish_outputs=publish_outputs,
            cached_outputs=cached_outputs,
            run_id=run_id,
            audit_meta=audit_meta,
        )
        _merge_output_recovery_meta(outputs, audit_meta)
        return outputs
    return None


def _recovery_run_id_for_step_result(step_result: Any) -> Optional[str]:
    run = getattr(step_result, "run", None)
    run_id = getattr(run, "id", None)
    if not getattr(step_result, "cache_hit", False):
        return run_id
    run_meta = getattr(run, "meta", None)
    if isinstance(run_meta, Mapping):
        cache_source = run_meta.get("cache_source")
        if cache_source:
            return str(cache_source)
    return run_id


def run_manifested_steps(
    *,
    stage_name: str,
    steps: Sequence[StepRef],
    outputs_holder: StepOutputsHolder,
    manifest_config: ManifestConfig,
    scenario: Any,
    state: Any,
    settings: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    name_suffix: str,
    iteration: int = 0,
    runtime_kwargs_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute steps with manifest checkpointing and stale detection.
    """
    validate_workflow_step_contracts(step_refs=steps)

    manifest = load_step_manifest(manifest_config.path) or {}
    stale_steps = _detect_stale_steps(
        manifest,
        outputs_holder,
        workspace,
        settings=settings,
        state=state,
    )
    if stale_steps:
        stale_steps = _expand_stale_manifest_steps(
            manifest_step_names=list(manifest.keys()),
            steps=steps,
            stale_steps=stale_steps,
        )
        for step_name in stale_steps:
            manifest.pop(step_name, None)
        save_step_manifest(manifest, manifest_config.path)

    runtime_kwargs = common_runtime_kwargs(
        settings=settings,
        state=state,
        workspace=workspace,
        **(runtime_kwargs_extra or {}),
    )

    for raw_step in steps:
        spec = raw_step
        model_name, resolved_year, resolved_iteration = _resolved_step_epoch_identity(
            step=spec,
            state=state,
            default_iteration=iteration,
        )
        expects_outputs = STEP_OUTPUTS_CLASSES.get(spec.name) is not None
        if spec.name in manifest:
            if expects_outputs:
                logger.info("[%s] %s already completed (skipping)", stage_name, spec.name)
                outputs = outputs_holder.get_attribute(spec.name)
                recovery_meta = {
                    "used_manifest_restore": True,
                    "used_output_replayer": bool(spec.output_replayer is not None),
                }
                if outputs is None:
                    outputs = _restore_outputs_from_manifest(
                        spec.name,
                        manifest,
                        workspace,
                        settings=settings,
                        state=state,
                    )
                    if outputs is not None:
                        outputs_holder.set_attribute(spec.name, outputs)
                if outputs is not None:
                    _publish_recovered_outputs(
                        step=spec,
                        outputs=outputs,
                        settings=settings,
                        state=state,
                        workspace=workspace,
                        coupler=coupler,
                        outputs_holder=outputs_holder,
                        run_id=manifest.get(spec.name, {}).get("run_id"),
                    )
                    _merge_output_recovery_meta(outputs, recovery_meta)
                    _emit_step_resolution_audit(
                        workspace=workspace,
                        stage_name=stage_name,
                        step_name=spec.name,
                        state=state,
                        resolution_mode="manifest_restore",
                        run_id=manifest.get(spec.name, {}).get("run_id"),
                        cache_hit=bool(manifest.get(spec.name, {}).get("cache_hit", False)),
                        recovery_meta=recovery_meta,
                    )
                    _emit_output_hydration_audit(
                        workspace=workspace,
                        settings=settings,
                        stage_name=stage_name,
                        step_name=spec.name,
                        state=state,
                        resolution_mode="manifest_restore",
                        outputs=outputs,
                        run_id=manifest.get(spec.name, {}).get("run_id"),
                        cache_hit=bool(manifest.get(spec.name, {}).get("cache_hit", False)),
                        recovery_meta=recovery_meta,
                    )
                remember_restored_run_id = getattr(
                    scenario, "remember_restored_run_id", None
                )
                if callable(remember_restored_run_id):
                    remember_restored_run_id(
                        model_name=model_name,
                        year=resolved_year,
                        iteration=resolved_iteration,
                        run_id=manifest.get(spec.name, {}).get("run_id"),
                    )
                continue
            logger.info(
                "[%s] %s has a manifest run_id but no declared outputs; rerunning "
                "instead of trusting manifest-only completion",
                stage_name,
                spec.name,
            )

        validate_step_ready(spec.name, outputs_holder)
        run_kwargs = _build_step_run_kwargs(
            step=spec,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
            stage_name=stage_name,
            default_iteration=iteration,
        )

        def _run_step(cache_options: Optional[CacheOptions]) -> Any:
            step_kwargs = dict(run_kwargs)
            if cache_options is not None:
                step_kwargs["cache_options"] = cache_options
            _warn_for_undeclared_step_inputs(
                step_name=spec.name,
                input_keys=step_kwargs.get("input_keys"),
                inputs=step_kwargs.get("inputs"),
                binding=step_kwargs.get("binding"),
                settings=settings,
                state=state,
                workspace=workspace,
            )
            return scenario.run(**step_kwargs)

        recovery_meta: Dict[str, Any] = {}

        def _recover_outputs(step_result: Any) -> Optional[Any]:
            return _recover_step_outputs(
                step=spec,
                step_name=spec.name,
                outputs_holder=outputs_holder,
                settings=settings,
                state=state,
                workspace=workspace,
                coupler=coupler,
                step_inputs=_resolved_step_inputs(spec),
                cached_outputs=getattr(step_result, "outputs", None),
                run_id=_recovery_run_id_for_step_result(step_result),
                publish_outputs=True,
                audit_meta=recovery_meta,
            )

        if expects_outputs:
            result, outputs, cache_meta = run_with_cache_recovery(
                stage_name=stage_name,
                step_name=spec.name,
                run_step=_run_step,
                read_outputs=lambda: outputs_holder.get_attribute(spec.name),
                recover_outputs=_recover_outputs,
            )
            recovery_meta.update(cache_meta)
            if outputs is None:
                raise RuntimeError(f"{spec.name} did not populate outputs_holder")
            serialized_outputs = serialize_step_outputs(outputs)
            if recovery_meta.get("overwrite_rerun"):
                resolution_mode = "overwrite_rerun_after_cache_hit"
            elif recovery_meta.get("used_output_recoverer"):
                resolution_mode = "cache_hit_recoverer"
            elif recovery_meta.get("initial_cache_hit"):
                resolution_mode = "cache_hit_direct"
            else:
                resolution_mode = "executed"
            _emit_step_resolution_audit(
                workspace=workspace,
                stage_name=stage_name,
                step_name=spec.name,
                state=state,
                resolution_mode=resolution_mode,
                run_id=getattr(getattr(result, "run", None), "id", None),
                cache_hit=bool(getattr(result, "cache_hit", False)),
                recovery_meta=recovery_meta,
            )
            _emit_output_hydration_audit(
                workspace=workspace,
                settings=settings,
                stage_name=stage_name,
                step_name=spec.name,
                state=state,
                resolution_mode=resolution_mode,
                outputs=outputs,
                run_id=getattr(getattr(result, "run", None), "id", None),
                cache_hit=bool(getattr(result, "cache_hit", False)),
                recovery_meta=recovery_meta,
            )
        else:
            # Steps without declared outputs can still be checkpointed. We record the
            # run id so restart reconstruction can re-materialize completed runs, but
            # do not force an outputs_holder hydration loop.
            result = _run_step(None)
            serialized_outputs = {}
            if not getattr(result, "cache_hit", False):
                cache_miss_explanation = log_cache_miss_explanation(
                    logger=logger,
                    result=result,
                    info_message="[%s] Cache miss for %s. reason=%s candidate_run_id=%s",
                    info_args=(stage_name, spec.name),
                    debug_message="[%s] Cache miss details for %s: %s",
                    debug_args=(stage_name, spec.name),
                )
                if cache_miss_explanation is not None:
                    recovery_meta["cache_miss_explanation"] = cache_miss_explanation
            _emit_step_resolution_audit(
                workspace=workspace,
                stage_name=stage_name,
                step_name=spec.name,
                state=state,
                resolution_mode="executed",
                run_id=getattr(getattr(result, "run", None), "id", None),
                cache_hit=bool(getattr(result, "cache_hit", False)),
                recovery_meta=recovery_meta,
            )
        manifest[spec.name] = {
            "completed_at": datetime.now().isoformat(),
            "cache_hit": bool(getattr(result, "cache_hit", False)),
            "run_id": getattr(getattr(result, "run", None), "id", None),
            "outputs": serialized_outputs,
        }
        save_step_manifest(manifest, manifest_config.path)


def run_workflow(
    *,
    stage_name: str,
    steps: Sequence[StepRef],
    scenario: Any,
    state: Any,
    settings: Any,
    workspace: Any,
    coupler: CouplerProtocol,
    outputs_holder: StepOutputsHolder,
    name_suffix: str,
    iteration: int = 0,
    runtime_kwargs_extra: Optional[Dict[str, Any]] = None,
    manifest_config: Optional[ManifestConfig] = None,
) -> None:
    """
    Execute a sequence of workflow steps using native step metadata.
    """
    validate_workflow_step_contracts(step_refs=steps)

    if manifest_config is not None:
        run_manifested_steps(
            stage_name=stage_name,
            steps=steps,
            outputs_holder=outputs_holder,
            manifest_config=manifest_config,
            scenario=scenario,
            state=state,
            settings=settings,
            workspace=workspace,
            coupler=coupler,
            name_suffix=name_suffix,
            iteration=iteration,
            runtime_kwargs_extra=runtime_kwargs_extra,
        )
        return

    runtime_kwargs = common_runtime_kwargs(
        settings=settings,
        state=state,
        workspace=workspace,
        **(runtime_kwargs_extra or {}),
    )
    for raw_step in steps:
        spec = raw_step
        validate_step_ready(spec.name, outputs_holder)
        run_kwargs = _build_step_run_kwargs(
            step=spec,
            settings=settings,
            state=state,
            workspace=workspace,
            runtime_kwargs=runtime_kwargs,
            stage_name=stage_name,
            default_iteration=iteration,
        )
        coupler_keys = None
        coupler_keys_fn = getattr(coupler, "keys", None)
        if callable(coupler_keys_fn):
            try:
                coupler_keys = list(coupler_keys_fn())
            except TypeError:
                coupler_keys = None
        logger.debug(
            "[%s] Running %s: inputs=%s input_keys=%s outputs=%s",
            stage_name,
            spec.name,
            bool(spec.inputs),
            spec.input_keys,
            spec.output_paths,
        )
        if coupler_keys is not None:
            logger.debug(
                "[%s] Coupler keys before %s: %s",
                stage_name,
                spec.name,
                coupler_keys,
            )

        def _run_step(cache_options: Optional[CacheOptions]) -> Any:
            step_kwargs = dict(run_kwargs)
            if cache_options is not None:
                step_kwargs["cache_options"] = cache_options
            _warn_for_undeclared_step_inputs(
                step_name=spec.name,
                input_keys=step_kwargs.get("input_keys"),
                inputs=step_kwargs.get("inputs"),
                binding=step_kwargs.get("binding"),
                settings=settings,
                state=state,
                workspace=workspace,
            )
            return scenario.run(**step_kwargs)

        recovery_meta: Dict[str, Any] = {}

        def _recover_outputs(step_result: Any) -> Optional[Any]:
            return _recover_step_outputs(
                step=spec,
                step_name=spec.name,
                outputs_holder=outputs_holder,
                settings=settings,
                state=state,
                workspace=workspace,
                coupler=coupler,
                step_inputs=_resolved_step_inputs(spec),
                cached_outputs=getattr(step_result, "outputs", None),
                run_id=_recovery_run_id_for_step_result(step_result),
                publish_outputs=True,
                audit_meta=recovery_meta,
            )

        result, outputs, cache_meta = run_with_cache_recovery(
            stage_name=stage_name,
            step_name=spec.name,
            run_step=_run_step,
            read_outputs=lambda: outputs_holder.get_attribute(spec.name),
            recover_outputs=_recover_outputs,
        )
        recovery_meta.update(cache_meta)
        if recovery_meta.get("overwrite_rerun"):
            resolution_mode = "overwrite_rerun_after_cache_hit"
        elif recovery_meta.get("used_output_recoverer"):
            resolution_mode = "cache_hit_recoverer"
        elif recovery_meta.get("initial_cache_hit"):
            resolution_mode = "cache_hit_direct"
        else:
            resolution_mode = "executed"
        _emit_step_resolution_audit(
            workspace=workspace,
            stage_name=stage_name,
            step_name=spec.name,
            state=state,
            resolution_mode=resolution_mode,
            run_id=getattr(getattr(result, "run", None), "id", None),
            cache_hit=bool(getattr(result, "cache_hit", False)),
            recovery_meta=recovery_meta,
        )
        _emit_output_hydration_audit(
            workspace=workspace,
            settings=settings,
            stage_name=stage_name,
            step_name=spec.name,
            state=state,
            resolution_mode=resolution_mode,
            outputs=outputs,
            run_id=getattr(getattr(result, "run", None), "id", None),
            cache_hit=bool(getattr(result, "cache_hit", False)),
            recovery_meta=recovery_meta,
        )

        if coupler_keys is not None:
            try:
                current_coupler_keys_fn = getattr(coupler, "keys", None)
                if not callable(current_coupler_keys_fn):
                    raise TypeError
                logger.debug(
                    "[%s] Coupler keys after %s: %s",
                    stage_name,
                    spec.name,
                    list(current_coupler_keys_fn()),
                )
            except TypeError:
                logger.debug(
                    "[%s] Coupler keys after %s: <unavailable>",
                    stage_name,
                    spec.name,
                )


def _detect_stale_steps(
    manifest: Dict[str, Any],
    outputs_holder: StepOutputsHolder,
    workspace: Any,
    *,
    settings: Any = None,
    state: Any = None,
) -> Set[str]:
    """
    Check which manifest entries have stale or missing outputs.
    """
    stale: Set[str] = set()
    for step_name, step_info in manifest.items():
        outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
        if outputs_class is None:
            continue
        outputs_data = step_info.get("outputs", {})
        try:
            outputs = deserialize_step_outputs(outputs_class, outputs_data)
            _coerce_outputs_path_fields(outputs, outputs_class)
            _remap_outputs_workspace_paths(
                outputs,
                outputs_class,
                workspace=workspace,
                step_name=step_name,
            )
            validate = getattr(outputs, "validate", None)
            if callable(validate):
                validate(
                    context=ValidationContext(
                        settings=settings,
                        state=state,
                        workspace=workspace,
                        step_name=step_name,
                    )
                )
            outputs_holder.set_attribute(step_name, outputs)
        except (AssertionError, FileNotFoundError) as exc:
            logger.warning(
                "Manifest outputs for %s are stale; will re-run (%s)",
                step_name,
                exc,
            )
            stale.add(step_name)
    return stale


def _expand_stale_manifest_steps(
    *,
    manifest_step_names: Sequence[str],
    steps: Sequence[StepRef],
    stale_steps: Set[str],
) -> Set[str]:
    """
    Invalidate later manifest entries after any stale upstream step.

    ``run_manifested_steps`` may reuse one manifest across multiple ordered
    invocations in the same stage. If an upstream manifest entry is stale,
    keeping later entries would allow a mixture of fresh upstream outputs with
    stale downstream artifacts from the old manifest. To keep restore behavior
    correct, invalidate the stale step and every later entry in the manifest
    sequence, not just the current invocation slice.
    """
    if not stale_steps:
        return set()

    scope = set(manifest_step_names)
    scope.update(step.name for step in steps)

    dependents: Dict[str, Set[str]] = {}
    for step_name, spec in steps_shared.STEP_DEPENDENCIES.items():
        for upstream in spec.get("depends_on", ()):
            dependents.setdefault(str(upstream), set()).add(step_name)

    expanded: Set[str] = set()
    pending = [step_name for step_name in stale_steps if step_name in scope]
    while pending:
        step_name = pending.pop()
        if step_name in expanded:
            continue
        expanded.add(step_name)
        for dependent in sorted(dependents.get(step_name, ())):
            if dependent in scope and dependent not in expanded:
                pending.append(dependent)
    return expanded


def _update_coupler_from_outputs(
    outputs: Any,
    *,
    coupler: CouplerProtocol,
    workspace: Any,
) -> None:
    _update_coupler_from_mapping(
        step_output_mapping(outputs, warn_lossy=False),
        coupler=coupler,
        workspace=workspace,
    )


def _update_coupler_from_mapping(
    mapping: Mapping[str, Any],
    *,
    coupler: CouplerProtocol,
    workspace: Any,
) -> None:
    if not mapping:
        return
    for key, value in mapping.items():
        canonical_key = canonical_artifact_key_from_raw_key(str(key))
        resolved = resolve_artifact_from_value(
            value,
            key=canonical_key,
            workspace=workspace,
        )
        path = artifact_to_path(value, workspace)
        if path is None:
            continue
        artifact = (
            resolved
            if (hasattr(resolved, "container_uri") or hasattr(resolved, "uri"))
            else None
        )
        set_coupler_from_artifact(
            coupler=coupler,
            key=canonical_key,
            artifact=artifact,
            fallback=path,
        )


def _restore_outputs_from_manifest(
    step_name: str,
    manifest: Dict[str, Any],
    workspace: Any,
    *,
    settings: Any = None,
    state: Any = None,
) -> Optional[Any]:
    """
    Restore step outputs from a manifest entry when possible.
    """
    step_info = manifest.get(step_name, {})
    outputs_class = STEP_OUTPUTS_CLASSES.get(step_name)
    if outputs_class is None:
        return None
    outputs_data = step_info.get("outputs", {})
    try:
        outputs = deserialize_step_outputs(outputs_class, outputs_data)
        _coerce_outputs_path_fields(outputs, outputs_class)
        _remap_outputs_workspace_paths(
            outputs,
            outputs_class,
            workspace=workspace,
            step_name=step_name,
        )
        validate = getattr(outputs, "validate", None)
        if callable(validate):
            validate(
                context=ValidationContext(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                    step_name=step_name,
                )
            )
        return outputs
    except (AssertionError, FileNotFoundError):
        return None


def _coerce_outputs_path_fields(outputs: Any, outputs_class: Any) -> None:
    """
    Normalize manifest-restored path fields to ``Path`` objects.

    Some StepOutputs classes use postponed annotation evaluation, which can
    leave deserialized path fields as plain strings. Coerce by declared
    StepOutputs path metadata so replay/output loggers see consistent types.
    """
    path_fields = tuple(getattr(outputs_class, "required_path_fields", ()) or ())
    path_fields += tuple(getattr(outputs_class, "optional_path_fields", ()) or ())
    for field_name in path_fields:
        value = getattr(outputs, field_name, None)
        if isinstance(value, str):
            setattr(outputs, field_name, Path(value))

    for field_name in tuple(getattr(outputs_class, "dict_path_fields", ()) or ()):
        value = getattr(outputs, field_name, None)
        if not isinstance(value, Mapping):
            continue
        if not any(isinstance(path_value, str) for path_value in value.values()):
            continue
        setattr(
            outputs,
            field_name,
            {
                key: (Path(path_value) if isinstance(path_value, str) else path_value)
                for key, path_value in value.items()
            },
        )


def _remap_outputs_workspace_paths(
    outputs: Any,
    outputs_class: Any,
    *,
    workspace: Any,
    step_name: Optional[str] = None,
) -> None:
    """
    Remap manifest-restored workspace-local paths into the current workspace.

    Restart manifests may serialize absolute paths from an older node-local
    workspace root such as ``/local/job123/.../<run_name>/...``. On restart,
    the run name is stable but the node-local job root changes. When the same
    relative path now exists under the current workspace root, rewrite the
    deserialized path so manifest restore can succeed without rerunning the
    step.
    """

    current_root_raw = getattr(workspace, "full_path", None)
    if not current_root_raw:
        return
    current_root = Path(current_root_raw)
    current_run_dir_name = current_root.name
    if not current_run_dir_name:
        return

    def _is_within_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _remap_path(path_value: Any) -> Any:
        if not isinstance(path_value, Path):
            return path_value

        direct_existing = resolve_existing_path(
            str(path_value),
            workspace=workspace,
            materialize_from_archive=True,
        )
        if direct_existing:
            return Path(direct_existing)

        if _is_within_root(path_value, current_root):
            return path_value

        matching_indices = [
            index
            for index, part in enumerate(path_value.parts)
            if part == current_run_dir_name
        ]
        for index in reversed(matching_indices):
            suffix = path_value.parts[index + 1 :]
            candidate = current_root.joinpath(*suffix)
            remapped_existing = resolve_existing_path(
                str(candidate),
                workspace=workspace,
                materialize_from_archive=True,
            )
            if remapped_existing:
                logger.info(
                    "Manifest restore remapped %s path from old workspace root: %s -> %s",
                    step_name or outputs_class.__name__,
                    path_value,
                    remapped_existing,
                )
                return Path(remapped_existing)
        return path_value

    for field_name in tuple(getattr(outputs_class, "required_path_fields", ()) or ()):
        setattr(outputs, field_name, _remap_path(getattr(outputs, field_name, None)))

    for field_name in tuple(getattr(outputs_class, "optional_path_fields", ()) or ()):
        setattr(outputs, field_name, _remap_path(getattr(outputs, field_name, None)))

    for field_name in tuple(getattr(outputs_class, "dict_path_fields", ()) or ()):
        value = getattr(outputs, field_name, None)
        if not isinstance(value, Mapping):
            continue
        remapped = {
            key: _remap_path(path_value)
            for key, path_value in value.items()
        }
        setattr(outputs, field_name, remapped)
