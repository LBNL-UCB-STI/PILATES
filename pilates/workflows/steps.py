from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    TYPE_CHECKING,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pilates.generic.model_factory import ModelFactory
from pilates.generic.records import FileRecord, RecordStore
from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import (
    artifact_to_path,
    log_and_set_input,
    log_and_set_output,
    record_store_to_outputs,
    resolve_artifact_from_value,
    update_coupler_from_beam_outputs,
)
from pilates.workflows.step_exec import (
    forecast_land_use,
    run_postprocessor,
    run_preprocessor,
    run_runner,
    run_traffic_assignment,
    warm_start_activities,
)
from workflow_state import WorkflowState

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.generic.records import RecordStore
    from pilates.workspace import Workspace

logger = logging.getLogger(__name__)

StepOutputsT = TypeVar("StepOutputsT")


def _serialize_value(value: Any) -> Any:
    """
    Convert Path-like values into YAML-safe primitives.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    Any
        Serialized value suitable for YAML.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_value(val) for val in value]
    return value


def serialize_step_outputs(outputs: Any) -> Dict[str, Any]:
    """
    Serialize a StepOutputs dataclass to primitives for the manifest.

    Parameters
    ----------
    outputs : Any
        Step outputs dataclass instance.

    Returns
    -------
    dict
        YAML-serializable payload.
    """
    data = asdict(outputs)
    return _serialize_value(data)


def _is_optional_path_type(field_type: Any) -> bool:
    """
    Check whether a type annotation represents Optional[Path].

    Parameters
    ----------
    field_type : Any
        Type annotation to inspect.

    Returns
    -------
    bool
        True if the annotation matches Optional[Path].
    """
    origin = get_origin(field_type)
    if origin is None:
        return False
    if origin is Union:
        args = get_args(field_type)
        return len(args) == 2 and Path in args and type(None) in args
    return False


def _is_dict_path_type(field_type: Any) -> bool:
    """
    Check whether a type annotation represents Dict[str, Path].

    Parameters
    ----------
    field_type : Any
        Type annotation to inspect.

    Returns
    -------
    bool
        True if the annotation matches Dict[str, Path].
    """
    origin = get_origin(field_type)
    if origin not in (dict, Dict):
        return False
    args = get_args(field_type)
    return len(args) == 2 and args[1] is Path


def deserialize_step_outputs(
    output_class: Type[StepOutputsT],
    data: Mapping[str, Any],
) -> StepOutputsT:
    """
    Reconstruct a StepOutputs dataclass from manifest data.

    Parameters
    ----------
    output_class : type
        Dataclass type to instantiate.
    data : mapping
        Serialized manifest entry.

    Returns
    -------
    StepOutputsT
        Reconstructed StepOutputs instance.
    """
    kwargs: Dict[str, Any] = {}
    for field in fields(output_class):
        if field.name not in data:
            continue
        value = data[field.name]
        if value is None:
            kwargs[field.name] = None
            continue
        if field.type is Path or _is_optional_path_type(field.type):
            kwargs[field.name] = Path(value)
            continue
        if _is_dict_path_type(field.type):
            kwargs[field.name] = {key: Path(val) for key, val in value.items()}
            continue
        kwargs[field.name] = value
    return output_class(**kwargs)


@dataclass
class ActivitySimPreprocessOutputs:
    """
    Outputs from the ActivitySim preprocess step.

    Attributes
    ----------
    mutable_data_dir : Path
        ActivitySim mutable data directory.
    land_use_table : Path
        Land use input table path.
    households_table : Path
        Households input table path.
    persons_table : Path
        Persons input table path.
    omx_skims : Path, optional
        OMX skims input path when present.
    """

    primary_output_attr: ClassVar[str] = "mutable_data_dir"
    record_keys: ClassVar[Dict[str, str]] = {
        "land_use_table": "land_use_asim_in",
        "households_table": "households_asim_in",
        "persons_table": "persons_asim_in",
        "omx_skims": "omx_skims",
    }

    mutable_data_dir: Path
    land_use_table: Path
    households_table: Path
    persons_table: Path
    omx_skims: Optional[Path] = None

    def validate(self) -> None:
        """
        Validate that required outputs exist on disk.
        """
        assert self.mutable_data_dir.exists(), (
            f"mutable_data_dir missing: {self.mutable_data_dir}"
        )
        assert self.land_use_table.exists(), (
            f"land_use_table missing: {self.land_use_table}"
        )
        assert self.households_table.exists(), (
            f"households_table missing: {self.households_table}"
        )
        assert self.persons_table.exists(), (
            f"persons_table missing: {self.persons_table}"
        )
        if self.omx_skims is not None:
            assert self.omx_skims.exists(), (
                f"omx_skims missing: {self.omx_skims}"
            )

    def to_record_store(self) -> RecordStore:
        """
        Convert outputs to a RecordStore for downstream steps.

        Returns
        -------
        RecordStore
            RecordStore containing input file records.
        """
        records = [
            FileRecord(
                file_path=str(self.land_use_table),
                short_name="land_use_asim_in",
                description="ActivitySim land use input table",
            ),
            FileRecord(
                file_path=str(self.households_table),
                short_name="households_asim_in",
                description="ActivitySim households input table",
            ),
            FileRecord(
                file_path=str(self.persons_table),
                short_name="persons_asim_in",
                description="ActivitySim persons input table",
            ),
        ]
        if self.omx_skims is not None:
            records.append(
                FileRecord(
                    file_path=str(self.omx_skims),
                    short_name="omx_skims",
                    description="ActivitySim OMX skims input",
                )
            )
        return RecordStore(recordList=records)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ActivitySimPreprocessOutputs":
        """
        Build outputs from a RecordStore.

        Parameters
        ----------
        record_store : RecordStore
            RecordStore produced by preprocessing.
        workspace : Workspace
            Workspace used to resolve paths.

        Returns
        -------
        ActivitySimPreprocessOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        values: Dict[str, Any] = {}
        for field_name, record_key in cls.record_keys.items():
            path = artifact_to_path(mapping.get(record_key), workspace)
            if path is not None:
                values[field_name] = Path(path)
        values["mutable_data_dir"] = Path(workspace.get_asim_mutable_data_dir())
        return cls(**values)


@dataclass
class ActivitySimRunOutputs:
    """
    Outputs from the ActivitySim run step.

    Attributes
    ----------
    output_dir : Path
        ActivitySim output directory.
    raw_outputs : dict
        Mapping of short_name to output path.
    """

    primary_output_attr: ClassVar[str] = "output_dir"
    output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        assert self.output_dir.exists(), f"output_dir missing: {self.output_dir}"
        for key, path in self.raw_outputs.items():
            if not path.exists():
                raise AssertionError(f"run output missing for {key}: {path}")

    def to_record_store(self) -> RecordStore:
        """
        Convert outputs to a RecordStore for downstream steps.

        Returns
        -------
        RecordStore
            RecordStore containing run output file records.
        """
        records = []
        for key, path in self.raw_outputs.items():
            records.append(
                FileRecord(
                    file_path=str(path),
                    short_name=key,
                    description=f"ActivitySim raw output: {key}",
                )
            )
        return RecordStore(recordList=records)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ActivitySimRunOutputs":
        """
        Build outputs from a RecordStore.

        Parameters
        ----------
        record_store : RecordStore
            RecordStore produced by the runner.
        workspace : Workspace
            Workspace used to resolve paths.

        Returns
        -------
        ActivitySimRunOutputs
            Parsed outputs.
        """
        mapping = record_store.to_mapping() if record_store is not None else {}
        raw_outputs: Dict[str, Path] = {}
        for key, value in mapping.items():
            path = artifact_to_path(value, workspace)
            if path is None:
                continue
            raw_outputs[key] = Path(path)
        return cls(
            output_dir=Path(workspace.get_asim_output_dir()),
            raw_outputs=raw_outputs,
        )


@dataclass
class ActivitySimPostprocessOutputs:
    """
    Outputs from the ActivitySim postprocess step.

    Attributes
    ----------
    usim_datastore_h5 : Path, optional
        Updated UrbanSim datastore path.
    asim_output_dir : Path
        ActivitySim output directory.
    processed_outputs : dict
        Mapping of short_name to postprocessed output path.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    usim_datastore_h5: Optional[Path]
    asim_output_dir: Path
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        assert self.asim_output_dir.exists(), (
            f"asim_output_dir missing: {self.asim_output_dir}"
        )
        if self.usim_datastore_h5 is not None:
            assert self.usim_datastore_h5.exists(), (
                f"usim_datastore_h5 missing: {self.usim_datastore_h5}"
            )
        for key, path in self.processed_outputs.items():
            if not path.exists():
                raise AssertionError(
                    f"postprocess output missing for {key}: {path}"
                )

    def to_record_store(self) -> RecordStore:
        """
        Convert outputs to a RecordStore for downstream steps.

        Returns
        -------
        RecordStore
            RecordStore containing postprocessed output file records.
        """
        records = []
        for key, path in self.processed_outputs.items():
            records.append(
                FileRecord(
                    file_path=str(path),
                    short_name=key,
                    description=f"ActivitySim output file: {key}",
                )
            )
        return RecordStore(recordList=records)

    @classmethod
    def from_record_store(
        cls, record_store: RecordStore, workspace: "Workspace"
    ) -> "ActivitySimPostprocessOutputs":
        """
        Build outputs from a RecordStore.

        Parameters
        ----------
        record_store : RecordStore
            RecordStore produced by the postprocessor.
        workspace : Workspace
            Workspace used to resolve paths.

        Returns
        -------
        ActivitySimPostprocessOutputs
            Parsed outputs.
        """
        usim_path = None
        processed_outputs: Dict[str, Path] = {}
        allowed_outputs = {
            "beam_plans",
            "disaggregate_accessibility",
            "households",
            "joint_tour_participants",
            "land_use",
            "non_mandatory_tour_destination_accessibility",
            "person_windows",
            "persons",
            "proto_disaggregate_accessibility",
            "proto_households",
            "proto_persons",
            "proto_persons_merged",
            "proto_tours",
            "school_destination_size",
            "school_modeled_size",
            "school_shadow_prices",
            "tours",
            "trips",
            "workplace_destination_size",
            "workplace_location_accessibility",
            "workplace_modeled_size",
            "workplace_shadow_prices",
        }
        if record_store is not None:
            for record in record_store.all_records():
                short_name = getattr(record, "short_name", "") or ""
                if short_name.startswith("usim_input_"):
                    usim_path = record.get_absolute_path(
                        base_path=workspace.full_path
                    )
                    continue
                if short_name in allowed_outputs:
                    record_path = record.get_absolute_path(
                        base_path=workspace.full_path
                    )
                    if record_path:
                        processed_outputs[short_name] = Path(record_path)
        return cls(
            usim_datastore_h5=Path(usim_path) if usim_path else None,
            asim_output_dir=Path(workspace.get_asim_output_dir()),
            processed_outputs=processed_outputs,
        )


@dataclass
class StepOutputsHolder:
    """
    Accumulates typed step outputs across the workflow.

    Attributes
    ----------
    activitysim_preprocess : ActivitySimPreprocessOutputs, optional
        Preprocess outputs.
    activitysim_run : ActivitySimRunOutputs, optional
        Run outputs.
    activitysim_postprocess : ActivitySimPostprocessOutputs, optional
        Postprocess outputs.
    """

    activitysim_preprocess: Optional[ActivitySimPreprocessOutputs] = None
    activitysim_run: Optional[ActivitySimRunOutputs] = None
    activitysim_postprocess: Optional[ActivitySimPostprocessOutputs] = None

    def set_attribute(self, step_name: str, outputs: Any) -> None:
        """
        Set a holder attribute by step name.

        Parameters
        ----------
        step_name : str
            Step name key.
        outputs : Any
            Outputs object to store.
        """
        attr = step_name.replace("-", "_")
        setattr(self, attr, outputs)

    def get_attribute(self, step_name: str) -> Any:
        """
        Retrieve a holder attribute by step name.

        Parameters
        ----------
        step_name : str
            Step name key.

        Returns
        -------
        Any
            Outputs object or None if missing.
        """
        attr = step_name.replace("-", "_")
        return getattr(self, attr, None)


STEP_OUTPUTS_CLASSES = {
    "activitysim_preprocess": ActivitySimPreprocessOutputs,
    "activitysim_run": ActivitySimRunOutputs,
    "activitysim_postprocess": ActivitySimPostprocessOutputs,
}


STEP_DEPENDENCIES = {
    "activitysim_preprocess": {
        "depends_on": [],
        "holder_inputs": [],
    },
    "activitysim_run": {
        "depends_on": ["activitysim_preprocess"],
        "holder_inputs": ["activitysim_preprocess"],
    },
    "activitysim_postprocess": {
        "depends_on": ["activitysim_run"],
        "holder_inputs": ["activitysim_run"],
    },
}


def validate_step_ready(step_name: str, outputs_holder: StepOutputsHolder) -> None:
    """
    Validate that dependencies for a step are satisfied.

    Parameters
    ----------
    step_name : str
        Step name to validate.
    outputs_holder : StepOutputsHolder
        Holder containing upstream outputs.
    """
    spec = STEP_DEPENDENCIES.get(step_name)
    if not spec:
        logger.warning("No dependency spec for %s; skipping validation", step_name)
        return
    for holder_input_key in spec["holder_inputs"]:
        holder_attr = holder_input_key.replace("-", "_")
        if getattr(outputs_holder, holder_attr, None) is None:
            raise RuntimeError(
                f"{step_name} requires {holder_input_key} to complete first, "
                "but it has not been executed or failed."
            )


def require_common_runtime(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Require runtime kwargs shared across step functions.

    Parameters
    ----------
    *names : str
        Additional required runtime kwarg names.

    Returns
    -------
    callable
        Decorator that enforces the runtime kwargs.
    """
    return cr.require_runtime_kwargs("settings", "state", "workspace", *names)


def _make_generic_step_function(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
    model_name: str,
    phase: str,
    outputs_class: Type[StepOutputsT],
    component_getter: Callable[[ModelFactory, WorkflowState], Any],
    component_executor: Callable[..., RecordStore],
    outputs_holder_setter: Callable[[StepOutputsHolder, StepOutputsT], None],
    input_logger: Optional[
        Callable[
            [PilatesConfig, WorkflowState, "Workspace", StepOutputsHolder],
            Mapping[str, Any],
        ]
    ] = None,
    output_logger: Optional[
        Callable[
            [StepOutputsT, PilatesConfig, WorkflowState, "Workspace", StepOutputsHolder],
            None,
        ]
    ] = None,
) -> Callable[..., None]:
    """
    Build a step function with common RecordStore-to-StepOutputs plumbing.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder used to store outputs for downstream steps.
    model_name : str
        Model identifier for logging.
    phase : str
        Step phase name (preprocess/run/postprocess).
    outputs_class : type
        StepOutputs dataclass type.
    component_getter : callable
        Callable that returns the component instance.
    component_executor : callable
        Callable that executes the component.
    outputs_holder_setter : callable
        Callback that stores outputs on the holder.
    input_logger : callable, optional
        Optional hook for logging step inputs.
    output_logger : callable, optional
        Optional hook for logging step outputs.

    Returns
    -------
    callable
        Step function compatible with Consist scenario execution.
    """
    @cr.require_runtime_kwargs("settings", "state", "workspace")
    def _step_func(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        **kwargs: Any,
    ) -> None:
        factory = ModelFactory()
        component = component_getter(factory, state)

        extra_kwargs: Dict[str, Any] = {}
        if input_logger is not None:
            extra_kwargs = input_logger(
                settings, state, workspace, outputs_holder
            ) or {}

        record_store = component_executor(
            component,
            workspace,
            outputs_holder,
            **extra_kwargs,
            **kwargs,
        )

        step_outputs = record_store_to_outputs(
            record_store=record_store,
            output_class=outputs_class,
            workspace=workspace,
        )
        step_outputs.validate()
        outputs_holder_setter(outputs_holder, step_outputs)

        if output_logger is not None:
            output_logger(step_outputs, settings, state, workspace, outputs_holder)

        logger.info("%s %s completed successfully", model_name, phase)

    return _step_func


def _execute_preprocess(
    preprocessor: Any,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a preprocessor using only the workspace.

    Parameters
    ----------
    preprocessor : object
        Preprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder for upstream outputs (unused).

    Returns
    -------
    RecordStore
        Preprocessor outputs.
    """
    return run_preprocessor(preprocessor, workspace)


def _execute_run(
    runner: Any,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    *,
    extra_inputs: Optional[RecordStore] = None,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a runner using upstream preprocess outputs.

    Parameters
    ----------
    runner : object
        Runner instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with preprocess outputs.
    extra_inputs : RecordStore, optional
        Additional inputs to merge into the runner input store.

    Returns
    -------
    RecordStore
        Runner outputs.
    """
    upstream = outputs_holder.activitysim_preprocess
    if upstream is None:
        raise RuntimeError("ActivitySim preprocess must complete first")
    input_store = upstream.to_record_store()
    if extra_inputs is not None:
        input_store += extra_inputs
    return run_runner(runner, input_store, workspace)


def _execute_postprocess(
    postprocessor: Any,
    workspace: "Workspace",
    outputs_holder: StepOutputsHolder,
    **kwargs: Any,
) -> RecordStore:
    """
    Execute a postprocessor using upstream run outputs.

    Parameters
    ----------
    postprocessor : object
        Postprocessor instance.
    workspace : Workspace
        Workspace used to resolve paths.
    outputs_holder : StepOutputsHolder
        Holder with run outputs.

    Returns
    -------
    RecordStore
        Postprocessor outputs.
    """
    upstream = outputs_holder.activitysim_run
    if upstream is None:
        raise RuntimeError("ActivitySim run must complete first")
    raw_outputs = upstream.to_record_store()
    return run_postprocessor(postprocessor, raw_outputs, workspace)


def make_urbansim_step(
    *,
    coupler: Any,
    year: int,
) -> Callable[..., None]:
    """
    Build the UrbanSim step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    year : int
        Simulation year for the run.

    Returns
    -------
    callable
        Step function for UrbanSim.
    """
    @require_common_runtime("usim_data_dir", "expected_outputs")
    def _run_urbansim_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        usim_data_dir: str,
        expected_outputs: Dict[str, Any],
    ) -> None:
        if state.is_start_year() and settings.activitysim.warm_start_activities:
            logger.info("[Main] Running warm start activities for ActivitySim.")
            warm_start_activities(settings, state, workspace)

        forecast_land_use(settings, year, state, workspace)
        usim_output_path = expected_outputs.get("usim_datastore_h5")
        if not usim_output_path:
            usim_output_fname = settings.urbansim.output_file_template.format(
                year=state.forecast_year
            )
            usim_output_path = os.path.join(usim_data_dir, usim_output_fname)
        if os.path.exists(usim_output_path):
            log_and_set_output(
                key="usim_datastore_h5",
                path=usim_output_path,
                description=(
                    f"UrbanSim datastore output for year {state.forecast_year}"
                ),
                coupler=coupler,
            )

    return _run_urbansim_step


def make_atlas_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    """
    Build the ATLAS step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.

    Returns
    -------
    callable
        Step function for ATLAS.
    """
    @cr.require_runtime_kwargs(
        "atlas_state",
        "base_state",
        "preprocessor",
        "runner",
        "postprocessor",
        "workspace",
        "usim_datastore_h5_path",
        "atlas_year",
        "expected_outputs",
    )
    def _run_atlas_step(
        *,
        atlas_state: WorkflowState,
        base_state: WorkflowState,
        preprocessor: Any,
        runner: Any,
        postprocessor: Any,
        workspace: Workspace,
        usim_datastore_h5_path: str,
        atlas_year: int,
        expected_outputs: Dict[str, Any],
    ) -> None:
        preprocessor.update_state(atlas_state)
        input_data = preprocessor.preprocess(workspace)

        runner.update_state(atlas_state)
        try:
            raw_outputs = runner.run(input_data, workspace)
            postprocessor.update_state(atlas_state)
            postprocessor.postprocess(raw_outputs, workspace)

            atlas_output_dir = expected_outputs.get("atlas_output_dir")
            if not atlas_output_dir:
                atlas_output_dir = workspace.get_atlas_output_dir()
            if os.path.exists(atlas_output_dir):
                log_and_set_output(
                    key="atlas_output_dir",
                    path=atlas_output_dir,
                    description=f"ATLAS output directory for year {atlas_year}",
                    coupler=coupler,
                )

            atlas_usim_output = expected_outputs.get("usim_datastore_h5")
            if not atlas_usim_output:
                atlas_usim_output = usim_datastore_h5_path
            if os.path.exists(atlas_usim_output):
                log_and_set_output(
                    key="usim_datastore_h5",
                    path=atlas_usim_output,
                    description=(
                        "UrbanSim datastore after ATLAS update for year "
                        f"{atlas_year}"
                    ),
                    coupler=coupler,
                )
            else:
                logger.warning(
                    "[Main] UrbanSim datastore not found after ATLAS postprocess: %s",
                    atlas_usim_output,
                )
        except Exception:
            from pilates.utils.failure_handling import persist_state_on_error

            persist_state_on_error(base_state, f"ATLAS year {atlas_year}")
            sys.exit(1)

    return _run_atlas_step


def make_activitysim_compile_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    """
    Build the ActivitySim compile step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.

    Returns
    -------
    callable
        Step function for ActivitySim compile.
    """
    @require_common_runtime("compile_outputs_holder", "expected_outputs")
    def _run_activitysim_compile_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        compile_outputs_holder: Dict[str, Any],
        expected_outputs: Dict[str, Any],
    ) -> None:
        factory = ModelFactory()
        from workflow_state import WorkflowState as _WorkflowState

        preprocessor = factory.get_preprocessor(
            "activitysim",
            state,
            major_stage=_WorkflowState.Stage.activity_demand,
        )
        compile_runner = factory.get_runner(
            "activitysim_compile",
            state,
            major_stage=_WorkflowState.Stage.activity_demand,
        )

        input_store = preprocessor.preprocess(workspace)
        omx_record = None
        if input_store:
            for record in input_store.all_records():
                if getattr(record, "short_name", None) == "omx_skims":
                    omx_record = record
                    break
        if omx_record is not None:
            omx_path = omx_record.get_absolute_path(base_path=workspace.full_path)
            if omx_path and os.path.exists(omx_path):
                cr.log_input(
                    omx_path,
                    key="omx_skims",
                    description="ActivitySim compile input skims (OMX)",
                )
        compile_outputs = compile_runner.run(input_store, workspace)

        compile_outputs_holder["input_store"] = input_store
        compile_outputs_holder["compile_outputs"] = compile_outputs

        zarr_record = None
        if compile_outputs:
            for record in compile_outputs.all_records():
                if record.short_name == "zarr_skims":
                    zarr_record = record
                    break
        zarr_output_path = expected_outputs.get("zarr_skims")
        if not zarr_output_path and zarr_record is not None:
            zarr_output_path = zarr_record.file_path
        if zarr_output_path and os.path.exists(zarr_output_path):
            log_and_set_output(
                key="zarr_skims",
                path=zarr_output_path,
                description="ActivitySim compiled zarr skims",
                coupler=coupler,
            )

    return _run_activitysim_compile_step


def make_activitysim_preprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim preprocess step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing preprocess outputs.

    Returns
    -------
    callable
        Step function for ActivitySim preprocess.
    """
    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        usim_input = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            usim_input = resolve_artifact_from_value(
                get_value("usim_datastore_h5"),
                key="usim_datastore_h5",
                workspace=workspace,
            )
        usim_path = artifact_to_path(usim_input, workspace)
        if usim_path and os.path.exists(usim_path):
            log_and_set_input(
                key="usim_datastore_h5",
                path=usim_path,
                description=(
                    f"UrbanSim datastore for ActivitySim year {state.year}"
                ),
                coupler=coupler,
            )
        return {}

    def _log_outputs(
        outputs: ActivitySimPreprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        log_and_set_output(
            key="asim_mutable_data_dir",
            path=str(outputs.mutable_data_dir),
            description="ActivitySim mutable data directory",
            coupler=coupler,
        )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="preprocess",
        outputs_class=ActivitySimPreprocessOutputs,
        component_getter=lambda factory, state: factory.get_preprocessor(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_preprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_preprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_activitysim_run_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim run step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing run outputs.

    Returns
    -------
    callable
        Step function for ActivitySim run.
    """
    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.activitysim_preprocess
        if upstream is None:
            raise RuntimeError("ActivitySim preprocess must complete first")

        log_and_set_input(
            key="asim_mutable_data_dir",
            path=str(upstream.mutable_data_dir),
            description=(
                f"ActivitySim mutable inputs for year {state.year}, iter {state.iteration}"
            ),
            coupler=coupler,
        )

        extra_inputs = RecordStore()
        zarr_value = None
        get_value = getattr(coupler, "get", None)
        if callable(get_value):
            zarr_value = resolve_artifact_from_value(
                get_value("zarr_skims"),
                key="zarr_skims",
                workspace=workspace,
            )
        zarr_path = artifact_to_path(zarr_value, workspace)
        if not zarr_path:
            candidate = os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
            if os.path.exists(candidate):
                zarr_path = candidate
        if zarr_path and os.path.exists(zarr_path):
            extra_inputs.add_record(
                FileRecord(
                    file_path=zarr_path,
                    short_name="zarr_skims",
                    description="Compiled ActivitySim skims (Zarr)",
                )
            )
            log_and_set_input(
                key="zarr_skims",
                path=zarr_path,
                description=(
                    f"ActivitySim compiled skims for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )
        return {"extra_inputs": extra_inputs}

    def _log_outputs(
        outputs: ActivitySimRunOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        log_and_set_output(
            key="asim_output_dir",
            path=str(outputs.output_dir),
            description=(
                f"ActivitySim output directory for year {state.year}, iter {state.iteration}"
            ),
            coupler=coupler,
        )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="run",
        outputs_class=ActivitySimRunOutputs,
        component_getter=lambda factory, state: factory.get_runner(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_run,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_run", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_activitysim_postprocess_step(
    *,
    coupler: Any,
    outputs_holder: StepOutputsHolder,
) -> Callable[..., None]:
    """
    Build the ActivitySim postprocess step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.
    outputs_holder : StepOutputsHolder
        Holder for storing postprocess outputs.

    Returns
    -------
    callable
        Step function for ActivitySim postprocess.
    """
    def _log_inputs(
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> Dict[str, Any]:
        upstream = holder.activitysim_run
        if upstream is None:
            raise RuntimeError("ActivitySim run must complete first")
        log_and_set_input(
            key="asim_output_dir",
            path=str(upstream.output_dir),
            description=(
                f"ActivitySim outputs for year {state.year}, iter {state.iteration}"
            ),
            coupler=coupler,
        )
        usim_dir = workspace.get_usim_mutable_data_dir()
        if os.path.exists(usim_dir):
            log_and_set_input(
                key="usim_mutable_data_dir",
                path=usim_dir,
                description="UrbanSim mutable data directory",
                coupler=coupler,
            )
        return {}

    def _log_outputs(
        outputs: ActivitySimPostprocessOutputs,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        holder: StepOutputsHolder,
    ) -> None:
        if outputs.usim_datastore_h5 is not None:
            log_and_set_output(
                key="usim_datastore_h5",
                path=str(outputs.usim_datastore_h5),
                description=(
                    f"UrbanSim datastore updated by ActivitySim for year {state.forecast_year}"
                ),
                coupler=coupler,
            )

    return _make_generic_step_function(
        coupler=coupler,
        outputs_holder=outputs_holder,
        model_name="activitysim",
        phase="postprocess",
        outputs_class=ActivitySimPostprocessOutputs,
        component_getter=lambda factory, state: factory.get_postprocessor(
            "activitysim", state, WorkflowState.Stage.activity_demand
        ),
        component_executor=_execute_postprocess,
        outputs_holder_setter=lambda holder, outputs: setattr(
            holder, "activitysim_postprocess", outputs
        ),
        input_logger=_log_inputs,
        output_logger=_log_outputs,
    )


def make_beam_step(
    *,
    coupler: Any,
) -> Callable[..., None]:
    """
    Build the BEAM step function.

    Parameters
    ----------
    coupler : object
        Consist coupler for input/output logging.

    Returns
    -------
    callable
        Step function for BEAM.
    """
    @require_common_runtime(
        "activity_demand_outputs",
        "previous_beam_outputs",
        "beam_inputs",
        "beam_mutable_dir",
        "beam_mutable_description",
        "beam_outputs_holder",
        "expected_outputs",
    )
    def _run_beam_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
        activity_demand_outputs: RecordStore,
        previous_beam_outputs: RecordStore,
        beam_inputs: Dict[str, Any],
        beam_mutable_dir: str,
        beam_mutable_description: str,
        beam_outputs_holder: Dict[str, Any],
        expected_outputs: Dict[str, Any],
    ) -> None:
        if beam_inputs:
            cr.log_artifacts(beam_inputs, direction="input")
        if beam_mutable_dir:
            cr.log_input(
                beam_mutable_dir,
                key="beam_mutable_data_dir",
                description=beam_mutable_description or "",
            )
        beam_outputs_holder["beam_outputs"] = run_traffic_assignment(
            settings,
            state,
            workspace,
            activity_demand_outputs,
            previous_beam_outputs,
        )
        beam_output_dir = expected_outputs.get("beam_output_dir")
        if not beam_output_dir:
            beam_output_dir = workspace.get_beam_output_dir()
        if os.path.exists(beam_output_dir):
            log_and_set_output(
                key="beam_output_dir",
                path=beam_output_dir,
                description=(
                    f"BEAM output directory for year {state.year}, iter {state.iteration}"
                ),
                coupler=coupler,
            )

        output_store = beam_outputs_holder.get("beam_outputs")
        update_coupler_from_beam_outputs(output_store, coupler, workspace)

    return _run_beam_step


def make_postprocessing_step() -> Callable[..., None]:
    """
    Build the postprocessing step function.

    Returns
    -------
    callable
        Step function for postprocessing.
    """
    @require_common_runtime()
    def _run_postprocessing_step(
        *,
        settings: PilatesConfig,
        state: WorkflowState,
        workspace: Workspace,
    ) -> None:
        if "postprocessing" in settings:
            from pilates.postprocessing.postprocessor import (
                copy_outputs_to_mep,
                process_event_file,
            )

            process_event_file(settings, state.forecast_year, state.current_inner_iter)
            copy_outputs_to_mep(
                settings,
                state.forecast_year,
                state.current_inner_iter,
                workspace,
            )

    return _run_postprocessing_step
