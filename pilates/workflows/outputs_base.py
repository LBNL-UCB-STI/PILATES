from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

if TYPE_CHECKING:
    from pilates.config import PilatesConfig
    from pilates.workspace import Workspace
    from pilates.workflows.atlas_state import AtlasSubState
    from pilates.workflows.state import WorkflowState

StepOutputsT = TypeVar("StepOutputsT")
logger = logging.getLogger(__name__)
ValidationLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationResult:
    """
    Result payload emitted by a semantic output validator.

    Attributes
    ----------
    message : str
        Human-readable validation message.
    metadata : mapping, optional
        Optional structured context to aid debugging.
    """

    message: str
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ValidationContext:
    """
    Runtime context supplied to semantic output validators.

    Attributes
    ----------
    settings : Any, optional
        Runtime settings object for the step.
    state : Any, optional
        Workflow state object for the step.
    workspace : Any, optional
        Workspace object used by the step.
    step_name : str, optional
        Canonical step name (for example ``activitysim_preprocess``).
    upstream_outputs : mapping
        Optional typed upstream-output context for semantic validation.
    """

    settings: Optional["PilatesConfig"] = None
    state: Optional["WorkflowState | AtlasSubState"] = None
    workspace: Optional["Workspace"] = None
    step_name: Optional[str] = None
    upstream_outputs: Mapping[str, Any] = field(default_factory=dict)


class OutputValidator(Protocol):
    """
    Protocol for semantic output validators.
    """

    name: str
    level: ValidationLevel

    def validate(
        self, outputs: Any, context: ValidationContext
    ) -> List[ValidationResult]:
        """
        Validate typed outputs under a runtime context.
        """

        ...


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
    for output_field in fields(output_class):
        if output_field.name not in data:
            continue
        value = data[output_field.name]
        if value is None:
            kwargs[output_field.name] = None
            continue
        if output_field.type is Path or _is_optional_path_type(output_field.type):
            kwargs[output_field.name] = Path(value)
            continue
        if _is_dict_path_type(output_field.type):
            kwargs[output_field.name] = {key: Path(val) for key, val in value.items()}
            continue
        kwargs[output_field.name] = value
    return output_class(**kwargs)


class StepOutputsBase:
    """
    Base class for typed step outputs.
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_path_fields: ClassVar[Tuple[str, ...]] = ()
    dict_path_fields: ClassVar[Tuple[str, ...]] = ()
    validators: ClassVar[Tuple[OutputValidator, ...]] = ()

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield records as (short_name, path, description) tuples.

        Returns
        -------
        iterable of tuple
            Record metadata tuples for step output inspection and serialization.
        """
        return ()

    def validate(self, context: Optional[ValidationContext] = None) -> None:
        """
        Validate that expected output paths exist.
        """

        def _path_exists(path_value: Any) -> bool:
            if path_value is None:
                return False
            exists_attr = getattr(path_value, "exists", None)
            if callable(exists_attr):
                return bool(exists_attr())
            if isinstance(path_value, (str, os.PathLike)):
                return Path(path_value).exists()
            return False

        logger.debug(
            "Validating %s: required=%s optional=%s dict=%s",
            self.__class__.__name__,
            self.required_path_fields,
            self.optional_path_fields,
            self.dict_path_fields,
        )
        for field_name in self.required_path_fields:
            path = getattr(self, field_name, None)
            if not _path_exists(path):
                raise AssertionError(f"{field_name} missing: {path}")

        for field_name in self.optional_path_fields:
            path = getattr(self, field_name, None)
            if path is None:
                continue
            if not _path_exists(path):
                raise AssertionError(f"{field_name} missing: {path}")

        for field_name in self.dict_path_fields:
            paths_dict = getattr(self, field_name, {}) or {}
            for key, path in paths_dict.items():
                if not _path_exists(path):
                    raise AssertionError(f"{field_name}[{key}] missing: {path}")

        validators = getattr(self.__class__, "validators", ()) or ()
        if not validators:
            return

        validation_context = context or ValidationContext(
            step_name=self.__class__.__name__
        )
        step_label = validation_context.step_name or self.__class__.__name__
        error_messages: list[str] = []

        for validator in validators:
            validator_name = getattr(validator, "name", validator.__class__.__name__)
            level = getattr(validator, "level", "error")
            if level not in ("error", "warning"):
                logger.warning(
                    "OUTPUT VALIDATION WARNING (%s): validator '%s' has unsupported "
                    "level '%s'; treating as 'error'.",
                    step_label,
                    validator_name,
                    level,
                )
                level = "error"

            try:
                results = validator.validate(self, validation_context) or []
            except Exception as exc:
                raise AssertionError(
                    f"Semantic validator '{validator_name}' crashed during "
                    f"validation for step '{step_label}': {exc}"
                ) from exc

            for result in results:
                message = str(getattr(result, "message", "")).strip()
                if not message:
                    continue
                metadata = getattr(result, "metadata", None)
                metadata_suffix = f" metadata={metadata}" if metadata else ""
                rendered = f"[{validator_name}] {message}{metadata_suffix}"
                if level == "warning":
                    logger.warning(
                        "OUTPUT VALIDATION WARNING (%s): %s",
                        step_label,
                        rendered,
                    )
                else:
                    error_messages.append(rendered)

        if error_messages:
            raise AssertionError(
                f"Semantic validation failed for step '{step_label}' "
                f"({self.__class__.__name__}). "
                "Fix the flagged output contract issue(s): " + "; ".join(error_messages)
            )
