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
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pilates.generic.records import FileRecord, RecordStore, sanitize_artifact_key
from pilates.workflows.coupler_namespace import resolve_coupler_value

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
        Snapshot view of upstream ``StepOutputsHolder`` entries.
    """

    settings: Any = None
    state: Any = None
    workspace: Any = None
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


def iter_step_output_items(outputs: Any) -> Tuple[Tuple[str, Any, str], ...]:
    """
    Return direct typed-output artifact items without round-tripping through RecordStore.

    Parameters
    ----------
    outputs : Any
        Step outputs object exposing ``_iter_record_items()``.

    Returns
    -------
    tuple[tuple[str, Any, str], ...]
        Ordered ``(key, path, description)`` items.
    """
    iter_items = getattr(outputs, "_iter_record_items", None)
    if not callable(iter_items):
        raise TypeError(
            f"{outputs.__class__.__name__} must implement _iter_record_items() "
            "for shared workflow artifact publication."
        )
    return tuple(iter_items())


def step_output_mapping(
    outputs: Any,
    *,
    warn_lossy: bool = True,
) -> Dict[str, str]:
    """
    Build a lossy key -> path mapping directly from typed outputs.

    Warning
    -------
    This helper intentionally strips artifact identity and content-hash-bearing
    objects down to filesystem path strings. Use
    ``step_output_handoff_mapping(...)`` for runtime handoffs between workflow
    steps/stages/iterations where Consist lineage should be preserved.

    Parameters
    ----------
    outputs : Any
        Step outputs object exposing ``_iter_record_items()``.
    warn_lossy : bool, optional
        Emit a warning that this helper is lossy. Callers should set this to
        ``False`` only for intentionally path-only serialization/replay flows.

    Returns
    -------
    dict[str, str]
        Mapping of output key to filesystem path string.
    """
    if warn_lossy:
        logger.warning(
            "step_output_mapping(...) is lossy and should not be used for runtime "
            "workflow handoffs; prefer step_output_handoff_mapping(...)."
        )
    mapping: Dict[str, str] = {}
    for key, path, _ in iter_step_output_items(outputs):
        sanitized_key = sanitize_artifact_key(key)
        if sanitized_key is None:
            logger.warning(
                "Invalid typed-output artifact key '%s' could not be sanitized; skipping.",
                key,
            )
            continue
        if sanitized_key != key:
            logger.warning(
                "Invalid typed-output artifact key '%s' sanitized to '%s' for Consist compatibility.",
                key,
                sanitized_key,
            )
        if sanitized_key in mapping:
            logger.warning(
                "Duplicate typed-output artifact key '%s' detected; keeping first path and skipping later duplicate.",
                sanitized_key,
            )
            continue
        mapping[sanitized_key] = str(path)
    return mapping


def step_output_handoff_mapping(
    outputs: Any,
    *,
    coupler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build a runtime handoff mapping, preserving coupler-published artifacts.

    This is for step-to-step or iteration-to-iteration handoffs where artifact
    identity matters. When the current step has already published a key into the
    coupler, the coupler value is preferred over the raw filesystem path string.
    """
    mapping: Dict[str, Any] = {}
    if coupler is None:
        logger.warning(
            "step_output_handoff_mapping(...) called without a readable coupler; "
            "handoff values will fall back to raw paths."
        )
    for key, path, _ in iter_step_output_items(outputs):
        sanitized_key = sanitize_artifact_key(key)
        if sanitized_key is None:
            logger.warning(
                "Invalid typed-output artifact key '%s' could not be sanitized; skipping.",
                key,
            )
            continue
        if sanitized_key != key:
            logger.warning(
                "Invalid typed-output artifact key '%s' sanitized to '%s' for Consist compatibility.",
                key,
                sanitized_key,
            )
        if sanitized_key in mapping:
            logger.warning(
                "Duplicate typed-output artifact key '%s' detected; keeping first path and skipping later duplicate.",
                sanitized_key,
            )
            continue
        resolved = resolve_coupler_value(coupler, sanitized_key)
        mapping[sanitized_key] = (
            resolved.value if resolved.value is not None else str(path)
        )
    return mapping


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


def declared_outputs_for_step_outputs_class(
    outputs_class: Type[Any],
) -> Tuple[str, ...]:
    """
    Resolve canonical declared output keys for a ``StepOutputs`` class.

    Precedence:
    1. Explicit ``declared_outputs`` class attribute.
    2. Fallback to required ``record_keys`` fields.

    Parameters
    ----------
    outputs_class : type
        Step outputs dataclass type.

    Returns
    -------
    tuple[str, ...]
        Ordered, deduplicated output key tuple.
    """
    explicit = getattr(outputs_class, "declared_outputs", None) or ()
    explicit_keys = [key for key in explicit if isinstance(key, str)]
    if explicit_keys:
        return tuple(dict.fromkeys(explicit_keys))

    record_keys = getattr(outputs_class, "record_keys", None) or {}
    required_fields = getattr(outputs_class, "required_path_fields", ()) or ()
    inferred: list[str] = []
    for field_name in required_fields:
        key = record_keys.get(field_name)
        if isinstance(key, str):
            inferred.append(key)
    return tuple(dict.fromkeys(inferred))


def required_outputs_for_step_outputs_class(
    outputs_class: Type[Any],
    state: Any = None,
) -> Tuple[str, ...]:
    """
    Resolve strict required output keys for a ``StepOutputs`` class.

    Precedence:
    1. Explicit ``required_outputs`` class attribute.
    2. State-expanded ``required_output_families`` class attribute.
    2. Fallback to all declared outputs for the class.
    """
    explicit = getattr(outputs_class, "required_outputs", None) or ()
    explicit_keys = [key for key in explicit if isinstance(key, str)]
    if explicit_keys:
        return tuple(dict.fromkeys(explicit_keys))
    families = getattr(outputs_class, "required_output_families", None) or ()
    family_keys = [key for key in families if isinstance(key, str)]
    if family_keys and state is not None:
        year = getattr(state, "forecast_year", None)
        if year is None:
            year = getattr(state, "current_year", getattr(state, "year", None))
        iteration = getattr(
            state,
            "current_inner_iter",
            getattr(state, "iteration", None),
        )
        atlas_year = getattr(state, "atlas_year", None)
        format_values = {
            "year": year,
            "forecast_year": getattr(state, "forecast_year", None),
            "iteration": iteration,
            "atlas_year": atlas_year,
        }
        expanded: list[str] = []
        for family in family_keys:
            try:
                expanded.append(family.format(**format_values))
            except Exception:
                continue
        if expanded:
            return tuple(dict.fromkeys(expanded))
    return declared_outputs_for_step_outputs_class(outputs_class)


class StepOutputsBase:
    """
    Base class for typed step outputs with RecordStore conversion.
    """

    declared_outputs: ClassVar[Tuple[str, ...]] = ()
    required_outputs: ClassVar[Tuple[str, ...]] = ()
    required_output_families: ClassVar[Tuple[str, ...]] = ()
    record_keys: ClassVar[Dict[str, str]] = {}
    record_descriptions: ClassVar[Dict[str, str]] = {}
    default_description: ClassVar[str] = "Step output"
    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_path_fields: ClassVar[Tuple[str, ...]] = ()
    dict_path_fields: ClassVar[Tuple[str, ...]] = ()
    validators: ClassVar[Tuple[OutputValidator, ...]] = ()

    @classmethod
    def declared_output_keys(cls) -> Tuple[str, ...]:
        """
        Return canonical declared output keys for this ``StepOutputs`` class.
        """
        return declared_outputs_for_step_outputs_class(cls)

    @classmethod
    def required_output_keys(cls, state: Any = None) -> Tuple[str, ...]:
        """
        Return strict required output keys for this ``StepOutputs`` class.
        """
        return required_outputs_for_step_outputs_class(cls, state=state)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield records as (short_name, path, description) tuples.

        Returns
        -------
        iterable of tuple
            Record metadata for RecordStore conversion.
        """
        for field_name, short_name in self.record_keys.items():
            value = getattr(self, field_name, None)
            if value is None:
                continue
            description = self.record_descriptions.get(
                field_name, f"{self.default_description}: {short_name}"
            )
            yield short_name, value, description

    def to_record_store(self) -> RecordStore:
        """
        Convert outputs to a RecordStore for downstream steps.

        Returns
        -------
        RecordStore
            RecordStore containing output file records.
        """
        records = []
        for short_name, path, description in self._iter_record_items():
            records.append(
                FileRecord(
                    file_path=str(path),
                    short_name=short_name,
                    description=description,
                )
            )
        return RecordStore(recordList=records)

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
