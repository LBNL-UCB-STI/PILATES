from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pilates.generic.records import FileRecord, RecordStore

StepOutputsT = TypeVar("StepOutputsT")
logger = logging.getLogger(__name__)


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


class StepOutputsBase:
    """
    Base class for typed step outputs with RecordStore conversion.
    """

    record_keys: ClassVar[Dict[str, str]] = {}
    record_descriptions: ClassVar[Dict[str, str]] = {}
    default_description: ClassVar[str] = "Step output"
    required_path_fields: ClassVar[Tuple[str, ...]] = ()
    optional_path_fields: ClassVar[Tuple[str, ...]] = ()
    dict_path_fields: ClassVar[Tuple[str, ...]] = ()

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

    def validate(self) -> None:
        """
        Validate that expected output paths exist.
        """
        logger.debug(
            "Validating %s: required=%s optional=%s dict=%s",
            self.__class__.__name__,
            self.required_path_fields,
            self.optional_path_fields,
            self.dict_path_fields,
        )
        for field_name in self.required_path_fields:
            path = getattr(self, field_name, None)
            if path is None or not path.exists():
                raise AssertionError(f"{field_name} missing: {path}")

        for field_name in self.optional_path_fields:
            path = getattr(self, field_name, None)
            if path is None:
                continue
            if not path.exists():
                raise AssertionError(f"{field_name} missing: {path}")

        for field_name in self.dict_path_fields:
            paths_dict = getattr(self, field_name, {}) or {}
            for key, path in paths_dict.items():
                if path is None or not path.exists():
                    raise AssertionError(f"{field_name}[{key}] missing: {path}")
