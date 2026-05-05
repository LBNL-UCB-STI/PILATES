"""
pilates/generic/records.py

Lightweight data structures for passing outputs between model methods.

This module defines minimal record types used during model execution:
- FileRecord: Simple file reference with essential metadata
- RecordStore: Container for passing outputs between model preprocessors/runners/postprocessors

All provenance tracking, lineage, and OpenLineage integration is handled by Consist.
These classes exist ONLY for inter-model data flow.
"""

import uuid
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)
_ARTIFACT_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,255}$")


def _sanitize_artifact_key(key: str) -> Optional[str]:
    if _ARTIFACT_KEY_RE.match(key):
        return key
    sanitized = re.sub(r"[^A-Za-z0-9_.:/-]", "_", key)
    if not sanitized:
        return None
    if not re.match(r"^[A-Za-z0-9_]", sanitized):
        sanitized = f"_{sanitized}"
    if len(sanitized) > 256:
        sanitized = sanitized[:256]
    return sanitized if _ARTIFACT_KEY_RE.match(sanitized) else None


def sanitize_artifact_key(key: str) -> Optional[str]:
    """
    Normalize a string into a Consist-compatible artifact key.
    """
    return _sanitize_artifact_key(key)


@dataclass(kw_only=True)
class FileRecord:
    """
    Minimal file record for inter-model data passing.

    Attributes:
        file_path: Path to the file on disk (relative to workspace recommended)
        short_name: Human-friendly name used as artifact key in Consist
        description: Optional descriptive text
        year: Optional year tag for context
        iteration: Optional iteration index for context
        metadata: Arbitrary key/value metadata dict
        content_hash: Optional content hash for the file contents
        unique_id: Stable identifier (auto-generated if not provided)
        container_uri: Optional Consist container URI when available
        uri: Optional legacy Consist URI when available
        h5_tables_used: Optional list of HDF5 table paths to log (exact matches)
    """

    file_path: str
    short_name: str
    description: Optional[str] = None
    year: Optional[int] = None
    iteration: Optional[int] = None
    sub_iteration: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    unique_id: Optional[str] = None
    container_uri: Optional[str] = None
    uri: Optional[str] = None
    h5_tables_used: Optional[List[str]] = None

    def __post_init__(self):
        # Auto-generate unique_id if not provided
        if self.unique_id is None:
            self.unique_id = str(uuid.uuid4())

    def __hash__(self):
        return hash(self.unique_id)

    def get_absolute_path(self, base_path: Optional[str] = None) -> Optional[str]:
        """
        Return an absolute file path for this record.

        If file_path is already absolute, returns it as-is. If it is relative and a
        base_path is provided, joins against base_path. If base_path is not provided,
        resolves relative to the current working directory.
        """
        if not self.file_path:
            return None
        if os.path.isabs(self.file_path):
            return self.file_path
        if base_path:
            return os.path.abspath(os.path.join(base_path, self.file_path))
        return os.path.abspath(self.file_path)


@dataclass(kw_only=True)
class RecordStore:
    """
    Lightweight container for record objects.

    Used to pass outputs from one model method to the next.
    Records are keyed by unique_id for easy lookup and merging.
    """

    records: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        recordDict: Optional[Dict[str, Any]] = None,
        recordList: Optional[List[Any]] = None,
    ) -> None:
        """
        Initialize RecordStore from dict or list of records.

        Args:
            recordDict: Dictionary keyed by unique_id
            recordList: List of record objects with unique_id
        """
        if recordDict is not None:
            if isinstance(recordDict, dict):
                self.records = recordDict
            else:
                raise TypeError("recordDict must be a dictionary")
        else:
            self.records = {}

        if recordList is not None:
            for record in recordList:
                self.add_record(record)

    def add_record(self, record: Any) -> "RecordStore":
        """Add a record to the store."""
        if not hasattr(record, "unique_id"):
            raise TypeError("Record must have a unique_id attribute")
        if record.unique_id is None:
            raise ValueError("Record unique_id must be set")
        self.records[record.unique_id] = record
        return self

    def all_records(self) -> List[Any]:
        """Return all records in the store."""
        return list(self.records.values())

    def all_unique_ids(self) -> List[str]:
        """Return all unique IDs in the store."""
        return list(self.records.keys())

    def remove_record_type(self, short_name: str) -> None:
        """Remove all records with the given short_name."""
        for record in list(self.all_records()):
            if getattr(record, "short_name", None) == short_name:
                logger.info(f"Removing record type {short_name} from record store")
                del self.records[record.unique_id]

    def get_record(self, unique_id: str) -> Optional[Any]:
        """Get a record by unique_id."""
        return self.records.get(unique_id)

    def __add__(self, other: "RecordStore") -> "RecordStore":
        """Merge two RecordStores (returns new instance, doesn't mutate)."""
        if not isinstance(other, RecordStore):
            raise TypeError("Operand must be a RecordStore")
        combined = {**self.records, **other.records}
        return RecordStore(recordDict=combined)

    def __iadd__(self, other: "RecordStore") -> "RecordStore":
        """Merge another RecordStore into this one (mutates)."""
        if not isinstance(other, RecordStore):
            raise TypeError("Operand must be a RecordStore")
        self.records.update(other.records)
        return self

    @classmethod
    def from_file_records(cls, record_hashes: List[str], file_records: Dict[str, Any]):
        """
        Construct a RecordStore from a list of record hashes and a mapping.

        Only hashes present in the supplied mapping are included.
        """
        records = [file_records[h] for h in record_hashes if h in file_records]
        return cls(recordList=records)

    def to_mapping(self) -> Dict[str, str]:
        """
        Return a key -> path/container URI mapping for Consist log_artifacts.

        Prefers record.container_uri (or legacy uri) when present, otherwise falls back
        to file_path/repo_path.
        If short_name is missing or collides, unique_id is used for the key.
        """
        mapping: Dict[str, str] = {}
        for record in self.all_records():
            key = getattr(record, "short_name", None) or getattr(
                record, "unique_id", None
            )
            if not key:
                logger.warning("Record missing short_name and unique_id; skipping.")
                continue

            path = getattr(record, "container_uri", None) or getattr(record, "uri", None)
            if not path:
                path = getattr(record, "file_path", None) or getattr(
                    record, "repo_path", None
                )

            if not path:
                logger.warning(
                    "Record '%s' missing container_uri/uri/file_path/repo_path; skipping.",
                    key,
                )
                continue

            sanitized_key = _sanitize_artifact_key(key)
            if sanitized_key is None:
                logger.warning(
                    "Invalid artifact key '%s' could not be sanitized; using unique_id.",
                    key,
                )
                sanitized_key = getattr(record, "unique_id", None)
            elif sanitized_key != key:
                logger.warning(
                    "Invalid artifact key '%s' sanitized to '%s' for Consist compatibility.",
                    key,
                    sanitized_key,
                )
            key = sanitized_key or key

            if key in mapping:
                fallback_key = getattr(record, "unique_id", None)
                if not fallback_key or fallback_key in mapping:
                    logger.warning(
                        f"Duplicate key '{key}' with no safe fallback; skipping record."
                    )
                    continue
                logger.warning(
                    f"Duplicate key '{key}' detected; using unique_id '{fallback_key}'."
                )
                key = fallback_key

            mapping[key] = path

        return mapping
