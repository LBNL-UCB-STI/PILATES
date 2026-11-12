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
from typing import Any, Dict, Iterator, List, Optional, Sequence

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

    Used internally by model preprocessors/runners/postprocessors.
    Records are keyed by unique_id for lookup and deduplication.
    """

    records: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, recordList: Optional[Sequence[Any]] = None) -> None:
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

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.records.values())

    def to_mapping(self) -> Dict[str, str]:
        """Return a ``short_name`` (or ``unique_id``) -> ``file_path`` mapping."""
        mapping: Dict[str, str] = {}
        for record in self.all_records():
            key = record.short_name or record.unique_id
            if not key or not record.file_path:
                continue
            mapping[key] = record.file_path
        return mapping
