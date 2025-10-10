import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any
import logging
import hashlib

from openlineage.client.facet import SchemaField, SchemaDatasetFacet
from openlineage.client.run import Dataset, InputDataset, OutputDataset

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Record:
    unique_id: Optional[str] = None
    created_at: Optional[str] = None
    short_name: Optional[str] = None
    description: Optional[str] = None
    exists: bool = True
    openlineage_id: Optional[str] = None

    def __post_init__(self):
        if self.openlineage_id is None:
            self.openlineage_id = str(uuid.uuid4())

    def __hash__(self):
        return hash(self.unique_id)


class RecordStore:
    """
    Base class for a record store that can be used to store and retrieve records.
    This is a token implementation. In a real implementation, this would
    interact with a database or other persistent storage.
    """

    def __init__(
        self,
        recordDict: Optional[Dict[str, Record]] = None,
        recordList: Optional[List[Record]] = None,
    ) -> None:
        if isinstance(recordDict, dict):
            # Validate that all values are Record instances
            for key, rec in recordDict.items():
                if not isinstance(rec, Record):
                    raise TypeError("All values in recordDict must be Record instances")
            self.records = recordDict
        elif recordDict is not None:
            raise TypeError("recordDict must be a dictionary.")
        else:
            self.records = {}
        if recordList is not None:
            # Accept a list of Record objects (used by tests and some preprocessors)
            for record in recordList:
                self.add_record(record)

    def __add__(self, other: "RecordStore") -> "RecordStore":
        """
        Combines the records of two RecordStore instances into a new RecordStore.
        """
        if not isinstance(other, RecordStore):
            raise TypeError("Operand must be an instance of RecordStore.")
        combined_records = {**self.records, **other.records}
        return RecordStore(recordDict=combined_records)

    def __iadd__(self, other: "RecordStore") -> "RecordStore":
        """
        Updates the current RecordStore by adding records from another RecordStore.
        """
        if not isinstance(other, RecordStore):
            raise TypeError("Operand must be an instance of RecordStore.")
        self.records.update(other.records)
        return self

    def add_record(self, record: Record) -> "RecordStore":
        if not isinstance(record, Record):
            raise TypeError("All items in recordList must be instances of Record.")
        if record.unique_id:
            self.records[record.unique_id] = record
        return self

    def remove_record_type(self, short_name: str):
        for record in self.all_records():
            if record.short_name == short_name:
                logger.info(
                    f"Removing record type {record.short_name} from record store"
                )
                del self.records[record.unique_id]

    def get_record(self, unique_id: str) -> Optional[Record]:
        return self.records.get(unique_id)

    def all_records(self) -> List[Union["FileRecord", "RepoRecord", "H5FileRecord", "H5TableRecord"]]:
        return list(self.records.values())

    def all_unique_ids(self) -> List[str]:
        return list(self.records.keys())


@dataclass(kw_only=True)
class FileRecord(Record):
    file_path: str
    models: List[str] = field(default_factory=list)
    description: Optional[str] = None
    year: Optional[int] = None
    source_file_paths: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    producing_run_id: Optional[str] = None
    consuming_run_ids: List[str] = field(default_factory=list)
    schema: Optional[List[Dict[str, str]]] = field(default_factory=list)

    def __hash__(self):
        return hash(self.unique_id)

    def _create_schema(self):
        facets = {}
        if self.schema:
            fields = [
                SchemaField(
                    name=f.get("name"),
                    type=f.get("type"),
                    description=f.get("description"),
                )
                for f in self.schema
            ]
            if fields:
                facets["schema"] = SchemaDatasetFacet(fields=fields)
        return facets

    def toDataset(self, namespace: Optional[str] = "default") -> Dataset:
        """
        Converts the FileRecord to an OpenLineage Dataset.
        """
        return Dataset(
            namespace=namespace,
            name=self.short_name or self.file_path,
            facets={
                "filePath": self.file_path,
                "description": self.description or "",
                "year": self.year,
                "metadata": self.metadata,
            }
            | self._create_schema(),
        )

    def toInputDataset(self, namespace: Optional[str] = "default") -> InputDataset:
        """
        Converts the FileRecord to an OpenLineage InputDataset.
        """
        return InputDataset(
            namespace=namespace,
            name=self.short_name or self.file_path,
            facets={
                "filePath": self.file_path,
                "description": self.description or "",
                "year": self.year,
                "metadata": self.metadata,
            }
            | self._create_schema(),
        )

    def toOutputDataset(self, namespace: Optional[str] = "default") -> OutputDataset:
        """
        Converts the FileRecord to an OpenLineage OutputDataset.
        """
        return OutputDataset(
            namespace=namespace,
            name=self.short_name or self.file_path,
            facets={
                "filePath": self.file_path,
                "description": self.description or "",
                "year": self.year,
                "metadata": self.metadata,
            }
            | self._create_schema(),
        )


@dataclass(kw_only=True)
class H5TableRecord(FileRecord):
    """
    Represents a table within an H5 file.
    Inherits from FileRecord but adds specific H5 context.
    """
    h5_file_unique_id: str  # Unique ID of the parent H5FileRecord
    table_name: str

    def __post_init__(self):
        super().__post_init__()
        # Ensure unique_id is based on table content hash, not file_path
        # This will be set during creation in the postprocessor
        if self.unique_id is None:
            # A placeholder unique_id if not explicitly provided, will be overwritten by hash
            self.unique_id = str(uuid.uuid4())


@dataclass(kw_only=True)
class H5FileRecord(Record):
    """
    Represents an H5 file container, holding references to its internal tables.
    """
    file_path: str
    models: List[str] = field(default_factory=list)
    description: Optional[str] = None
    year: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    table_record_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # Ensure unique_id is based on the hash of the H5 file content
        # This will be set during creation in the postprocessor
        if self.unique_id is None:
            # A placeholder unique_id if not explicitly provided, will be overwritten by hash
            self.unique_id = str(uuid.uuid4())

    def toDataset(self, namespace: Optional[str] = "default") -> Dataset:
        """
        Converts the H5FileRecord to an OpenLineage Dataset.
        """
        return Dataset(
            namespace=namespace,
            name=self.short_name or self.file_path,
            facets={
                "filePath": self.file_path,
                "description": self.description or "",
                "year": self.year,
                "metadata": self.metadata,
            },
        )

    def toInputDataset(self, namespace: Optional[str] = "default") -> InputDataset:
        """
        Converts the H5FileRecord to an OpenLineage InputDataset.
        """
        return InputDataset(
            namespace=namespace,
            name=self.short_name or self.file_path,
            facets={
                "filePath": self.file_path,
                "description": self.description or "",
                "year": self.year,
                "metadata": self.metadata,
            },
        )

    def toOutputDataset(self, namespace: Optional[str] = "default") -> OutputDataset:
        """
        Converts the H5FileRecord to an OpenLineage OutputDataset.
        """
        return OutputDataset(
            namespace=namespace,
            name=self.short_name or self.file_path,
            facets={
                "filePath": self.file_path,
                "description": self.description or "",
                "year": self.year,
                "metadata": self.metadata,
            },
        )


@dataclass(kw_only=True)
class RepoRecord(Record):
    repo_path: Optional[str] = None
    description: Optional[str] = None
    accessed_at: Optional[str] = None

    def __hash__(self):
        return hash(self.unique_id)

    def toDataset(self, namespace: Optional[str] = "default") -> Dataset:
        """
        Converts the FileRecord to an OpenLineage Dataset.
        """
        return Dataset(
            namespace=namespace,
            name=self.short_name or self.repo_path,
            facets={
                "filePath": self.repo_path,
                "description": self.description or "",
            },
        )

    def toInputDataset(self, namespace: Optional[str] = "default") -> InputDataset:
        """
        Converts the FileRecord to an OpenLineage InputDataset.
        """
        return InputDataset(
            namespace=namespace,
            name=self.short_name or self.repo_path,
            facets={
                "filePath": self.repo_path,
                "description": self.description or "",
            },
        )

    def toOutputDataset(self, namespace: Optional[str] = "default") -> OutputDataset:
        """
        Converts the FileRecord to an OpenLineage OutputDataset.
        """
        return OutputDataset(
            namespace=namespace,
            name=self.short_name or self.repo_path,
            facets={
                "filePath": self.repo_path,
                "description": self.description or "",
            },
        )


@dataclass(kw_only=True, unsafe_hash=True)
class ModelRunInfo(Record):
    model: str
    year: int
    iteration: Optional[int] = None
    description: Optional[str] = None
    completed_at: Optional[str] = None
    input_record_hashes: List[str] = field(default_factory=list)
    output_record_hashes: List[str] = field(default_factory=list)
    status: str = "uninitialized"


@dataclass(kw_only=True)
class OpenLineageEventMetadata:
    """Lightweight metadata for OpenLineage events without full event payload."""

    event_time: str
    event_type: str  # START, COMPLETE, FAIL
    run_uuid: str  # OpenLineage run UUID
    job_name: str  # Formatted job name with year/iteration
    model_run_id: str  # Internal PILATES model run ID


@dataclass(kw_only=True)
class PilatesRunInfo:
    run_id: str
    created_at: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    models_used: list = field(default_factory=list)
    settings_hash: Optional[str] = None
    code_version: Optional[str] = None
    hostname: Optional[str] = None
    file_records: Dict[str, Union["FileRecord", "H5FileRecord", "H5TableRecord"]] = field(default_factory=dict)
    repo_records: Dict[str, RepoRecord] = field(default_factory=dict)
    model_runs: Dict[str, ModelRunInfo] = field(default_factory=dict)
    config_snapshot: Optional[Dict[str, Any]] = None
    openlineage_event_metadata: List[OpenLineageEventMetadata] = field(
        default_factory=list
    )

    def get_latest_model_run(self, model_name: str) -> Optional[str]:
        """
        Returns the latest ModelRunInfo for the given model name, or None if not found.
        """
        runs = [
            run.unique_id
            for run in self.model_runs.values()
            if getattr(run, "model", None) == model_name
        ]
        if not runs:
            return None

        # Prefer started_at, fallback to created_at
        def get_time(run):
            iso_time = getattr(run, "created_at", None)
            if iso_time:
                return datetime.fromisoformat(iso_time).timestamp()
            else:
                logger.warning(f"Failed to determine creation time for {run}")
                return 0.0

        run_times = {run: get_time(self.model_runs[run]) for run in runs}
        logger.info(f"Looking at model runs for {model_name}: {run_times}")
        found_run = max(runs, key=lambda r: get_time(self.model_runs[r]))
        logger.info(
            f"Latest model run for {model_name} is {found_run} with time {run_times[found_run]}"
        )
        return found_run

    def get_most_recent_record(self, short_name: str) -> Optional[FileRecord]:
        """
        Returns the most recent FileRecord with the given short name.
        """
        records = [
            record
            for record in self.file_records.values()
            if record.short_name == short_name
        ]
        if not records:
            return None

        # Prefer created_at, fallback to unique_id
        def get_time(record):
            iso_time = getattr(record, "created_at", None)
            if iso_time:
                return datetime.fromisoformat(iso_time)
            else:
                logger.warning(f"Failed to determine creation time for {record}")
                return 0.0

        return max(records, key=get_time)

    def get_run_outputs(self, model_run_hash: str) -> List[str]:
        """
        Returns a list of file hashes that are outputs of the specified model run.
        """
        run_info = self.model_runs.get(model_run_hash)
        if not run_info:
            return []

        return run_info.output_record_hashes

    def get_latest_model_run_output_records(self, model_name: str) -> List[FileRecord]:
        """
        Returns a list of FileRecords that are outputs of the latest model run for the given model name.
        """
        latest_run_hash = self.get_latest_model_run(model_name)
        if not latest_run_hash:
            return []

        run_outputs = self.get_run_outputs(latest_run_hash)

        return [self.file_records[h] for h in run_outputs if h in self.file_records]