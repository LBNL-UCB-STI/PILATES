"""
pilates/generic/records.py

Data structures for tracking files, model runs, and provenance within PILATES.

This module defines lightweight record types used by the provenance subsystem:
- Record: base dataclass with common fields and OpenLineage id generation.
- RecordStore: in-memory container for Record objects with convenience helpers.
- FileRecord / H5FileRecord / H5TableRecord: file- and HDF5-specific records.
- RepoRecord: repository reference record.
- ModelRunInfo: metadata for a model execution.
- OpenLineageEventMetadata: compact event metadata for OpenLineage events.
- PilatesRunInfo: top-level run metadata aggregating records and model runs.

These classes are intentionally simple and serializable; they are used to
assemble OpenLineage Dataset objects and to persist run metadata elsewhere.
"""
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any
import logging

from openlineage.client.facet import SchemaField, SchemaDatasetFacet
from openlineage.client.run import Dataset, InputDataset, OutputDataset

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Record:
    """
    Base record with common provenance fields.

    Attributes:
        unique_id: Optional stable unique identifier for the record (application-level).
        created_at: ISO timestamp string of when the record was created.
        short_name: Human-friendly short name used for OpenLineage dataset naming.
        description: Optional descriptive text.
        exists: Whether the referenced file/resource currently exists.
        openlineage_id: UUID used specifically for OpenLineage; generated if absent.
    """
    unique_id: Optional[str] = None
    created_at: Optional[str] = None
    short_name: Optional[str] = None
    description: Optional[str] = None
    exists: bool = True
    openlineage_id: Optional[str] = None

    def __post_init__(self):
        # Ensure every record has an OpenLineage UUID for event payloads.
        if self.openlineage_id is None:
            self.openlineage_id = str(uuid.uuid4())

    def __hash__(self):
        # Hash stable by unique_id to allow set/dict usage keyed by id.
        return hash(self.unique_id)


class RecordStore:
    """
    Lightweight container for Record objects.

    Intended for in-memory aggregation of FileRecord/H5FileRecord/RepoRecord objects.
    It provides simple merging and convenience accessors used during initialization
    and postprocessing.
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
        Return a new RecordStore containing records from both operands.

        Does not mutate the operands.
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
        """Return the Record with the given unique_id or None if not present."""
        return self.records.get(unique_id)

    def all_records(self) -> List[Union["FileRecord", "RepoRecord", "H5FileRecord", "H5TableRecord"]]:
        """Return a list of all Record objects currently in the store."""
        return list(self.records.values())

    def all_unique_ids(self) -> List[str]:
        """Return a list of all unique ids present in the store."""
        return list(self.records.keys())

    @classmethod
    def from_file_records(cls, record_hashes: List[str], file_records: Dict[str, Any]):
        """
        Construct a RecordStore from a list of record hashes and a mapping.

        Only hashes present in the supplied mapping are included.
        """
        records = [file_records[h] for h in record_hashes if h in file_records]
        return cls(recordList=records)


@dataclass(kw_only=True)
class FileRecord(Record):
    """
    Record describing a single file resource (CSV, JSON, H5 container, etc.).

    Attributes:
        file_path: Path to the file on disk (used for OpenLineage facets).
        models: List of model names that produced/consume this file.
        description: Optional description (overrides base.description).
        year: Optional year tag associated with the dataset.
        source_file_paths: List of original source paths used to create this file.
        metadata: Arbitrary key/value metadata dict.
        producing_run_id: ModelRunInfo.unique_id that produced this file, if known.
        consuming_run_ids: List of model run ids that consume this file.
        schema: Optional schema description used to generate OpenLineage schema facets.
    """
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

    def _create_schema(self) -> Dict[str, SchemaDatasetFacet]:
        """
        Create OpenLineage schema facet if `schema` is provided.

        Returns a mapping of facets to merge into the dataset facets.
        """
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
        Convert this FileRecord to an OpenLineage Dataset object.

        The dataset facets include filePath, description, year, metadata and an
        optional schema facet.
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
    Represents an individual table inside an H5 container.

    The `h5_file_unique_id` links back to the parent H5FileRecord, and
    `table_name` is the internal HDF5 path (e.g., '/households').
    """
    h5_file_unique_id: str  # Unique ID of the parent H5FileRecord
    table_name: str

    def __post_init__(self):
        # Preserve behavior from FileRecord and ensure an openlineage id exists.
        super().__post_init__()
        # When created without unique_id, a placeholder is assigned here.
        if self.unique_id is None:
            # A placeholder unique id; postprocessors typically overwrite based on content hash.
            self.unique_id = str(uuid.uuid4())


@dataclass(kw_only=True)
class H5FileRecord(Record):
    """
    Represents an H5 file container and references to contained tables.

    Attributes:
        file_path: Path to the H5 file.
        table_record_ids: List of unique_ids for contained H5TableRecord entries.
        Other fields are similar to FileRecord and used for provenance linkage.
    """
    file_path: str
    models: List[str] = field(default_factory=list)
    description: Optional[str] = None
    year: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    table_record_ids: List[str] = field(default_factory=list)
    source_file_paths: List[str] = field(default_factory=list)
    producing_run_id: Optional[str] = None
    consuming_run_ids: List[str] = field(default_factory=list)
    schema: Optional[List[Dict[str, str]]] = field(default_factory=list)

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
    """
    Simple record for referencing external code repositories or directories.

    Fields:
        repo_path: Filesystem path or URL of the repository.
        accessed_at: ISO timestamp of when the repo was captured/accessed.
    """
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
    """
    Represents a single model run execution and its input/output record hashes.

    Fields:
        model: Model name (e.g., 'activitysim', 'beam').
        year: Year associated with the run.
        iteration: Optional supply/demand iteration index.
        description: Optional descriptive text.
        completed_at: ISO timestamp when the run completed.
        input_record_hashes: List of unique_ids for input records.
        output_record_hashes: List of unique_ids for outputs produced by this run.
        status: Run status string (e.g., 'uninitialized', 'running', 'completed', 'failed').
    """
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
    """
    Aggregated metadata for an entire PILATES run.

    Contains file and repo records, model run metadata, configuration snapshot,
    and a lightweight list of produced OpenLineage event metadata.
    """
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
        Return the unique_id of the most recent ModelRunInfo for the given model.

        If no runs exist for the model, return None.
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
        Return the most recently created FileRecord with the given short_name.

        If multiple records share the short_name, the record with the latest
        `created_at` timestamp is returned.
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
        Return a list of file unique_ids that are outputs of the specified model run.
        """
        run_info = self.model_runs.get(model_run_hash)
        if not run_info:
            return []

        return run_info.output_record_hashes

    def get_latest_model_run_output_records(self, model_name: str) -> List[FileRecord]:
        """
        Return a list of FileRecord objects that are outputs of the latest model run.

        If no runs exist for the model an empty list is returned.
        """
        latest_run_hash = self.get_latest_model_run(model_name)
        if not latest_run_hash:
            return []

        run_outputs = self.get_run_outputs(latest_run_hash)

        return [self.file_records[h] for h in run_outputs if h in self.file_records]