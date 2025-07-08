from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Record:
    unique_id: Optional[str] = None
    created_at: Optional[str] = None
    short_name: Optional[str] = None
    description: Optional[str] = None
    exists: bool = True


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
        self.records = recordDict or {}
        if recordList is not None:
            for record in recordList:
                if not isinstance(record, Record):
                    raise TypeError(
                        "All items in recordList must be instances of Record."
                    )
                if record.unique_id:
                    self.records[record.unique_id] = record

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

    def add_record(self, record: Record):
        if record.unique_id:
            self.records[record.unique_id] = record

    def remove_record_type(self, short_name: str):
        for record in self.records:
            if record.short_name == short_name:
                logger.info(
                    f"Removing record type {record.short_name} from record store"
                )
                del self.records[record.unique_id]

    def get_record(self, unique_id: str) -> Optional[Record]:
        return self.records.get(unique_id)

    def all_records(self) -> List[Record]:
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


@dataclass(kw_only=True)
class RepoRecord(Record):
    repo_path: Optional[str] = None
    description: Optional[str] = None
    accessed_at: Optional[str] = None


@dataclass(kw_only=True)
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
class PilatesRunInfo:
    run_id: str
    created_at: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    models_used: list = field(default_factory=list)
    settings_hash: Optional[str] = None
    code_version: Optional[str] = None
    hostname: Optional[str] = None
    file_records: Dict[str, "FileRecord"] = field(default_factory=dict)
    repo_records: Dict[str, List[RepoRecord]] = field(default_factory=dict)
    model_runs: Dict[str, ModelRunInfo] = field(default_factory=dict)

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
                return datetime.fromisoformat(iso_time)
            else:
                return 0.0

        return max(runs, key=get_time)

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
