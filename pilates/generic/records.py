from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union


class Record:
    unique_id: str
    created_at: Optional[str] = None
    short_name: Optional[str] = None
    description: Optional[str] = None

class RecordStore:
    """
    Base class for a record store that can be used to store and retrieve records.
    This is a token implementation. In a real implementation, this would
    interact with a database or other persistent storage.
    """

    def __init__(self, recordDict: Optional[Dict[str, Record]] = None, recordList: Optional[List[Record]]=None) -> None:
        self.records = recordDict or {}
        if recordList is not None:
            for record in recordList:
                if not isinstance(record, Record):
                    raise TypeError("All items in recordList must be instances of Record.")
                self.records[record.unique_id] = record

    def __add__(self, other: 'RecordStore') -> 'RecordStore':
        """
        Combines the records of two RecordStore instances into a new RecordStore.
        """
        if not isinstance(other, RecordStore):
            raise TypeError("Operand must be an instance of RecordStore.")
        combined_records = {**self.records, **other.records}
        return RecordStore(recordDict=combined_records)

    def __iadd__(self, other: 'RecordStore') -> 'RecordStore':
        """
        Updates the current RecordStore by adding records from another RecordStore.
        """
        if not isinstance(other, RecordStore):
            raise TypeError("Operand must be an instance of RecordStore.")
        self.records.update(other.records)
        return self

    def add_record(self, record: Record):
        self.records[record.unique_id] = record

    def get_record(self, unique_id: str) -> Optional[Record]:
        return self.records.get(unique_id)

    def all_records(self) -> List[Record]:
        return list(self.records.values())

@dataclass(kw_only=True)
class InputRecord(Record):
    file_path: str
    source_run_id: Optional[str] = None
    input_type: str = "unknown"
    file_hash: Optional[str] = None
    source_file_paths: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class OutputRecord:
    file_path: str
    output_type: Optional[str] = None
    model_run_id: Optional[str] = None
    created_at: Optional[str] = None
    year: Optional[int] = None
    source_file_paths: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class RepoRecord:
    repo_path: str
    description: Optional[str] = None
    accessed_at: Optional[str] = None


@dataclass(kw_only=True)
class ModelRunInfo:
    model: str
    year: int
    iteration: Optional[int] = None
    description: Optional[str] = None
    started_at: Optional[str] = None
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
    inputs: Dict[str, Union[Dict[str, List[InputRecord]], Dict[str, List[RepoRecord]]]]  = field(default_factory=lambda: {"files": {}, "repos": {}})
    outputs: dict = field(default_factory=dict)
    model_runs: Dict[str, ModelRunInfo] = field(default_factory=dict)

