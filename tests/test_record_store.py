import pytest
from pilates.generic.records import RecordStore, FileRecord


def make_file_record(uid: str, short_name: str, path: str) -> FileRecord:
    return FileRecord(
        unique_id=uid,
        short_name=short_name,
        file_path=path,
        description="test file",
    )


def test_recordstore_add_and_all_records():
    r1 = make_file_record("id1", "file1", "/tmp/file1")
    r2 = make_file_record("id2", "file2", "/tmp/file2")

    store = RecordStore()
    store.add_record(r1)
    store.add_record(r2)

    assert store.records["id1"] is r1
    assert store.records["id2"] is r2
    assert set(store.all_records()) == {r1, r2}
    assert store.to_mapping() == {"file1": "/tmp/file1", "file2": "/tmp/file2"}


def test_recordstore_invalid_initialization():
    with pytest.raises(TypeError):
        RecordStore(recordList=[object()])
