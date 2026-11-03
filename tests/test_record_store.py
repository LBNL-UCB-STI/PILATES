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


def test_recordstore_combination_operations():
    r1 = make_file_record("id1", "file1", "/tmp/file1")
    r2 = make_file_record("id2", "file2", "/tmp/file2")
    r3 = make_file_record("id3", "file3", "/tmp/file3")

    store_a = RecordStore(recordDict={"id1": r1, "id2": r2})
    store_b = RecordStore(recordDict={"id3": r3})

    # In-place addition (+=)
    store_a += store_b
    assert store_a.records["id3"] is r3

    # New store via +
    store_c = store_a + store_b
    # store_c should contain id1, id2, id3 (id3 already present, same object)
    assert set(store_c.records.keys()) == {"id1", "id2", "id3"}
    assert store_c.records["id1"] is r1
    assert store_c.records["id2"] is r2
    assert store_c.records["id3"] is r3


def test_recordstore_invalid_initialization():
    # Passing a non-dict recordDict should raise TypeError
    with pytest.raises(TypeError):
        RecordStore(recordDict=["not", "a", "dict"])

    # The above line passes a correct Record; to trigger error we need a bad item:
    with pytest.raises(TypeError):
        RecordStore(recordList=[object()])
