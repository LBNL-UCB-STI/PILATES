import pytest
from pilates.generic.records import RecordStore, FileRecord


def make_file_record(uid: str, short_name: str, path: str) -> FileRecord:
    return FileRecord(
        unique_id=uid,
        short_name=short_name,
        file_path=path,
        description="test file",
    )


def test_recordstore_add_and_get():
    r1 = make_file_record("id1", "file1", "/tmp/file1")
    r2 = make_file_record("id2", "file2", "/tmp/file2")

    store = RecordStore()
    store.add_record(r1)
    store.add_record(r2)

    # Retrieval by unique_id
    assert store.get_record("id1") is r1
    assert store.get_record("id2") is r2

    # All records and ids
    all_ids = store.all_unique_ids()
    assert set(all_ids) == {"id1", "id2"}
    all_records = store.all_records()
    assert set(all_records) == {r1, r2}


def test_recordstore_combination_operations():
    r1 = make_file_record("id1", "file1", "/tmp/file1")
    r2 = make_file_record("id2", "file2", "/tmp/file2")
    r3 = make_file_record("id3", "file3", "/tmp/file3")

    store_a = RecordStore(recordDict={"id1": r1, "id2": r2})
    store_b = RecordStore(recordDict={"id3": r3})

    # In-place addition (+=)
    store_a += store_b
    assert store_a.get_record("id3") is r3

    # New store via +
    store_c = store_a + store_b
    # store_c should contain id1, id2, id3 (id3 already present, same object)
    assert set(store_c.all_unique_ids()) == {"id1", "id2", "id3"}
    assert store_c.get_record("id1") is r1
    assert store_c.get_record("id2") is r2
    assert store_c.get_record("id3") is r3


def test_recordstore_remove_record_type():
    r_file = make_file_record("fid", "common", "/tmp/file")
    r_repo = make_file_record("rid", "common", "/tmp/file2")

    store = RecordStore(recordDict={"fid": r_file, "rid": r_repo})

    # Remove all records with short_name "common"
    store.remove_record_type("common")

    # After removal, the store should be empty
    assert store.all_unique_ids() == []
    assert store.all_records() == []


def test_recordstore_invalid_initialization():
    # Passing a non-dict recordDict should raise TypeError
    with pytest.raises(TypeError):
        RecordStore(recordDict=["not", "a", "dict"])

    # The above line passes a correct Record; to trigger error we need a bad item:
    with pytest.raises(TypeError):
        RecordStore(recordList=[object()])
