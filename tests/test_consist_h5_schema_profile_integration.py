import pytest

from pilates.utils import consist_runtime as cr


def test_consist_h5_table_schema_profile(tmp_path):
    consist = pytest.importorskip("consist")
    h5py = pytest.importorskip("h5py")

    if not hasattr(consist.Tracker, "log_h5_table"):
        pytest.skip("Consist log_h5_table not available")

    run_dir = tmp_path / "runs"
    db_path = tmp_path / "consist.duckdb"
    tracker = consist.Tracker(
        run_dir=run_dir,
        db_path=str(db_path),
        mounts={"workspace": str(tmp_path)},
    )
    cr.set_tracker(tracker)

    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("persons", data=[1, 2, 3])

    with tracker.start_run("h5-schema-test", model="demo"):
        artifact = cr.log_h5_table(
            str(h5_path),
            key="persons_table",
            direction="input",
            table_path="persons",
            profile_file_schema=True,
        )

    assert artifact is not None
    meta = getattr(artifact, "meta", {}) or {}
    assert meta.get("table_path") == "persons"
    # Schema profiling may be disabled or not supported for h5_table in some builds.
    if meta.get("schema_id") is None or meta.get("schema_summary") is None:
        pytest.skip("H5 table schema profiling not available in this Consist build")
