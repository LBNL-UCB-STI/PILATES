import os

import pytest

from pilates.utils import consist_runtime as cr


def test_consist_log_h5_container_integration(tmp_path):
    consist = pytest.importorskip("consist")
    h5py = pytest.importorskip("h5py")

    if not hasattr(consist.Tracker, "log_h5_container"):
        pytest.skip("Consist log_h5_container not available")

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

    with tracker.start_run("h5-test", model="demo"):
        artifact = cr.log_h5_container(
            str(h5_path),
            key="usim_datastore_h5",
            direction="input",
            hash_tables="never",
        )

    assert artifact is not None
    # New Consist behavior may return (container_artifact, table_artifacts)
    if isinstance(artifact, tuple):
        container = artifact[0]
    else:
        container = artifact
    assert getattr(container, "key", None) == "usim_datastore_h5"
    assert os.path.exists(h5_path)
