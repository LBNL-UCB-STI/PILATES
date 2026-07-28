import os

import pytest
from consist.types import H5ChildSpec

from pilates.utils import consist_runtime as cr
from pilates.utils.coupler_helpers import log_and_set_output
from pilates.workflows.artifact_keys import (
    USIM_DATASTORE_CURRENT_H5,
    USIM_POPULATION_SOURCE_H5,
)
from pilates.workflows.stages import vehicle_ownership as vehicle_ownership_stage


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
            container_recovery_unit="parent_file",
            child_recovery_policy="descriptive_only",
        )

    assert artifact is not None
    # New Consist behavior may return (container_artifact, table_artifacts)
    if isinstance(artifact, tuple):
        container = artifact[0]
    else:
        container = artifact
    assert getattr(container, "key", None) == "usim_datastore_h5"
    assert os.path.exists(h5_path)


def test_vehicle_ownership_aliases_real_population_h5_container_with_children(
    tmp_path,
):
    """Current and population roles must share one persisted H5 container."""
    consist = pytest.importorskip("consist")
    h5py = pytest.importorskip("h5py")
    tracker = consist.Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "consist.duckdb"),
        mounts={"workspace": str(tmp_path)},
    )
    h5_path = tmp_path / "population.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_group("2023").create_dataset("households", data=[1, 2, 3])

    class _Coupler:
        def __init__(self):
            self.values = {}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def set(self, key, value):
            self.values[key] = value

        def set_from_artifact(self, key, artifact):
            self.values[key] = artifact

    coupler = _Coupler()
    cr.set_enabled(True)
    try:
        with cr.use_tracker(tracker):
            with tracker.start_run("atlas-h5-alias", model="atlas"):
                log_and_set_output(
                    key=USIM_POPULATION_SOURCE_H5,
                    path=str(h5_path),
                    description="ATLAS population source",
                    coupler=coupler,
                    h5_container=True,
                    container_recovery_unit="parent_file",
                    child_recovery_policy="descriptive_only",
                    child_specs={
                        "/2023/households": H5ChildSpec(
                            key="atlas_postprocess_usim_households_table_updated",
                            description="ATLAS-updated households",
                        )
                    },
                    child_selection="include_only",
                )
                vehicle_ownership_stage._publish_current_h5_alias(
                    coupler=coupler,
                    fallback_path=h5_path,
                )
    finally:
        cr.set_enabled(None)

    population_source = coupler.get(USIM_POPULATION_SOURCE_H5)
    current = coupler.get(USIM_DATASTORE_CURRENT_H5)

    assert current is population_source
    assert getattr(population_source, "key", None) == USIM_POPULATION_SOURCE_H5
    assert [child.key for child in tracker.get_child_artifacts(population_source)] == [
        "atlas_postprocess_usim_households_table_updated"
    ]
