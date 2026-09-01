import os
from pathlib import Path

import h5py
import pandas as pd
import pytest

from pilates.utils.usim_h5 import (
    create_usim_population_source_snapshot,
    ensure_usim_population_year_table_aliases,
    resolve_usim_population_table_paths,
)


def _write_population_store(path: Path, tables: tuple[str, ...]) -> Path:
    with pd.HDFStore(path, mode="w") as store:
        for index, table_name in enumerate(tables):
            store.put(table_name, pd.DataFrame({"marker": [index]}))
    return path


def test_ensure_usim_population_year_table_aliases_creates_year_scoped_links(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "population.h5"
    _write_population_store(h5_path, ("households", "persons", "jobs", "blocks"))

    result = ensure_usim_population_year_table_aliases(
        h5_path=str(h5_path),
        year=2023,
    )

    assert result == {
        "created": [
            "/2023/households",
            "/2023/persons",
            "/2023/jobs",
            "/2023/blocks",
        ],
        "existing": [],
        "missing_root": [],
    }

    with h5py.File(h5_path, "r") as handle:
        for table_name in ("households", "persons", "jobs", "blocks"):
            assert f"/2023/{table_name}" in handle
            assert handle[f"/2023/{table_name}"].id == handle[table_name].id


def test_ensure_usim_population_year_table_aliases_reports_missing_root_tables(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "population.h5"
    _write_population_store(h5_path, ("households", "persons"))

    result = ensure_usim_population_year_table_aliases(
        h5_path=str(h5_path),
        year=2023,
    )

    assert result["created"] == ["/2023/households", "/2023/persons"]
    assert result["existing"] == []
    assert result["missing_root"] == ["/jobs", "/blocks"]

    with h5py.File(h5_path, "r") as handle:
        assert "/2023/households" in handle
        assert "/2023/persons" in handle
        assert "/2023/jobs" not in handle
        assert "/2023/blocks" not in handle


def test_create_snapshot_adds_exact_year_links_and_resolves_them(
    tmp_path: Path,
) -> None:
    source = _write_population_store(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    destination = tmp_path / "snapshot.h5"

    result = create_usim_population_source_snapshot(
        source_h5=source, destination_h5=destination, year=2023
    )

    assert result == destination
    resolved = resolve_usim_population_table_paths(
        h5_path=str(destination), year=2023, require_exact_year=True
    )
    assert set(resolved.values()) == {
        "/2023/households",
        "/2023/persons",
        "/2023/jobs",
        "/2023/blocks",
    }
    with h5py.File(destination, "r") as handle:
        for table_name in ("households", "persons", "jobs", "blocks"):
            assert handle[f"/2023/{table_name}"].id == handle[f"/{table_name}"].id
    with h5py.File(source, "r") as handle:
        assert "/2023" not in handle


@pytest.mark.parametrize(
    "case, expected_error",
    (
        ("missing", FileNotFoundError),
        ("same", ValueError),
        ("missing_root", RuntimeError),
    ),
)
def test_create_snapshot_validation_leaves_existing_destination_unchanged(
    tmp_path: Path, case: str, expected_error: type[Exception]
) -> None:
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"
    _write_population_store(destination, ("households", "persons", "jobs", "blocks"))
    before = destination.read_bytes()
    if case == "missing":
        source = tmp_path / "missing.h5"
    elif case == "same":
        source = destination
    else:
        _write_population_store(source, ("households", "persons", "jobs"))

    with pytest.raises(expected_error):
        create_usim_population_source_snapshot(
            source_h5=source, destination_h5=destination, year=2023
        )

    assert destination.read_bytes() == before


def test_create_snapshot_rejects_existing_hard_link_destination(
    tmp_path: Path,
) -> None:
    source = _write_population_store(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    destination = tmp_path / "destination.h5"
    os.link(source, destination)

    with pytest.raises(ValueError, match="must differ from source"):
        create_usim_population_source_snapshot(
            source_h5=source, destination_h5=destination, year=2023
        )


def test_create_snapshot_preserves_existing_exact_year_table_and_links_missing_sibling(
    tmp_path: Path,
) -> None:
    source = _write_population_store(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    with pd.HDFStore(source, mode="a") as store:
        store.put("2030/households", pd.DataFrame({"marker": [99]}))
    destination = tmp_path / "snapshot.h5"

    create_usim_population_source_snapshot(
        source_h5=source, destination_h5=destination, year=2030
    )

    with pd.HDFStore(destination, mode="r") as store:
        assert store["/2030/households"]["marker"].tolist() == [99]
    with h5py.File(destination, "r") as handle:
        assert handle["/2030/persons"].id == handle["/persons"].id


def test_create_snapshot_overwrites_existing_destination(
    tmp_path: Path,
) -> None:
    source = _write_population_store(
        tmp_path / "source.h5", ("households", "persons", "jobs", "blocks")
    )
    destination = _write_population_store(
        tmp_path / "destination.h5", ("households", "persons", "jobs", "blocks")
    )
    with pd.HDFStore(source, mode="a") as store:
        store.put("households", pd.DataFrame({"marker": [42]}))

    create_usim_population_source_snapshot(
        source_h5=source, destination_h5=destination, year=2023
    )

    with pd.HDFStore(destination, mode="r") as store:
        assert store["/households"]["marker"].tolist() == [42]
