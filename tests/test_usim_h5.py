from pathlib import Path

import h5py

from pilates.utils.usim_h5 import ensure_usim_population_year_table_aliases


def _write_population_h5(path: Path, tables: tuple[str, ...]) -> None:
    with h5py.File(path, "w") as handle:
        for table_name in tables:
            handle.create_dataset(table_name, data=[1])


def test_ensure_usim_population_year_table_aliases_creates_year_scoped_links(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "population.h5"
    _write_population_h5(h5_path, ("households", "persons", "jobs", "blocks"))

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
    _write_population_h5(h5_path, ("households", "persons"))

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
