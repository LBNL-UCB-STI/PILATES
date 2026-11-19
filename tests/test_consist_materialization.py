from pathlib import Path

import pytest

import consist


def test_outside_mounted_artifact_materializes_with_relative_layout(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "consist.db"
    historical_run_dir = tmp_path / "historical_run"
    outside_input_root = tmp_path / "external_inputs"
    fresh_workspace = tmp_path / "fresh_workspace"

    source_relpath = Path("beam") / "input" / "year_2025" / "config.xml"
    source_file = outside_input_root / source_relpath
    source_file.parent.mkdir(parents=True)
    source_bytes = b"<beam><source>outside-run-tree</source></beam>\n"
    source_file.write_bytes(source_bytes)

    producer = consist.Tracker(
        run_dir=historical_run_dir,
        db_path=str(db_path),
        mounts={"beam_input": str(outside_input_root)},
    )
    with producer.start_run(run_id="historical_beam_inputs", model="pilates"):
        artifact = producer.log_artifact(
            source_file,
            key="beam_input_archived",
            direction="output",
        )

    assert artifact.container_uri == "beam_input://beam/input/year_2025/config.xml"
    assert not source_file.resolve().is_relative_to(historical_run_dir.resolve())

    consumer = consist.Tracker(run_dir=fresh_workspace, db_path=str(db_path))
    historical_artifact = consumer.get_artifact(artifact.id)

    result = consumer.materialize_artifact(
        historical_artifact,
        target_root=fresh_workspace,
        preserve_existing=True,
    )

    expected_path = fresh_workspace / source_relpath
    assert result.status == "materialized_from_historical_root"
    assert result.resolvable is True
    assert result.path == expected_path.resolve()
    assert expected_path.read_bytes() == source_bytes


def test_nested_zarr_skims_output_hydrates_recursively(tmp_path: Path) -> None:
    db_path = tmp_path / "consist.db"
    historical_run_dir = tmp_path / "historical_run"
    fresh_workspace = tmp_path / "fresh_workspace"

    zarr_root = historical_run_dir / "outputs" / "zarr_skims"
    (zarr_root / "2025" / "am").mkdir(parents=True)
    (zarr_root / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    (zarr_root / "2025" / "am" / "0.0").write_bytes(b"skim-bytes")
    (zarr_root / "2025" / "am" / ".zarray").write_text(
        '{"shape": [1], "chunks": [1]}\n',
        encoding="utf-8",
    )

    producer = consist.Tracker(run_dir=historical_run_dir, db_path=str(db_path))
    with producer.start_run(run_id="historical_zarr_skims", model="pilates"):
        producer.log_output(zarr_root, key="zarr_skims")

    consumer = consist.Tracker(run_dir=fresh_workspace, db_path=str(db_path))
    hydrated = consumer.hydrate_run_outputs(
        "historical_zarr_skims",
        target_root=fresh_workspace,
        keys=["zarr_skims"],
        preserve_existing=True,
    )

    zarr_output = hydrated["zarr_skims"]
    expected_root = fresh_workspace / "outputs" / "zarr_skims"
    assert zarr_output.status == "materialized_from_filesystem"
    assert zarr_output.resolvable is True
    assert zarr_output.path == expected_root.resolve()
    assert (expected_root / ".zgroup").read_text(encoding="utf-8") == (
        '{"zarr_format": 2}\n'
    )
    assert (expected_root / "2025" / "am" / "0.0").read_bytes() == b"skim-bytes"
    assert (expected_root / "2025" / "am" / ".zarray").exists()


def test_materialize_run_outputs_by_usim_population_key_preserves_existing(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "consist.db"
    historical_run_dir = tmp_path / "historical_run"
    fresh_workspace = tmp_path / "fresh_workspace"

    population_path = historical_run_dir / "outputs" / "usim" / "population.h5"
    population_path.parent.mkdir(parents=True)
    population_path.write_bytes(b"historical-population-h5")

    producer = consist.Tracker(run_dir=historical_run_dir, db_path=str(db_path))
    with producer.start_run(run_id="historical_usim_population", model="pilates"):
        producer.log_output(population_path, key="usim_population_source_h5")

    existing_destination = fresh_workspace / "outputs" / "usim" / "population.h5"
    existing_destination.parent.mkdir(parents=True)
    existing_destination.write_bytes(b"operator-kept-local-file")

    consumer = consist.Tracker(run_dir=fresh_workspace, db_path=str(db_path))
    result = consumer.materialize_run_outputs(
        "historical_usim_population",
        target_root=fresh_workspace,
        keys=["usim_population_source_h5"],
        preserve_existing=True,
    )

    assert result.skipped_existing == ["usim_population_source_h5"]
    assert result.materialized == {}
    assert result.failed == []
    assert existing_destination.read_bytes() == b"operator-kept-local-file"


def test_materialize_run_outputs_missing_key_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "consist.db"
    historical_run_dir = tmp_path / "historical_run"
    fresh_workspace = tmp_path / "fresh_workspace"

    output_path = historical_run_dir / "outputs" / "zarr_skims" / ".zgroup"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"zarr_format": 2}\n', encoding="utf-8")

    producer = consist.Tracker(run_dir=historical_run_dir, db_path=str(db_path))
    with producer.start_run(run_id="historical_outputs", model="pilates"):
        producer.log_output(output_path, key="zarr_skims")

    consumer = consist.Tracker(run_dir=fresh_workspace, db_path=str(db_path))
    with pytest.raises(KeyError, match="missing_key"):
        consumer.materialize_run_outputs(
            "historical_outputs",
            target_root=fresh_workspace,
            keys=["missing_key"],
        )
