from types import SimpleNamespace

import pytest
import pandas as pd

from pilates.atlas.postprocessor import resolve_atlas_usim_datastore_path
from pilates.workflows.stages.vehicle_ownership import (
    _validate_population_h5_for_activitysim_year,
)


def _settings(output_template: str = "model_data_{year}.h5") -> SimpleNamespace:
    return SimpleNamespace(
        urbansim=SimpleNamespace(output_file_template=output_template),
    )


def test_validate_population_h5_for_activitysim_year_rejects_root_only_forecast_h5(
    tmp_path,
):
    h5_path = tmp_path / "model_data_2021.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/{table_name}", pd.DataFrame({"value": [1]}))

    with pytest.raises(KeyError, match="require_exact_year=True"):
        _validate_population_h5_for_activitysim_year(
            path=h5_path,
            year=2021,
            context="test",
        )


def test_validate_population_h5_for_activitysim_year_accepts_year_scoped_tables(
    tmp_path,
):
    h5_path = tmp_path / "model_data_2021.h5"
    with pd.HDFStore(h5_path, mode="w") as store:
        for table_name in ("households", "persons", "jobs", "blocks"):
            store.put(f"/2021/{table_name}", pd.DataFrame({"value": [1]}))

    _validate_population_h5_for_activitysim_year(
        path=h5_path,
        year=2021,
        context="test",
    )


def test_resolve_atlas_usim_datastore_path_accepts_artifact_like_explicit_value(
    tmp_path,
):
    usim_dir = tmp_path / "workspace" / "urbansim" / "data"
    usim_dir.mkdir(parents=True)
    explicit = usim_dir / "artifact_current.h5"
    explicit.write_text("", encoding="utf-8")

    workspace = SimpleNamespace(
        full_path=str(tmp_path / "workspace"),
        get_usim_mutable_data_dir=lambda: str(usim_dir),
    )
    state = SimpleNamespace(
        atlas_usim_datastore_h5=SimpleNamespace(
            path="urbansim/data/artifact_current.h5"
        ),
        forecast_year=2029,
        is_start_year=lambda: False,
    )

    resolved = resolve_atlas_usim_datastore_path(
        settings=_settings(output_template="model_data_{year}.h5"),
        state=state,
        workspace=workspace,
    )

    assert resolved == explicit
