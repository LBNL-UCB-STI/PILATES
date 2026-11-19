from __future__ import annotations

import consist
from consist import CacheOptions, ExecutionOptions, Tracker


def test_scenario_run_resolves_define_step_metadata_without_proxy(tmp_path) -> None:
    tracker = Tracker(
        run_dir=tmp_path / "runs",
        db_path=str(tmp_path / "runs" / "provenance.duckdb"),
    )
    observed_runtime_kwargs: list[dict[str, object]] = []

    @consist.define_step(
        model="beam_preprocess",
        name_template="beam_preprocess_{year}",
        tags=lambda ctx: [f"stage:{ctx.stage}"],
    )
    def _dummy_step(token: object = None) -> dict[str, object]:
        observed_runtime_kwargs.append({"token": token})
        return {}

    with tracker.scenario("scenario_direct", model="pilates_workflow") as scenario:
        result = scenario.run(
            fn=_dummy_step,
            year=2020,
            stage="initialize_data",
            execution_options=ExecutionOptions(
                runtime_kwargs={"token": "identity-token"}
            ),
            cache_options=CacheOptions(cache_mode="overwrite"),
        )

    assert result.run.model_name == "beam_preprocess"
    assert "beam_preprocess_2020" in result.run.id
    assert "stage:initialize_data" in (result.run.tags or [])
    assert observed_runtime_kwargs == [{"token": "identity-token"}]
