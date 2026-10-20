from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

from pilates.beam.beam_input_staging import (
    BeamDataHelper,
    copy_plans_from_asim,
    copy_with_compression_asim_file_to_beam,
    merge_replanned_asim_files,
)
from pilates.generic.records import FileRecord, RecordStore


def _create_minimal_plans_df(
    person_ids: List[int],
    plan_indices: Optional[List[int]] = None,
    scores: Optional[List[float]] = None,
    selected: Optional[List[bool]] = None,
    use_camel_case: bool = False,
) -> pd.DataFrame:
    rows = []
    n = len(person_ids)
    p_indices = plan_indices if plan_indices is not None else [0] * n
    p_scores = scores if scores is not None else [np.nan] * n
    p_selected = selected if selected is not None else [True] * n

    for i in range(n):
        pid = person_ids[i]
        p_idx = p_indices[i]
        sc = p_scores[i]
        sel = p_selected[i]
        if use_camel_case:
            rows.append(
                {
                    "personId": pid,
                    "planIndex": p_idx,
                    "planElementIndex": 1,
                    "planElementType": "activity",
                    "activityType": "Home",
                    "activityLocationX": 100.0,
                    "activityLocationY": 200.0,
                    "activityEndTime": 28800.0,  # 8 hours in seconds
                    "legMode": "car",
                    "legDepartureTime": 28800.0,
                    "legTravelTime": 1200.0,
                    "planScore": sc,
                    "planSelected": sel,
                    "tripId": 10,
                    "tourId": 20,
                }
            )
        else:
            rows.append(
                {
                    "person_id": pid,
                    "planindex": p_idx,
                    "PlanElementIndex": 1,
                    "ActivityElement": "activity",
                    "ActivityType": "Home",
                    "x": 100.0,
                    "y": 200.0,
                    "departure_time": 8.0,
                    "trip_mode": "car",
                    "legDepartureTime": 28800.0,
                    "legTravelTime": 1200.0,
                    "planscore": sc,
                    "planselected": sel,
                    "trip_id": 10,
                    "tour_id": 20,
                }
            )
    return pd.DataFrame(rows)


def _dummy_format_records(
    file_stem: str,
    file_path: str,
    file_format: str,
    description_prefix: str,
) -> List[FileRecord]:
    return [
        FileRecord(
            file_path=file_path,
            short_name=file_stem,
            description=f"{description_prefix}: {file_stem}",
        )
    ]


def test_beam_data_helper_format_autodetection(tmp_path: Path) -> None:
    """Format is auto-detected from suffix even if file_format param differs."""
    # 1. Gzipped CSV with file_format='parquet'
    csv_gz_path = tmp_path / "plans.csv.gz"
    df_raw = _create_minimal_plans_df([1, 2], use_camel_case=True)
    df_raw.to_csv(csv_gz_path, index=False, compression="gzip")

    df_cleaned = BeamDataHelper.read_and_clean(
        str(csv_gz_path), "plans", file_format="parquet"
    )
    assert len(df_cleaned) == 2
    assert "person_id" in df_cleaned.columns
    assert "departure_time" in df_cleaned.columns
    assert df_cleaned.loc[0, "departure_time"] == pytest.approx(8.0)

    # 2. Parquet file with file_format='csv'
    pq_path = tmp_path / "plans.parquet"
    df_raw = _create_minimal_plans_df([3, 4], use_camel_case=False)
    df_raw.to_parquet(pq_path, index=False)

    df_cleaned_pq = BeamDataHelper.read_and_clean(
        str(pq_path), "plans", file_format="csv"
    )
    assert len(df_cleaned_pq) == 2
    assert "person_id" in df_cleaned_pq.columns
    assert list(df_cleaned_pq["person_id"]) == [3, 4]


def test_beam_data_helper_unit_conversion_and_renames(tmp_path: Path) -> None:
    """activityEndTime (seconds) converts to departure_time (hours) and camelCase renames."""
    csv_path = tmp_path / "plans_camel.csv"
    df_raw = _create_minimal_plans_df([101], use_camel_case=True)
    df_raw.to_csv(csv_path, index=False)

    df_cleaned = BeamDataHelper.read_and_clean(
        str(csv_path), "plans", file_format="csv"
    )
    assert "activityEndTime" not in df_cleaned.columns
    assert "departure_time" in df_cleaned.columns
    # 28800 seconds / 3600 = 8.0 hours
    assert df_cleaned.loc[0, "departure_time"] == pytest.approx(8.0)
    assert df_cleaned.loc[0, "person_id"] == 101
    assert df_cleaned.loc[0, "ActivityElement"] == "activity"
    assert df_cleaned.loc[0, "ActivityType"] == "Home"
    assert df_cleaned.loc[0, "trip_mode"] == "car"
    assert df_cleaned.loc[0, "x"] == 100.0
    assert df_cleaned.loc[0, "y"] == 200.0
    assert df_cleaned.loc[0, "legDepartureTime"] == 28800.0
    assert df_cleaned.loc[0, "legTravelTime"] == 1200.0
    assert "legtraveltime" not in df_cleaned.columns


def test_stage1_merge_assigns_planindex_and_nan_score(tmp_path: Path) -> None:
    """New plans get planindex = max(old) + 1 and planscore = np.nan."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    # Prior plans: Person 1 has plans 0 and 1; Person 2 has plan 0
    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[10.0]),
            _create_minimal_plans_df([1], [1], scores=[15.0]),
            _create_minimal_plans_df([2], [0], scores=[5.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "plans.parquet"
    prior_plans.to_parquet(prior_plans_path, index=False)

    # Prior persons & households
    prior_per = pd.DataFrame(
        {"household_id": [1, 2]}, index=pd.Index([1, 2], name="person_id")
    )
    prior_per.to_parquet(scenario_dir / "persons.parquet", index=True)
    prior_hh = pd.DataFrame(
        {"cars": [1, 2]}, index=pd.Index([1, 2], name="household_id")
    )
    prior_hh.to_parquet(scenario_dir / "households.parquet", index=True)

    # ActivitySim produces candidate plan for Person 1, Person 2, and new Person 3
    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0]),
            _create_minimal_plans_df([2], [0]),
            _create_minimal_plans_df([3], [0]),
        ],
        ignore_index=True,
    )
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)

    new_per = pd.DataFrame(
        {"household_id": [1, 2, 3]}, index=pd.Index([1, 2, 3], name="person_id")
    )
    new_per_path = asim_dir / "persons.parquet"
    new_per.to_parquet(new_per_path, index=True)

    new_hh = pd.DataFrame(
        {"cars": [1, 2, 1]}, index=pd.Index([1, 2, 3], name="household_id")
    )
    new_hh_path = asim_dir / "households.parquet"
    new_hh.to_parquet(new_hh_path, index=True)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(new_per_path), None),
        "households": (str(new_hh_path), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_parquet(prior_plans_path)
    p1_plans = merged_plans[merged_plans["person_id"] == 1]
    p2_plans = merged_plans[merged_plans["person_id"] == 2]
    p3_plans = merged_plans[merged_plans["person_id"] == 3]

    assert len(p1_plans) == 3
    assert list(p1_plans["planindex"]) == [0, 1, 2]
    # New plan for p1 (planindex 2) must have planscore NaN and be selected
    p1_new = p1_plans[p1_plans["planindex"] == 2].iloc[0]
    assert np.isnan(p1_new["planscore"])
    assert p1_new["planselected"] is True or p1_new["planselected"] == 1

    assert len(p2_plans) == 2
    assert list(p2_plans["planindex"]) == [0, 1]
    p2_new = p2_plans[p2_plans["planindex"] == 1].iloc[0]
    assert np.isnan(p2_new["planscore"])
    assert p2_new["planselected"] is True or p2_new["planselected"] == 1

    assert len(p3_plans) == 1
    assert list(p3_plans["planindex"]) == [0]
    assert np.isnan(p3_plans.iloc[0]["planscore"])


def test_stage2_pruning_worst_plan_for_removal(tmp_path: Path) -> None:
    """WorstPlanForRemoval prunes NaN scores first, then lowest scores, keeping new plan."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    # Person 1 has 4 historical plans:
    # Plan 0: score = -50.0
    # Plan 1: score = 20.0
    # Plan 2: score = np.nan (unexecuted historical plan)
    # Plan 3: score = 5.0
    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[-50.0]),
            _create_minimal_plans_df([1], [1], scores=[20.0]),
            _create_minimal_plans_df([1], [2], scores=[np.nan]),
            _create_minimal_plans_df([1], [3], scores=[5.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "plans.parquet"
    prior_plans.to_parquet(prior_plans_path, index=False)

    prior_per = pd.DataFrame(
        {"household_id": [1]}, index=pd.Index([1], name="person_id")
    )
    prior_per.to_parquet(scenario_dir / "persons.parquet", index=True)
    prior_hh = pd.DataFrame({"cars": [1]}, index=pd.Index([1], name="household_id"))
    prior_hh.to_parquet(scenario_dir / "households.parquet", index=True)

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = _create_minimal_plans_df([1], [0])  # new candidate plan
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(scenario_dir / "persons.parquet"), None),
        "households": (str(scenario_dir / "households.parquet"), None),
    }

    # max_plans_memory = 3: Person 1 currently has 4 hist + 1 new = 5 plans.
    # Must prune 2 historical plans:
    # 1st to prune: Plan 2 (NaN -> -inf)
    # 2nd to prune: Plan 0 (score -50.0)
    # Surviving historical: Plan 1 (20.0) and Plan 3 (5.0)
    # Plus new plan. Total = 3 plans.
    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=3, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_parquet(prior_plans_path)
    assert len(merged_plans) == 3
    # Surviving plans must be contiguously reindexed 0, 1, 2
    assert list(merged_plans["planindex"]) == [0, 1, 2]
    # Scores of surviving plans: 20.0, 5.0, and NaN (the new plan)
    scores = list(merged_plans["planscore"])
    assert scores[0] == pytest.approx(20.0)
    assert scores[1] == pytest.approx(5.0)
    assert np.isnan(scores[2])


def test_stage2_pruning_tie_breaking_older_first(tmp_path: Path) -> None:
    """Ties in removal score prune older planindex first."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    # Person 1 has 2 historical plans with equal score 10.0
    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[10.0]),
            _create_minimal_plans_df([1], [1], scores=[10.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "plans.parquet"
    prior_plans.to_parquet(prior_plans_path, index=False)

    prior_per = pd.DataFrame(
        {"household_id": [1]}, index=pd.Index([1], name="person_id")
    )
    prior_per.to_parquet(scenario_dir / "persons.parquet", index=True)
    prior_hh = pd.DataFrame({"cars": [1]}, index=pd.Index([1], name="household_id"))
    prior_hh.to_parquet(scenario_dir / "households.parquet", index=True)

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = _create_minimal_plans_df([1], [0])
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(scenario_dir / "persons.parquet"), None),
        "households": (str(scenario_dir / "households.parquet"), None),
    }

    # max_plans_memory = 2: 2 hist + 1 new = 3. Prune 1 hist plan.
    # Between plan 0 and plan 1 (both score 10.0), plan 0 is older and pruned.
    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=2, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_parquet(prior_plans_path)
    assert len(merged_plans) == 2
    assert list(merged_plans["planindex"]) == [0, 1]
    assert merged_plans.loc[0, "planscore"] == pytest.approx(10.0)
    assert np.isnan(merged_plans.loc[1, "planscore"])


def test_stage3_policy_exp_beta_and_score_persistence(tmp_path: Path) -> None:
    """ExpBeta selection uses transient prior without modifying disk NaN score."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    # Person 1: historical plan score = -100.0 (very bad, should switch to new plan with prior 0.0)
    # Person 2: historical plan score = +100.0 (very good, should keep old plan)
    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[-100.0]),
            _create_minimal_plans_df([2], [0], scores=[100.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "plans.parquet"
    prior_plans.to_parquet(prior_plans_path, index=False)

    prior_per = pd.DataFrame(
        {"household_id": [1, 2]}, index=pd.Index([1, 2], name="person_id")
    )
    prior_per.to_parquet(scenario_dir / "persons.parquet", index=True)
    prior_hh = pd.DataFrame(
        {"cars": [1, 2]}, index=pd.Index([1, 2], name="household_id")
    )
    prior_hh.to_parquet(scenario_dir / "households.parquet", index=True)

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0]),
            _create_minimal_plans_df([2], [0]),
        ],
        ignore_index=True,
    )
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(scenario_dir / "persons.parquet"), None),
        "households": (str(scenario_dir / "households.parquet"), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5,
                plan_memory_beta=1.0,
                new_plan_selection_policy="exp_beta",
                new_plan_initial_score=0.0,
                plan_selection_random_seed=42,
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_parquet(prior_plans_path)
    p1_plans = merged_plans[merged_plans["person_id"] == 1]
    p2_plans = merged_plans[merged_plans["person_id"] == 2]

    # Person 1: new plan (index 1) should be selected because -100 << 0
    p1_selected = p1_plans[p1_plans["planselected"] == True]
    assert len(p1_selected) == 1
    assert p1_selected.iloc[0]["planindex"] == 1
    # Crucially, its planscore on disk must remain NaN!
    assert np.isnan(p1_selected.iloc[0]["planscore"])

    # Person 2: historical plan (index 0) should be selected because 100 >> 0
    p2_selected = p2_plans[p2_plans["planselected"] == True]
    assert len(p2_selected) == 1
    assert p2_selected.iloc[0]["planindex"] == 0
    assert p2_selected.iloc[0]["planscore"] == pytest.approx(100.0)

    # For person 2, the unexecuted new plan (index 1) must still have planscore = NaN
    p2_unselected = p2_plans[p2_plans["planindex"] == 1].iloc[0]
    assert np.isnan(p2_unselected["planscore"])


def test_stage3_policy_probabilistic_new(tmp_path: Path) -> None:
    """Probabilistic new forces new plan with adoption rate alpha, else uses exp_beta."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[50.0]),
            _create_minimal_plans_df([2], [0], scores=[50.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "plans.parquet"
    prior_plans.to_parquet(prior_plans_path, index=False)

    prior_per = pd.DataFrame(
        {"household_id": [1, 2]}, index=pd.Index([1, 2], name="person_id")
    )
    prior_per.to_parquet(scenario_dir / "persons.parquet", index=True)
    prior_hh = pd.DataFrame(
        {"cars": [1, 2]}, index=pd.Index([1, 2], name="household_id")
    )
    prior_hh.to_parquet(scenario_dir / "households.parquet", index=True)

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0]),
            _create_minimal_plans_df([2], [0]),
        ],
        ignore_index=True,
    )
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(scenario_dir / "persons.parquet"), None),
        "households": (str(scenario_dir / "households.parquet"), None),
    }

    # With adoption_rate = 1.0, forces new plan for all replanned agents despite old score 50.0 > 0.0
    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5,
                new_plan_selection_policy="probabilistic_new",
                new_plan_adoption_rate=1.0,
                new_plan_initial_score=0.0,
                plan_selection_random_seed=42,
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_parquet(prior_plans_path)
    for pid in [1, 2]:
        p_plans = merged_plans[merged_plans["person_id"] == pid]
        p_sel = p_plans[p_plans["planselected"] == True].iloc[0]
        # New plan is planindex 1
        assert p_sel["planindex"] == 1


def test_extinct_agents_dropped_in_cross_year_merging(tmp_path: Path) -> None:
    """Extinct agents from prior year BEAM plans are purged on cross-year merge."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    # Prior year plans contain Person 1 and Person 2
    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[10.0]),
            _create_minimal_plans_df([2], [0], scores=[20.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "beam_plans_out.csv.gz"
    prior_plans.to_csv(prior_plans_path, index=False, compression="gzip")

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    # New year population: Person 1 and Person 3 (Person 2 departed/extinct)
    new_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0]),
            _create_minimal_plans_df([3], [0]),
        ],
        ignore_index=True,
    )
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)

    new_per = pd.DataFrame(
        {"household_id": [1, 3]}, index=pd.Index([1, 3], name="person_id")
    )
    new_per_path = asim_dir / "persons.parquet"
    new_per.to_parquet(new_per_path, index=True)

    new_hh = pd.DataFrame({"cars": [1, 1]}, index=pd.Index([1, 3], name="household_id"))
    new_hh_path = asim_dir / "households.parquet"
    new_hh.to_parquet(new_hh_path, index=True)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(new_per_path), None),
        "households": (str(new_hh_path), None),
        "beam_plans_out": (str(prior_plans_path), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(max_plans_memory=5, discard_plans_every_year=False),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2025, current_inner_iter=0),
    )

    out_plans = pd.read_parquet(scenario_dir / "plans.parquet")
    # Person 2 must NOT be in output
    assert 2 not in list(out_plans["person_id"])
    # Person 1 has 2 plans (prior plan 0 + new plan 1)
    p1_plans = out_plans[out_plans["person_id"] == 1]
    assert len(p1_plans) == 2
    # Person 3 has 1 plan (index 0)
    p3_plans = out_plans[out_plans["person_id"] == 3]
    assert len(p3_plans) == 1


def test_copy_plans_from_asim_routing(tmp_path: Path) -> None:
    """copy_plans_from_asim routes to merge vs initial copy under correct conditions."""
    input_records = RecordStore()
    input_records.add_record(
        FileRecord(file_path="hh.parquet", short_name="households")
    )
    input_records.add_record(FileRecord(file_path="per.parquet", short_name="persons"))
    input_records.add_record(
        FileRecord(file_path="plans.parquet", short_name="beam_plans")
    )

    # Create dummy files
    for name in ["hh.parquet", "per.parquet", "plans.parquet"]:
        (tmp_path / name).touch()

    called_functions = []

    def mock_copy_initial(*args: Any, **kwargs: Any) -> List[FileRecord]:
        called_functions.append("initial")
        return []

    def mock_merge(*args: Any, **kwargs: Any) -> List[FileRecord]:
        called_functions.append("merge")
        return []

    base_settings = SimpleNamespace(
        activitysim=SimpleNamespace(file_format="parquet", replan_iters=1),
        beam=SimpleNamespace(discard_plans_every_year=False),
    )
    workspace = SimpleNamespace(full_path=str(tmp_path))

    # Case 1: current_inner_iter = 0, no beam_plans_out -> initial copy
    called_functions.clear()
    copy_plans_from_asim(
        input_records=input_records,
        workspace=workspace,
        state=SimpleNamespace(current_inner_iter=0, full_settings=base_settings),
        settings=base_settings,
        required_input_data=["households", "persons", "beam_plans"],
        copy_initial_asim_files_fn=mock_copy_initial,
        merge_replanned_asim_files_fn=mock_merge,
    )
    assert called_functions == ["initial"]

    # Case 2: current_inner_iter = 1, replan_iters = 1 -> merge
    called_functions.clear()
    copy_plans_from_asim(
        input_records=input_records,
        workspace=workspace,
        state=SimpleNamespace(current_inner_iter=1, full_settings=base_settings),
        settings=base_settings,
        required_input_data=["households", "persons", "beam_plans"],
        copy_initial_asim_files_fn=mock_copy_initial,
        merge_replanned_asim_files_fn=mock_merge,
    )
    assert called_functions == ["merge"]

    # Case 3: current_inner_iter = 1, replan_iters = 0 -> initial copy
    called_functions.clear()
    no_replan_settings = SimpleNamespace(
        activitysim=SimpleNamespace(file_format="parquet", replan_iters=0),
        beam=SimpleNamespace(discard_plans_every_year=True),
    )
    copy_plans_from_asim(
        input_records=input_records,
        workspace=workspace,
        state=SimpleNamespace(current_inner_iter=1, full_settings=no_replan_settings),
        settings=no_replan_settings,
        required_input_data=["households", "persons", "beam_plans"],
        copy_initial_asim_files_fn=mock_copy_initial,
        merge_replanned_asim_files_fn=mock_merge,
    )
    assert called_functions == ["initial"]

    # Case 4: current_inner_iter = 0, discard_plans_every_year = False, beam_plans_out present -> merge
    records_with_beam_out = RecordStore()
    records_with_beam_out.add_record(
        FileRecord(file_path="hh.parquet", short_name="households")
    )
    records_with_beam_out.add_record(
        FileRecord(file_path="per.parquet", short_name="persons")
    )
    records_with_beam_out.add_record(
        FileRecord(file_path="plans.parquet", short_name="beam_plans")
    )
    records_with_beam_out.add_record(
        FileRecord(file_path="out.csv.gz", short_name="beam_plans_out")
    )
    (tmp_path / "out.csv.gz").touch()

    called_functions.clear()
    copy_plans_from_asim(
        input_records=records_with_beam_out,
        workspace=workspace,
        state=SimpleNamespace(current_inner_iter=0, full_settings=base_settings),
        settings=base_settings,
        required_input_data=["households", "persons", "beam_plans", "beam_plans_out"],
        copy_initial_asim_files_fn=mock_copy_initial,
        merge_replanned_asim_files_fn=mock_merge,
    )
    assert called_functions == ["merge"]


def test_merge_replanned_asim_files_csv_format_preserves_id_columns(
    tmp_path: Path,
) -> None:
    """CSV output must retain person_id in persons.csv.gz and household_id in households.csv.gz."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    prior_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0], scores=[10.0]),
            _create_minimal_plans_df([2], [0], scores=[5.0]),
        ],
        ignore_index=True,
    )
    prior_plans_path = scenario_dir / "plans.csv.gz"
    prior_plans.to_csv(prior_plans_path, index=False, compression="gzip")

    prior_per = pd.DataFrame({"household_id": [1, 2], "person_id": [1, 2]})
    prior_per.to_csv(scenario_dir / "persons.csv.gz", index=False, compression="gzip")
    prior_hh = pd.DataFrame({"cars": [1, 2], "household_id": [1, 2]})
    prior_hh.to_csv(scenario_dir / "households.csv.gz", index=False, compression="gzip")

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = pd.concat(
        [
            _create_minimal_plans_df([1], [0]),
            _create_minimal_plans_df([2], [0]),
        ],
        ignore_index=True,
    )
    new_plans_path = asim_dir / "beam_plans.csv.gz"
    new_plans.to_csv(new_plans_path, index=False, compression="gzip")

    new_per = pd.DataFrame({"household_id": [1, 2], "person_id": [1, 2]})
    new_per_path = asim_dir / "persons.csv.gz"
    new_per.to_csv(new_per_path, index=False, compression="gzip")

    new_hh = pd.DataFrame({"cars": [2, 1], "household_id": [1, 2]})
    new_hh_path = asim_dir / "households.csv.gz"
    new_hh.to_csv(new_hh_path, index=False, compression="gzip")

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(new_per_path), None),
        "households": (str(new_hh_path), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="csv",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    # Read the raw CSVs directly with pd.read_csv to verify columns exist on disk
    raw_per = pd.read_csv(scenario_dir / "persons.csv.gz", compression="gzip")
    assert "person_id" in raw_per.columns
    assert sorted(raw_per["person_id"].tolist()) == [1, 2]

    raw_hh = pd.read_csv(scenario_dir / "households.csv.gz", compression="gzip")
    assert "household_id" in raw_hh.columns
    assert sorted(raw_hh["household_id"].tolist()) == [1, 2]

    raw_plans = pd.read_csv(scenario_dir / "plans.csv.gz", compression="gzip")
    assert "person_id" in raw_plans.columns
    assert "planindex" in raw_plans.columns

    # Verify clean round-trip with BeamDataHelper.read_and_clean
    cleaned_per = BeamDataHelper.read_and_clean(
        str(scenario_dir / "persons.csv.gz"), "persons", "csv"
    )
    assert cleaned_per.index.name == "person_id"
    assert sorted(cleaned_per.index.tolist()) == [1, 2]

    cleaned_hh = BeamDataHelper.read_and_clean(
        str(scenario_dir / "households.csv.gz"), "households", "csv"
    )
    assert cleaned_hh.index.name == "household_id"
    assert sorted(cleaned_hh.index.tolist()) == [1, 2]


def test_merge_replanned_asim_files_parquet_no_index_level(tmp_path: Path) -> None:
    """Parquet outputs must not write __index_level_0__ into files."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()

    prior_plans = _create_minimal_plans_df([1], [0], scores=[10.0])
    prior_plans.to_parquet(scenario_dir / "plans.parquet", index=False)
    prior_per = pd.DataFrame({"household_id": [1], "person_id": [1]})
    prior_per.to_parquet(scenario_dir / "persons.parquet", index=False)
    prior_hh = pd.DataFrame({"cars": [1], "household_id": [1]})
    prior_hh.to_parquet(scenario_dir / "households.parquet", index=False)

    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()
    new_plans = _create_minimal_plans_df([1], [0])
    new_plans_path = asim_dir / "beam_plans.parquet"
    new_plans.to_parquet(new_plans_path, index=False)
    new_per_path = asim_dir / "persons.parquet"
    prior_per.to_parquet(new_per_path, index=False)
    new_hh_path = asim_dir / "households.parquet"
    prior_hh.to_parquet(new_hh_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(new_per_path), None),
        "households": (str(new_hh_path), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="parquet",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    for stem in ["plans", "persons", "households"]:
        df = pd.read_parquet(scenario_dir / f"{stem}.parquet")
        assert "__index_level_0__" not in df.columns, (
            f"{stem}.parquet has __index_level_0__"
        )


def test_beam_data_helper_lowercase_leg_renames(tmp_path: Path) -> None:
    """All lowercase leg* fields are normalized to camelCase without duplicate columns or NaN overwrites."""
    csv_path = tmp_path / "plans_lower.csv"
    df_raw = pd.DataFrame(
        [
            {
                "personid": 10,
                "planindex": 0,
                "planelementtype": "leg",
                "planelementindex": 1,
                "activitytype": "Work",
                "activitylocationx": 100.0,
                "activitylocationy": 200.0,
                "legmode": "walk",
                "legdeparturetime": 3600.0,
                "legtraveltime": 600.0,
                "legroutetype": "walk_route",
                "legroutestartlink": "link_1",
                "legrouteendlink": "link_2",
                "legroutetraveltime": 600.0,
                "legroutedistance": 1500.0,
                "legroutelinks": "1,2",
                "legvehicleids": "veh_1",
                "planscore": 2.5,
                "planselected": True,
            }
        ]
    )
    df_raw.to_csv(csv_path, index=False)

    df_cleaned = BeamDataHelper.read_and_clean(
        str(csv_path), "plans", file_format="csv"
    )

    assert df_cleaned.loc[0, "person_id"] == 10
    assert df_cleaned.loc[0, "ActivityElement"] == "leg"
    assert df_cleaned.loc[0, "PlanElementIndex"] == 1
    assert df_cleaned.loc[0, "ActivityType"] == "Work"
    assert df_cleaned.loc[0, "trip_mode"] == "walk"
    assert df_cleaned.loc[0, "x"] == 100.0
    assert df_cleaned.loc[0, "y"] == 200.0
    assert df_cleaned.loc[0, "legDepartureTime"] == 3600.0
    assert df_cleaned.loc[0, "legTravelTime"] == 600.0
    assert df_cleaned.loc[0, "legRouteType"] == "walk_route"
    assert df_cleaned.loc[0, "legRouteStartLink"] == "link_1"
    assert df_cleaned.loc[0, "legRouteEndLink"] == "link_2"
    assert df_cleaned.loc[0, "legRouteTravelTime"] == 600.0
    assert df_cleaned.loc[0, "legRouteDistance"] == 1500.0
    assert df_cleaned.loc[0, "legRouteLinks"] == "1,2"
    assert df_cleaned.loc[0, "legVehicleIds"] == "veh_1"

    # Verify no lowercase remnants exist
    for col in [
        "personid",
        "planelementtype",
        "planelementindex",
        "activitytype",
        "activitylocationx",
        "activitylocationy",
        "legmode",
        "legdeparturetime",
        "legtraveltime",
        "legroutetype",
        "legroutestartlink",
        "legrouteendlink",
        "legroutetraveltime",
        "legroutedistance",
        "legroutelinks",
        "legvehicleids",
    ]:
        assert col not in df_cleaned.columns


def test_copy_with_compression_asim_file_to_beam_csv_and_parquet(
    tmp_path: Path,
) -> None:
    """copy_with_compression_asim_file_to_beam preserves id columns and avoids index pollution."""
    csv_in = tmp_path / "persons_in.csv"
    pd.DataFrame({"person_id": [1, 2], "household_id": [10, 20]}).to_csv(
        csv_in, index=False
    )

    # Test CSV output
    scenario_csv = tmp_path / "scenario_csv"
    scenario_csv.mkdir()
    copy_with_compression_asim_file_to_beam(
        asim_file_path=str(csv_in),
        beam_file_name="persons",
        file_format="csv",
        beam_scenario_folder=str(scenario_csv),
        state=None,
    )
    raw_csv = pd.read_csv(scenario_csv / "persons.csv.gz", compression="gzip")
    assert "person_id" in raw_csv.columns
    assert sorted(raw_csv["person_id"].tolist()) == [1, 2]

    # Test Parquet output
    scenario_pq = tmp_path / "scenario_pq"
    scenario_pq.mkdir()
    copy_with_compression_asim_file_to_beam(
        asim_file_path=str(csv_in),
        beam_file_name="persons",
        file_format="parquet",
        beam_scenario_folder=str(scenario_pq),
        state=None,
    )
    raw_pq = pd.read_parquet(scenario_pq / "persons.parquet")
    assert "person_id" in raw_pq.columns
    assert "__index_level_0__" not in raw_pq.columns


def test_exp_beta_selection_deterministic_regardless_of_plan_order(
    tmp_path: Path,
) -> None:
    """exp_beta sampling produces identical selection regardless of input DataFrame row order."""

    def run_selection(order: List[int]) -> List[bool]:
        scenario_dir = tmp_path / f"scenario_{'_'.join(map(str, order))}"
        scenario_dir.mkdir()

        p0 = _create_minimal_plans_df([1], [0], scores=[5.0])
        p1 = _create_minimal_plans_df([1], [1], scores=[6.0])
        all_plans = [p0, p1]
        prior_plans = pd.concat([all_plans[i] for i in order], ignore_index=True)
        prior_plans.to_parquet(scenario_dir / "plans.parquet", index=False)

        prior_per = pd.DataFrame({"household_id": [1], "person_id": [1]})
        prior_per.to_parquet(scenario_dir / "persons.parquet", index=False)
        prior_hh = pd.DataFrame({"cars": [1], "household_id": [1]})
        prior_hh.to_parquet(scenario_dir / "households.parquet", index=False)

        asim_dir = tmp_path / f"asim_{'_'.join(map(str, order))}"
        asim_dir.mkdir()
        new_plans = _create_minimal_plans_df([1], [0])  # score NaN -> effective 0.0
        new_plans_path = asim_dir / "beam_plans.parquet"
        new_plans.to_parquet(new_plans_path, index=False)

        asim_file_paths = {
            "beam_plans": (str(new_plans_path), None),
            "persons": (str(scenario_dir / "persons.parquet"), None),
            "households": (str(scenario_dir / "households.parquet"), None),
        }

        merge_replanned_asim_files(
            asim_file_paths=asim_file_paths,
            file_format="parquet",
            workspace=SimpleNamespace(),
            resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
            format_specific_output_records_fn=_dummy_format_records,
            settings=SimpleNamespace(
                beam=SimpleNamespace(
                    max_plans_memory=5,
                    plan_memory_beta=0.5,
                    new_plan_selection_policy="exp_beta",
                    new_plan_initial_score=0.0,
                    plan_selection_random_seed=12345,
                ),
                activitysim=SimpleNamespace(replan_hh_samp_size=0),
            ),
            state=SimpleNamespace(current_year=2020, current_inner_iter=1),
        )

        res = pd.read_parquet(scenario_dir / "plans.parquet")
        return list(res.sort_values("planindex")["planselected"])

    sel_order_0_1 = run_selection([0, 1])
    sel_order_1_0 = run_selection([1, 0])
    assert sel_order_0_1 == sel_order_1_0


def test_beam_data_helper_coerces_float_string_and_preserves_alphanumeric_trip_tour_ids(
    tmp_path: Path,
) -> None:
    """Float-formatted string tripId/tourId coerce to Int64, while alphanumeric IDs stay intact."""
    # 1. Float-formatted integer strings (e.g. '501.0') and empty strings on activity rows
    csv_float_ids = tmp_path / "plans_float_ids.csv.gz"
    df_float = _create_minimal_plans_df([1, 1], use_camel_case=True)
    df_float["tripId"] = ["", "501.0"]
    df_float["tourId"] = ["", "20.0"]
    df_float.to_csv(csv_float_ids, index=False, compression="gzip")

    cleaned_float = BeamDataHelper.read_and_clean(
        str(csv_float_ids), "plans", file_format="csv"
    )
    assert cleaned_float["trip_id"].dtype == pd.Int64Dtype()
    assert cleaned_float["tour_id"].dtype == pd.Int64Dtype()
    assert pd.isna(cleaned_float.loc[0, "trip_id"])
    assert cleaned_float.loc[1, "trip_id"] == 501
    assert cleaned_float.loc[1, "tour_id"] == 20

    # 2. Alphanumeric IDs (e.g. 'trip_1', 'tour_1') do not crash read_csv
    csv_alpha_ids = tmp_path / "plans_alpha_ids.csv.gz"
    df_alpha = _create_minimal_plans_df([1, 1], use_camel_case=True)
    df_alpha["tripId"] = ["", "trip_1"]
    df_alpha["tourId"] = ["", "tour_1"]
    df_alpha.to_csv(csv_alpha_ids, index=False, compression="gzip")

    cleaned_alpha = BeamDataHelper.read_and_clean(
        str(csv_alpha_ids), "plans", file_format="csv"
    )
    assert cleaned_alpha.loc[1, "trip_id"] == "trip_1"
    assert cleaned_alpha.loc[1, "tour_id"] == "tour_1"


def test_beam_data_helper_preserves_freight_person_ids(tmp_path: Path) -> None:
    """Alphanumeric freight agent IDs (e.g. 'ft-f94a4b1') do not crash read_csv and are preserved."""
    csv_freight_path = tmp_path / "plans_with_freight.csv.gz"
    df_raw = pd.DataFrame(
        {
            "personId": [10001, 10001, "ft-f94a4b1", "ft-f94a4b1"],
            "planIndex": [0, 0, 0, 0],
            "planElementIndex": [1, 2, 1, 2],
            "planElementType": ["activity", "leg", "activity", "leg"],
            "activityType": ["Home", "", "Depot", ""],
            "activityLocationX": [10.0, np.nan, 20.0, np.nan],
            "activityLocationY": [10.0, np.nan, 20.0, np.nan],
            "activityEndTime": [28800.0, np.nan, 32400.0, np.nan],
            "legMode": ["", "car", "", "truck"],
            "legDepartureTime": [np.nan, 28800.0, np.nan, 32400.0],
            "legTravelTime": [np.nan, 1200.0, np.nan, 1800.0],
            "planScore": [50.0, 50.0, 25.0, 25.0],
            "planSelected": [True, True, True, True],
            "tripId": ["", "1", "", "trip_ft_1"],
            "tourId": ["", "10", "", "tour_ft_1"],
        }
    )
    df_raw.to_csv(csv_freight_path, index=False, compression="gzip")

    cleaned = BeamDataHelper.read_and_clean(
        str(csv_freight_path), "plans", file_format="csv"
    )
    assert len(cleaned) == 4
    assert list(cleaned["person_id"]) == ["10001", "10001", "ft-f94a4b1", "ft-f94a4b1"]
    assert cleaned.loc[2, "person_id"] == "ft-f94a4b1"
    assert cleaned.loc[2, "ActivityType"] == "Depot"
    assert cleaned.loc[3, "trip_mode"] == "truck"
    assert cleaned.loc[3, "trip_id"] == "trip_ft_1"


def test_merge_replanned_asim_files_preserves_freight_and_drops_extinct_passengers(
    tmp_path: Path,
) -> None:
    """merge_replanned_asim_files retains freight agents intact, merges passenger plans, and drops extinct passengers."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()

    # Previous BEAM output (iteration 0) contains:
    # - Active passenger 1 (score 100.0)
    # - Extinct passenger 99999 (was in previous run, but dropped from current population)
    # - Freight agent "ft-f94a4b1" (with routed leg links)
    prior_beam_plans_path = scenario_dir / "beam_plans_out.csv.gz"
    prior_plans_df = pd.DataFrame(
        {
            "personId": [1, 1, 99999, 99999, "ft-f94a4b1", "ft-f94a4b1"],
            "planIndex": [0, 0, 0, 0, 0, 0],
            "planElementIndex": [1, 2, 1, 2, 1, 2],
            "planElementType": [
                "activity",
                "leg",
                "activity",
                "leg",
                "activity",
                "leg",
            ],
            "activityType": ["Home", "", "Home", "", "Depot", ""],
            "activityLocationX": [1.0, np.nan, 2.0, np.nan, 3.0, np.nan],
            "activityLocationY": [1.0, np.nan, 2.0, np.nan, 3.0, np.nan],
            "activityEndTime": [28800.0, np.nan, 28800.0, np.nan, 28800.0, np.nan],
            "legMode": ["", "car", "", "car", "", "truck"],
            "legDepartureTime": [np.nan, 28800.0, np.nan, 28800.0, np.nan, 28800.0],
            "legTravelTime": [np.nan, 1200.0, np.nan, 1200.0, np.nan, 1500.0],
            "legRouteLinks": [
                None,
                "link_1,link_2",
                None,
                "link_3,link_4",
                None,
                "link_f1,link_f2",
            ],
            "legVehicleIds": [None, "veh_p1", None, "veh_p99", None, "veh_ft_1"],
            "planScore": [100.0, 100.0, 50.0, 50.0, 75.0, 75.0],
            "planSelected": [True, True, True, True, True, True],
            "tripId": ["", "101", "", "999", "", "ft_trip_1"],
            "tourId": ["", "201", "", "999", "", "ft_tour_1"],
        }
    )
    prior_plans_df.to_csv(prior_beam_plans_path, index=False, compression="gzip")

    # Current population (persons_final): only passenger 1 and new passenger 2 (passenger 99999 is extinct)
    prior_per = pd.DataFrame({"household_id": [1, 2], "person_id": [1, 2]})
    prior_per.to_csv(scenario_dir / "persons.csv.gz", index=False, compression="gzip")
    prior_hh = pd.DataFrame({"household_id": [1, 2], "cars": [1, 2]})
    prior_hh.to_csv(scenario_dir / "households.csv.gz", index=False, compression="gzip")

    # ActivitySim replanned plans: new plans for passenger 1 and passenger 2
    new_plans_df = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "planindex": [0, 0, 0, 0],
            "PlanElementIndex": [1, 2, 1, 2],
            "ActivityElement": ["activity", "leg", "activity", "leg"],
            "ActivityType": ["Home", "", "Work", ""],
            "x": [10.0, np.nan, 20.0, np.nan],
            "y": [10.0, np.nan, 20.0, np.nan],
            "departure_time": [8.0, np.nan, 9.0, np.nan],
            "trip_mode": ["", "walk", "", "bike"],
            "trip_id": [None, 102, None, 202],
            "tour_id": [None, 202, None, 302],
        }
    )
    new_plans_path = asim_dir / "plans_beam_in.parquet"
    new_plans_df.to_parquet(new_plans_path, index=False)

    new_per = pd.DataFrame({"household_id": [1, 2], "person_id": [1, 2]})
    new_per_path = asim_dir / "persons_beam_in.parquet"
    new_per.to_parquet(new_per_path, index=False)

    new_hh = pd.DataFrame({"household_id": [1, 2], "cars": [1, 2]})
    new_hh_path = asim_dir / "households_beam_in.parquet"
    new_hh.to_parquet(new_hh_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(new_per_path), None),
        "households": (str(new_hh_path), None),
        "beam_plans_out": (str(prior_beam_plans_path), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="csv",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_csv(scenario_dir / "plans.csv.gz", compression="gzip")

    # 1. Passenger 1 has 2 plans (historical plan 0 and new plan 1)
    p1_plans = merged_plans[merged_plans["person_id"].astype(str) == "1"]
    assert len(p1_plans) == 4  # 2 elements * 2 plans
    assert set(p1_plans["planindex"]) == {0, 1}
    p1_new = p1_plans[p1_plans["planindex"] == 1]
    assert (p1_new["planselected"] == True).all()

    # 2. Passenger 2 has 1 plan (new plan 0)
    p2_plans = merged_plans[merged_plans["person_id"].astype(str) == "2"]
    assert len(p2_plans) == 2
    assert (p2_plans["planselected"] == True).all()

    # 3. Extinct passenger 99999 was dropped
    assert "99999" not in merged_plans["person_id"].astype(str).values

    # 4. Freight agent "ft-f94a4b1" is preserved intact
    ft_plans = merged_plans[merged_plans["person_id"] == "ft-f94a4b1"]
    assert len(ft_plans) == 2
    assert ft_plans.iloc[0]["ActivityType"] == "Depot"
    assert ft_plans.iloc[1]["trip_mode"] == "truck"
    assert ft_plans.iloc[1]["legRouteLinks"] == "link_f1,link_f2"
    assert (ft_plans["planselected"] == True).all()
    assert (ft_plans["planscore"] == 75.0).all()


def test_beam_data_helper_coerce_nullable_integer_ids() -> None:
    """coerce_nullable_integer_ids correctly coerces whole numbers and preserves strings/fractions."""
    # 1. Whole floats -> Int64
    s_float = pd.Series([1.0, 2.0, np.nan])
    coerced = BeamDataHelper.coerce_nullable_integer_ids(s_float)
    assert coerced.dtype == pd.Int64Dtype()
    assert list(coerced.dropna()) == [1, 2]

    # 2. Fractions -> unchanged float
    s_frac = pd.Series([1.5, 2.0, np.nan])
    coerced_frac = BeamDataHelper.coerce_nullable_integer_ids(s_frac)
    assert coerced_frac.dtype == s_frac.dtype

    # 3. Numeric strings -> Int64
    s_str = pd.Series(["10", "20", None])
    coerced_str = BeamDataHelper.coerce_nullable_integer_ids(s_str)
    assert coerced_str.dtype == pd.Int64Dtype()
    assert list(coerced_str.dropna()) == [10, 20]

    # 4. Alphanumeric freight strings -> unchanged object
    s_ft = pd.Series(["ft-1", "ft-2", None])
    coerced_ft = BeamDataHelper.coerce_nullable_integer_ids(s_ft)
    assert coerced_ft.dtype == object

    # 5. Mixed numeric and alphanumeric -> unchanged object
    s_mixed = pd.Series(["10001", "ft-f94a4b1"])
    coerced_mixed = BeamDataHelper.coerce_nullable_integer_ids(s_mixed)
    assert coerced_mixed.dtype == object

    # 6. All nulls -> Int64
    s_null = pd.Series([None, np.nan])
    coerced_null = BeamDataHelper.coerce_nullable_integer_ids(s_null)
    assert coerced_null.dtype == pd.Int64Dtype()


def test_merge_replanned_asim_files_preserves_numeric_freight_via_truck_mode(
    tmp_path: Path,
) -> None:
    """Freight agent with a purely numeric ID (e.g. 88888) is preserved if using truck/freight trip mode."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    asim_dir = tmp_path / "asim"
    asim_dir.mkdir()

    # Prior BEAM output has passenger 1, extinct passenger 99999, and numeric freight agent 88888 (truck mode)
    prior_beam_plans_path = scenario_dir / "beam_plans_out.csv.gz"
    prior_plans_df = pd.DataFrame(
        {
            "personId": [1, 1, 99999, 99999, 88888, 88888],
            "planIndex": [0, 0, 0, 0, 0, 0],
            "planElementIndex": [1, 2, 1, 2, 1, 2],
            "planElementType": [
                "activity",
                "leg",
                "activity",
                "leg",
                "activity",
                "leg",
            ],
            "activityType": ["Home", "", "Home", "", "Depot", ""],
            "activityLocationX": [1.0, np.nan, 2.0, np.nan, 3.0, np.nan],
            "activityLocationY": [1.0, np.nan, 2.0, np.nan, 3.0, np.nan],
            "activityEndTime": [28800.0, np.nan, 28800.0, np.nan, 28800.0, np.nan],
            "legMode": ["", "car", "", "car", "", "truck"],
            "legDepartureTime": [np.nan, 28800.0, np.nan, 28800.0, np.nan, 28800.0],
            "legTravelTime": [np.nan, 1200.0, np.nan, 1200.0, np.nan, 1500.0],
            "planScore": [100.0, 100.0, 50.0, 50.0, 80.0, 80.0],
            "planSelected": [True, True, True, True, True, True],
            "tripId": ["", "101", "", "999", "", "888881"],
            "tourId": ["", "201", "", "999", "", "888881"],
        }
    )
    prior_plans_df.to_csv(prior_beam_plans_path, index=False, compression="gzip")

    # Current population: only passenger 1
    prior_per = pd.DataFrame({"household_id": [1], "person_id": [1]})
    prior_per.to_csv(scenario_dir / "persons.csv.gz", index=False, compression="gzip")
    prior_hh = pd.DataFrame({"household_id": [1], "cars": [1]})
    prior_hh.to_csv(scenario_dir / "households.csv.gz", index=False, compression="gzip")

    # ActivitySim produces replanned plan for passenger 1
    new_plans_df = pd.DataFrame(
        {
            "person_id": [1, 1],
            "planindex": [0, 0],
            "PlanElementIndex": [1, 2],
            "ActivityElement": ["activity", "leg"],
            "ActivityType": ["Home", ""],
            "x": [10.0, np.nan],
            "y": [10.0, np.nan],
            "departure_time": [8.0, np.nan],
            "trip_mode": ["", "walk"],
            "trip_id": [None, 102],
            "tour_id": [None, 202],
        }
    )
    new_plans_path = asim_dir / "plans_beam_in.parquet"
    new_plans_df.to_parquet(new_plans_path, index=False)

    new_per = pd.DataFrame({"household_id": [1], "person_id": [1]})
    new_per_path = asim_dir / "persons_beam_in.parquet"
    new_per.to_parquet(new_per_path, index=False)

    new_hh = pd.DataFrame({"household_id": [1], "cars": [1]})
    new_hh_path = asim_dir / "households_beam_in.parquet"
    new_hh.to_parquet(new_hh_path, index=False)

    asim_file_paths = {
        "beam_plans": (str(new_plans_path), None),
        "persons": (str(new_per_path), None),
        "households": (str(new_hh_path), None),
        "beam_plans_out": (str(prior_beam_plans_path), None),
    }

    merge_replanned_asim_files(
        asim_file_paths=asim_file_paths,
        file_format="csv",
        workspace=SimpleNamespace(),
        resolve_beam_exchange_scenario_folder_fn=lambda _: str(scenario_dir),
        format_specific_output_records_fn=_dummy_format_records,
        settings=SimpleNamespace(
            beam=SimpleNamespace(
                max_plans_memory=5, new_plan_selection_policy="force_new"
            ),
            activitysim=SimpleNamespace(replan_hh_samp_size=0),
        ),
        state=SimpleNamespace(current_year=2020, current_inner_iter=1),
    )

    merged_plans = pd.read_csv(scenario_dir / "plans.csv.gz", compression="gzip")

    # Numeric freight agent 88888 is retained because of truck trip_mode
    ft_plans = merged_plans[merged_plans["person_id"].astype(str) == "88888"]
    assert len(ft_plans) == 2
    assert ft_plans.iloc[0]["ActivityType"] == "Depot"
    assert ft_plans.iloc[1]["trip_mode"] == "truck"

    # Extinct passenger 99999 was dropped
    assert "99999" not in merged_plans["person_id"].astype(str).values
