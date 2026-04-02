from types import SimpleNamespace

import pandas as pd

from pilates.activitysim.preprocessor import _update_persons_table


def _households() -> pd.DataFrame:
    households = pd.DataFrame(
        {"block_id": ["0001", "0002", "0003", "0004"]},
        index=pd.Index([1, 2, 3, 4], name="household_id"),
    )
    return households


def _blocks() -> pd.DataFrame:
    blocks = pd.DataFrame(
        {
            "TAZ": [101, 102, 103, 104],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
        },
        index=pd.Index(["0001", "0002", "0003", "0004"], name="block_id"),
    )
    return blocks


def test_update_persons_table_keeps_missing_worker_student_locations_and_flags_them():
    persons = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 4],
            "age": [40, 15, 35, 10, 0],
            "worker": [1, 0, 1, 0, 0],
            "student": [0, 1, 0, 0, 0],
            "work_zone_id": [-1, -1, 555, -1, -1],
            "school_zone_id": [-1, -1, -1, -1, -1],
        },
        index=pd.Index([10, 20, 30, 40, 50], name="person_id"),
    )

    updated = _update_persons_table(
        persons,
        _households(),
        pd.Series([], dtype=int),
        _blocks(),
        settings={
            "activitysim": {
                "workplace_reassignment_share": 0.0,
                "school_reassignment_share": 0.0,
                "random_seed": 7,
            }
        },
        state=SimpleNamespace(year=2029, iteration=0),
    )

    assert set(updated.index) == {10, 20, 30, 40}
    assert updated.loc[10, "needs_workplace_reassignment"]
    assert not updated.loc[10, "needs_school_reassignment"]
    assert updated.loc[20, "needs_school_reassignment"]
    assert not updated.loc[20, "needs_workplace_reassignment"]
    assert not updated.loc[30, "needs_workplace_reassignment"]
    assert not updated.loc[30, "needs_school_reassignment"]


def test_update_persons_table_marks_configured_share_of_valid_assignments_for_reassignment():
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "age": [40, 41, 38, 39, 14, 15, 16, 17],
            "worker": [1, 1, 1, 1, 0, 0, 0, 0],
            "student": [0, 0, 0, 0, 1, 1, 1, 1],
            "work_zone_id": [501, 502, 503, 504, -1, -1, -1, -1],
            "school_zone_id": [-1, -1, -1, -1, 601, 602, 603, 604],
        },
        index=pd.Index([10, 11, 12, 13, 20, 21, 22, 23], name="person_id"),
    )

    updated = _update_persons_table(
        persons,
        _households(),
        pd.Series([], dtype=int),
        _blocks(),
        settings={
            "activitysim": {
                "workplace_reassignment_share": 0.5,
                "school_reassignment_share": 0.5,
                "random_seed": 17,
            }
        },
        state=SimpleNamespace(year=2029, iteration=0),
    )

    assert updated.loc[[10, 11, 12, 13], "needs_workplace_reassignment"].sum() == 2
    assert updated.loc[[20, 21, 22, 23], "needs_school_reassignment"].sum() == 2
    assert not updated["needs_workplace_reassignment"].loc[[20, 21, 22, 23]].any()
    assert not updated["needs_school_reassignment"].loc[[10, 11, 12, 13]].any()


def test_update_persons_table_emits_canonical_activitysim_location_columns_and_aliases():
    persons = pd.DataFrame(
        {
            "household_id": [1, 2],
            "age": [40, 15],
            "worker": [1, 0],
            "student": [0, 1],
            "work_zone_id": [501, -1],
            "school_zone_id": [-1, 601],
        },
        index=pd.Index([10, 20], name="person_id"),
    )

    updated = _update_persons_table(
        persons,
        _households(),
        pd.Series([], dtype=int),
        _blocks(),
        settings={
            "activitysim": {
                "workplace_reassignment_share": 0.0,
                "school_reassignment_share": 0.0,
                "random_seed": 7,
            }
        },
        state=SimpleNamespace(year=2029, iteration=0),
    )

    assert updated["workplace_zone_id"].to_dict() == {10: 501, 20: -1}
    assert updated["workplace_taz"].to_dict() == {10: 501, 20: -1}
    assert updated["work_zone_id"].to_dict() == {10: 501, 20: -1}
    assert updated["school_zone_id"].to_dict() == {10: -1, 20: 601}
    assert updated["school_taz"].to_dict() == {10: -1, 20: 601}
    assert "workplace_location_logsum" in updated.columns
    assert "school_location_logsum" in updated.columns
    assert updated["workplace_location_logsum"].isna().all()
    assert updated["school_location_logsum"].isna().all()
