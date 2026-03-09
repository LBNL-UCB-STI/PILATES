import pandas as pd
import pytest

from pilates.activitysim.preprocessor import _validate_household_person_consistency


def test_validate_household_person_consistency_accepts_matching_inputs():
    households = pd.DataFrame({"persons": [2, 1]}, index=pd.Index([10, 20], name="household_id"))
    persons = pd.DataFrame(
        {
            "household_id": [10, 10, 20],
            "age": [40, 12, 33],
        }
    )

    _validate_household_person_consistency(households, persons)


def test_validate_household_person_consistency_rejects_households_without_persons():
    households = pd.DataFrame({"persons": [2, 1]}, index=pd.Index([10, 20], name="household_id"))
    persons = pd.DataFrame(
        {
            "household_id": [10, 10],
            "age": [40, 12],
        }
    )

    with pytest.raises(ValueError, match="households_without_persons=1"):
        _validate_household_person_consistency(households, persons)


def test_validate_household_person_consistency_rejects_orphan_person_refs():
    households = pd.DataFrame({"persons": [2]}, index=pd.Index([10], name="household_id"))
    persons = pd.DataFrame(
        {
            "household_id": [10, 99],
            "age": [40, 12],
        }
    )

    with pytest.raises(ValueError, match="orphan_person_household_refs=1"):
        _validate_household_person_consistency(households, persons)
