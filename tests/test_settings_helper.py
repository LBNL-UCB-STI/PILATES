from types import SimpleNamespace

from pilates.utils.settings_helper import get


def test_get_supports_plain_attribute_objects():
    settings = SimpleNamespace(
        run=SimpleNamespace(models=SimpleNamespace(activity_demand="activitysim")),
        activitysim=SimpleNamespace(file_format="parquet"),
    )

    assert get(settings, "run.models.activity_demand") == "activitysim"
    assert get(settings, "activitysim.file_format") == "parquet"
