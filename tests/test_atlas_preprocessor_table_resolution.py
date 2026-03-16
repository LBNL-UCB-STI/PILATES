from pilates.atlas.preprocessor import _resolve_atlas_h5_table_key


class _FakeHDFStore:
    def __init__(self, keys):
        self._keys = tuple(keys)

    def __contains__(self, key):
        normalized = key if str(key).startswith("/") else f"/{key}"
        return normalized in self._keys

    def keys(self):
        return list(self._keys)


def test_resolve_atlas_h5_table_key_prefers_year_scoped_when_available(tmp_path):
    store = _FakeHDFStore(("/households", "/2019/households"))
    resolved = _resolve_atlas_h5_table_key(
        store, year=2019, table="households", is_start_year=False
    )

    assert resolved == "/2019/households"


def test_resolve_atlas_h5_table_key_falls_back_to_root_for_non_start_year(tmp_path):
    store = _FakeHDFStore(("/households",))
    resolved = _resolve_atlas_h5_table_key(
        store, year=2019, table="households", is_start_year=False
    )

    assert resolved == "households"


def test_resolve_atlas_h5_table_key_uses_root_for_start_year(tmp_path):
    store = _FakeHDFStore(("/households",))
    resolved = _resolve_atlas_h5_table_key(
        store, year=2017, table="households", is_start_year=True
    )

    assert resolved == "households"


def test_resolve_atlas_h5_table_key_falls_back_to_prior_year_scoped_table(tmp_path):
    store = _FakeHDFStore(("/2023/households",))
    resolved = _resolve_atlas_h5_table_key(
        store, year=2025, table="households", is_start_year=False
    )

    assert resolved == "/2023/households"


def test_resolve_atlas_h5_table_key_falls_back_to_earliest_future_year_scoped_table(
    tmp_path,
):
    store = _FakeHDFStore(("/2027/households",))
    resolved = _resolve_atlas_h5_table_key(
        store, year=2025, table="households", is_start_year=False
    )

    assert resolved == "/2027/households"
