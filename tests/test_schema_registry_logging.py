import types

from pilates.utils import consist_runtime as cr


def _install_consist_stub(monkeypatch, calls):
    def _log_input(path, key=None, enabled=None, **meta):
        calls.append(("input", key, meta))
        return {"path": path, "key": key}

    def _log_output(path, key=None, enabled=None, **meta):
        calls.append(("output", key, meta))
        return {"path": path, "key": key}

    stub = types.SimpleNamespace(log_input=_log_input, log_output=_log_output)
    monkeypatch.setattr(cr, "consist", stub)


def test_log_input_attaches_schema_for_known_key(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_input("/tmp/households.csv", key="households_asim_in", enabled=True)

    assert calls
    _, key, meta = calls[0]
    assert key == "households_asim_in"
    assert meta["schema"].__name__ == "HouseholdsAsimIn"


def test_log_output_does_not_override_explicit_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    class _ExplicitSchema:
        pass

    cr.log_output(
        "/tmp/households.csv",
        key="households_asim_in",
        enabled=True,
        schema=_ExplicitSchema,
    )

    assert calls
    _, _, meta = calls[0]
    assert meta["schema"] is _ExplicitSchema


def test_log_input_unknown_key_has_no_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_input("/tmp/unknown.csv", key="not_a_known_key", enabled=True)

    assert calls
    _, _, meta = calls[0]
    assert "schema" not in meta


def test_log_input_alias_key_attaches_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_input("/tmp/households.csv", key="asim_households_in", enabled=True)

    assert calls
    _, key, meta = calls[0]
    assert key == "asim_households_in"
    assert meta["schema"].__name__ == "HouseholdsAsimIn"


def test_log_output_linkstats_key_attaches_beam_linkstats_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    cr.log_output("/tmp/linkstats.parquet", key="linkstats", enabled=True)

    assert calls
    _, key, meta = calls[0]
    assert key == "linkstats"
    assert meta["schema"].__name__ == "BeamLinkstats"


def test_log_output_phys_sim_linkstats_key_attaches_beam_linkstats_schema(monkeypatch):
    calls = []
    _install_consist_stub(monkeypatch, calls)

    key = "linkstats_unmodified_parquet__y2018__i0__phys_sim_iter9__beam_sub_iter1"
    cr.log_output("/tmp/linkstats.parquet", key=key, enabled=True)

    assert calls
    _, resolved_key, meta = calls[0]
    assert resolved_key == key
    assert meta["schema"].__name__ == "BeamLinkstats"
