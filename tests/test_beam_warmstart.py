from types import SimpleNamespace

from pilates.utils.beam_warmstart import resolve_initial_linkstats_path


class DummyWorkspace:
    def __init__(self, full_path):
        self.full_path = str(full_path)

    def get_beam_mutable_data_dir(self):
        return f"{self.full_path}/beam/input"


def test_resolve_initial_linkstats_path_uses_configured_relative_path(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    configured = (
        tmp_path
        / "beam"
        / "input"
        / "sfbay"
        / "custom"
        / "warmstart.csv.gz"
    )
    configured.parent.mkdir(parents=True, exist_ok=True)
    configured.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(
            router_directory="r5/network",
            warmstart_linkstats_path="custom/warmstart.csv.gz",
        ),
    )

    resolved = resolve_initial_linkstats_path(settings, workspace)

    assert resolved == str(configured)


def test_resolve_initial_linkstats_path_expands_router_directory_placeholder(tmp_path):
    workspace = DummyWorkspace(tmp_path)
    configured = (
        tmp_path
        / "beam"
        / "input"
        / "sfbay"
        / "r5"
        / "network"
        / "init.linkstats.csv.gz"
    )
    configured.parent.mkdir(parents=True, exist_ok=True)
    configured.write_text("linkstats", encoding="utf-8")

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(
            router_directory="r5/network",
            warmstart_linkstats_path="{router_directory}/init.linkstats.csv.gz",
        ),
    )

    resolved = resolve_initial_linkstats_path(settings, workspace)

    assert resolved == str(configured)


def test_resolve_initial_linkstats_path_returns_none_when_unset(tmp_path):
    workspace = DummyWorkspace(tmp_path)

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay"),
        beam=SimpleNamespace(
            router_directory="r5/network",
            warmstart_linkstats_path=None,
        ),
    )

    resolved = resolve_initial_linkstats_path(settings, workspace)

    assert resolved is None
