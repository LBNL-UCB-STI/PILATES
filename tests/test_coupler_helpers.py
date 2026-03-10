from pilates.utils import coupler_helpers


class CouplerWithSetFromArtifact:
    def __init__(self) -> None:
        self.calls = []

    def set_from_artifact(self, key, value) -> None:
        self.calls.append(("set_from_artifact", key, value))

    def set(self, key, value) -> None:
        self.calls.append(("set", key, value))


class CouplerWithSetOnly:
    def __init__(self) -> None:
        self.calls = []

    def set(self, key, value) -> None:
        self.calls.append(("set", key, value))


class CouplerWithNamespaceView(CouplerWithSetFromArtifact):
    class _View:
        def __init__(self, outer, namespace) -> None:
            self._outer = outer
            self._namespace = namespace

        def set(self, key, value) -> None:
            self._outer.calls.append(("view.set", f"{self._namespace}/{key}", value))

    def view(self, namespace):
        return self._View(self, namespace)


def test_set_coupler_from_artifact_prefers_set_from_artifact() -> None:
    coupler = CouplerWithSetFromArtifact()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "usim_datastore_h5", "artifact", fallback="fallback"
    )
    assert coupler.calls == [("set_from_artifact", "usim_datastore_h5", "artifact")]


def test_set_coupler_from_artifact_falls_back_to_set() -> None:
    coupler = CouplerWithSetOnly()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "usim_datastore_h5", None, fallback="/tmp/path.h5"
    )
    assert coupler.calls == [("set", "usim_datastore_h5", "/tmp/path.h5")]


def test_set_coupler_from_artifact_resolves_alias_key() -> None:
    coupler = CouplerWithSetOnly()
    coupler_helpers.set_coupler_from_artifact(
        coupler, "asim_households_in", None, fallback="/tmp/households.csv"
    )
    assert coupler.calls == [("set", "households_asim_in", "/tmp/households.csv")]


def test_set_coupler_from_artifact_publishes_namespaced_and_legacy_keys() -> None:
    coupler = CouplerWithNamespaceView()
    coupler_helpers.set_coupler_from_artifact(
        coupler,
        "linkstats_warmstart",
        None,
        fallback="/tmp/linkstats.csv.gz",
    )
    assert ("view.set", "beam/linkstats_warmstart", "/tmp/linkstats.csv.gz") in coupler.calls
    assert ("set_from_artifact", "linkstats_warmstart", "/tmp/linkstats.csv.gz") in coupler.calls
