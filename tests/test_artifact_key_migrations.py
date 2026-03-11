from pilates.workflows.artifact_keys import ASIM_HOUSEHOLDS_IN
from pilates.workflows.artifact_key_migrations import (
    canonicalize_artifact_mapping,
    resolve_artifact_key,
)
from pilates.workflows.artifact_keys import ArtifactKeys
from pilates.workflows.artifact_constants import BEAM_FULL_SKIMS


def test_resolve_artifact_key_alias():
    assert resolve_artifact_key("asim_households_in") == ArtifactKeys.ASIM_HOUSEHOLDS_IN


def test_canonicalize_artifact_mapping_prefers_canonical():
    mapping = canonicalize_artifact_mapping(
        {
            "asim_households_in": "/tmp/alias.csv",
            ArtifactKeys.ASIM_HOUSEHOLDS_IN: "/tmp/canonical.csv",
        }
    )
    assert mapping[ArtifactKeys.ASIM_HOUSEHOLDS_IN] == "/tmp/canonical.csv"


def test_artifact_constants_compatibility_exports():
    assert ASIM_HOUSEHOLDS_IN == ArtifactKeys.ASIM_HOUSEHOLDS_IN
    assert BEAM_FULL_SKIMS == ArtifactKeys.BEAM_FULL_SKIMS


def test_resolve_artifact_key_usim_current_aliases():
    assert (
        resolve_artifact_key("usim_datastore_current_h5")
        == ArtifactKeys.USIM_DATASTORE_CURRENT_H5
    )
    assert (
        resolve_artifact_key("usim_datastore_h5")
        == ArtifactKeys.USIM_DATASTORE_CURRENT_H5
    )
