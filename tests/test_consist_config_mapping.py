import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from pilates.config.models import (
    BeamLinkstatsAdmissionConfig,
    PilatesConfig,
    admission_policy_fingerprint,
)
from pilates.utils.consist_config import (
    build_activitysim_identity_inputs,
    build_beam_identity_inputs,
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
    build_urbansim_facet,
    build_urbansim_identity_config,
)


_DECLARED_IDENTITY = "sha256:file:" + "a" * 64


def _settings_with_urbansim_admission(
    *,
    mode: str = "strict",
    identity: str = _DECLARED_IDENTITY,
    source_uri: str = "s3://catalog/baselines/input.h5",
    source_label: str = "catalog baseline",
) -> PilatesConfig:
    return PilatesConfig(
        run={
            "region": "test",
            "scenario": "test",
            "start_year": 2020,
            "end_year": 2021,
            "output_directory": "/tmp/output",
            "output_run_name": "test-run",
            "models": {
                "land_use": "urbansim",
                "travel": None,
                "activity_demand": None,
                "vehicle_ownership": None,
            },
        },
        shared={
            "geography": {"FIPS": {"county": ["06001"]}, "local_crs": "EPSG:32048"},
            "skims": {"fname": "skims.h5"},
            "database": {"enabled": True, "type": "duckdb", "path": "/tmp/test.duckdb"},
        },
        infrastructure={
            "container_manager": "docker",
            "singularity_images": {},
            "docker_images": {},
            "docker_config": {"stdout": False, "pull_latest": False},
        },
        urbansim={
            "local_data_input_folder": "usim_input",
            "local_mutable_data_folder": "usim_mutable",
            "client_base_folder": "/app",
            "client_data_folder": "/tmp",
            "input_file_template": "input_{region_id}.h5",
            "input_file_template_year": "input_{region_id}_{year}.h5",
            "output_file_template": "output_{year}.h5",
            "command_template": "echo",
            "admission": {
                "initial_datastore": {
                    "mode": mode,
                    "expectation": {
                        "kind": "declared_digest",
                        "identity": identity,
                        "source_uri": source_uri,
                        "source_label": source_label,
                    },
                }
            },
        },
    )


def test_build_scenario_consist_kwargs_includes_run_facet():
    settings = MagicMock()
    settings.get_initialization_signature.return_value = {"hello": "world"}
    settings.run = MagicMock()
    settings.run.to_consist_facet.return_value = {"region": "test", "start_year": 2010}

    kwargs = build_scenario_consist_kwargs(settings)
    assert kwargs["config"] == {"hello": "world"}
    assert kwargs["facet"]["run"] == {"region": "test", "start_year": 2010}
    assert kwargs["facet_schema_version"] == "pilates_scenario_v1"


def test_build_activitysim_identity_inputs_requires_dir(tmp_path):
    settings = MagicMock()
    settings.activitysim = MagicMock()
    settings.activitysim.local_mutable_configs_folder = "activitysim/configs"

    (tmp_path / "activitysim" / "configs").mkdir(parents=True)
    identity_inputs = build_activitysim_identity_inputs(settings, str(tmp_path))
    assert len(identity_inputs) == 1
    assert identity_inputs[0][0] == "asim_mutable_configs"
    assert identity_inputs[0][1] == (tmp_path / "activitysim" / "configs")


def test_build_activitysim_identity_inputs_missing_dir_raises(tmp_path):
    settings = MagicMock()
    settings.activitysim = MagicMock()
    settings.activitysim.local_mutable_configs_folder = "activitysim/configs"

    with pytest.raises(FileNotFoundError):
        build_activitysim_identity_inputs(settings, str(tmp_path))


def test_build_beam_identity_inputs_discovers_conf_files(tmp_path):
    settings = MagicMock()
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/input"

    root = tmp_path / "beam" / "input"
    (root / "sub").mkdir(parents=True)
    (root / "a.conf").write_text("a=1")
    (root / "sub" / "b.conf").write_text("b=2")
    (root / "sub" / "ignore.txt").write_text("nope")

    identity_inputs = build_beam_identity_inputs(settings, str(tmp_path))
    labels = [lbl for (lbl, _) in identity_inputs]
    paths = [p for (_, p) in identity_inputs]

    assert labels == ["beam_conf/a.conf", "beam_conf/sub/b.conf"]
    assert paths == [root / "a.conf", root / "sub" / "b.conf"]


def test_build_step_consist_kwargs_beam_includes_identity_inputs(tmp_path):
    settings = MagicMock()
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/input"
    settings.beam.config = "main.conf"
    settings.beam.sample = 1.0
    settings.beam.replanning_portion = 0.4
    settings.beam.memory = "180g"
    settings.beam.discard_plans_every_year = False
    settings.beam.max_plans_memory = 5
    settings.beam.router_directory = "r5"
    settings.beam.scenario_folder = "scenario"
    settings.beam.to_consist_facet.return_value = {"sample": 1.0}

    root = tmp_path / "beam" / "input"
    root.mkdir(parents=True)
    (root / "main.conf").write_text("x=1")

    kwargs = build_step_consist_kwargs("beam", settings, workspace_path=str(tmp_path))
    assert kwargs["facet_schema_version"] == "beam_v1"
    assert kwargs["facet"] == {"sample": 1.0}
    assert kwargs["identity_inputs"][0][0] == "beam_conf/main.conf"


def test_build_beam_identity_inputs_missing_dir_returns_empty(tmp_path):
    settings = MagicMock()
    settings.beam = MagicMock()
    settings.beam.local_mutable_data_folder = "beam/input"

    assert build_beam_identity_inputs(settings, str(tmp_path)) == []


def test_urbansim_declared_digest_admission_uses_sanitized_cache_identity():
    settings = _settings_with_urbansim_admission()

    signature = settings.get_initialization_signature()
    policy = signature["urbansim_admission"]["initial_datastore"]

    assert policy == {
        "mode": "strict",
        "kind": "declared_digest",
        "identity": _DECLARED_IDENTITY,
    }
    assert (
        build_urbansim_identity_config(settings)["admission"]["initial_datastore"]
        == policy
    )

    changed_mode = _settings_with_urbansim_admission(mode="warn")
    changed_identity = _settings_with_urbansim_admission(
        identity="sha256:file:" + "b" * 64
    )
    changed_source_uri = _settings_with_urbansim_admission(
        source_uri="https://mirror.example.test/other-input.h5"
    )
    changed_source_label = _settings_with_urbansim_admission(
        source_label="regional baseline mirror"
    )

    assert changed_mode.get_initialization_signature() != signature
    assert changed_identity.get_initialization_signature() != signature
    assert changed_source_uri.get_initialization_signature() == signature
    assert changed_source_label.get_initialization_signature() == signature
    assert (
        build_urbansim_identity_config(changed_source_label)
        == build_urbansim_identity_config(settings)
    )
    assert (
        build_urbansim_facet(changed_source_label)["admission"]["initial_datastore"]
        ["expectation"]["source_label"]
        == "regional baseline mirror"
    )


@pytest.mark.parametrize(
    "identity",
    ["sha256:file:" + "A" * 64, "sha256:file:not-a-digest"],
)
def test_urbansim_declared_digest_admission_rejects_noncanonical_identity(identity):
    with pytest.raises(ValidationError, match="sha256:file:<64 lowercase hexadecimal"):
        _settings_with_urbansim_admission(identity=identity)


def test_beam_linkstats_admission_normalizes_nested_and_legacy_expected_bytes_path():
    nested = BeamLinkstatsAdmissionConfig(
        mode="strict",
        expectation={
            "kind": "prior_run",
            "expected_run_id": "baseline-run",
            "artifact_key": "linkstats_warmstart",
            "expected_bytes_path": "/archive/nested-linkstats.csv.gz",
        },
    )
    legacy = BeamLinkstatsAdmissionConfig(
        mode="strict",
        expected_run_id="baseline-run",
        artifact_key="linkstats_warmstart",
        expected_bytes_path="/archive/legacy-linkstats.csv.gz",
    )

    assert nested.expectation.expected_bytes_path == "/archive/nested-linkstats.csv.gz"
    assert nested.expected_bytes_path == "/archive/nested-linkstats.csv.gz"
    assert legacy.expectation.expected_bytes_path == "/archive/legacy-linkstats.csv.gz"
    assert legacy.expected_bytes_path == "/archive/legacy-linkstats.csv.gz"
    assert admission_policy_fingerprint(nested) == {
        "mode": "strict",
        "kind": "prior_run",
        "expected_run_id": "baseline-run",
        "artifact_key": "linkstats_warmstart",
    }
