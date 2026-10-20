"""Unit test for SFBay ActivitySim <-> BEAM supply-demand loop configs and inheritance."""

from pathlib import Path

from pilates.config.models import load_config, resolve_config_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_HPC_PATH = REPO_ROOT / "scenarios/sfbay/settings-sfbay-consist-base-hpc.yaml"
SMOKE_HPC_PATH = REPO_ROOT / "scenarios/sfbay/settings-sfbay-smoke-sd-loop-hpc.yaml"
SMOKE_LOCAL_PATH = REPO_ROOT / "scenarios/sfbay/settings-sfbay-smoke-sd-loop-local.yaml"

EXPECTED_SKIMS_DIGEST = (
    "sha256:file:347d8b2f1459ec2c1751e6d3f33730a8923c9c82b1ca6458f82930c5cd01eb8f"
)
EXPECTED_LINKSTATS_DIGEST = (
    "sha256:file:26e8153fa3e73cf3635c45cb00b185acf327dd75338ad77ca5e30ce060c6b2e5"
)


def test_base_hpc_config_loads_and_has_expected_admissions():
    cfg = load_config(BASE_HPC_PATH)
    assert cfg.run.region == "sfbay"
    assert cfg.shared.skims.fname == "skims.omx"
    assert cfg.beam.config == "sfbay-pilates-base-omx.conf"
    assert cfg.beam.router_directory == "r5/sfbay-cbg5500-weakConn-network"

    # BEAM linkstats admission
    assert cfg.beam.admission is not None
    assert cfg.beam.admission.linkstats is not None
    assert cfg.beam.admission.linkstats.mode == "strict"
    assert (
        cfg.beam.admission.linkstats.expectation.identity == EXPECTED_LINKSTATS_DIGEST
    )

    # ActivitySim initial skims admission
    assert cfg.activitysim.admission is not None
    assert cfg.activitysim.admission.initial_skims is not None
    assert cfg.activitysim.admission.initial_skims.mode == "strict"
    assert (
        cfg.activitysim.admission.initial_skims.expectation.identity
        == EXPECTED_SKIMS_DIGEST
    )


def test_smoke_sd_loop_hpc_inherits_and_specializes():
    raw_dict = resolve_config_dict(SMOKE_HPC_PATH)
    assert "base" not in raw_dict
    assert "extends" not in raw_dict

    cfg = load_config(SMOKE_HPC_PATH)
    assert cfg.run.region == "sfbay"
    assert cfg.run.supply_demand_iters == 2
    assert cfg.run.output_run_name == "consist-sfbay-smoke-sd-loop-2019"
    assert cfg.run.models.activity_demand == "activitysim"
    assert cfg.run.models.travel == "beam"
    assert cfg.run.models.land_use is None
    assert cfg.run.models.vehicle_ownership is None

    # ActivitySim specialization
    assert cfg.activitysim.household_sample_size == 5000
    assert cfg.activitysim.num_processes == 20
    assert (
        cfg.activitysim.admission.initial_skims.expectation.identity
        == EXPECTED_SKIMS_DIGEST
    )

    # BEAM specialization
    assert cfg.beam.memory == "150g"
    assert cfg.beam.sample == 0.1
    assert (
        cfg.beam.admission.linkstats.expectation.identity == EXPECTED_LINKSTATS_DIGEST
    )


def test_smoke_sd_loop_local_inherits_from_smoke_hpc():
    cfg = load_config(SMOKE_LOCAL_PATH)
    assert cfg.run.region == "sfbay"
    assert cfg.run.supply_demand_iters == 2
    assert cfg.activitysim.household_sample_size == 5000
    assert cfg.activitysim.num_processes == 4
    assert cfg.infrastructure.container_manager == "docker"
    assert cfg.beam.memory == "16g"


def test_smoke_sd_loop_resolves_when_referenced_from_repo_root(tmp_path: Path):
    # Simulates job_runner.sh writing settings_<jobid>.yaml at repo root
    simulated_runner_file = REPO_ROOT / f"settings_test_simulated_{tmp_path.name}.yaml"
    try:
        simulated_runner_file.write_text(
            SMOKE_HPC_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
        cfg = load_config(simulated_runner_file)
        assert cfg.run.region == "sfbay"
        assert cfg.beam.memory == "150g"
    finally:
        if simulated_runner_file.exists():
            simulated_runner_file.unlink()
