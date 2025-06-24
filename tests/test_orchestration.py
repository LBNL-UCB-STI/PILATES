import subprocess
import os
import pytest
import yaml

# Get the directory of the test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the path to the run.py script (assuming it's in the parent directory)
RUN_SCRIPT = os.path.abspath(os.path.join(TEST_DIR, '..', 'run.py'))
# Get the path to the stubs directory
STUBS_DIR = os.path.join(TEST_DIR, 'stubs')

# Define the paths to the test configuration files
SETTINGS_FILES = {
    'landuse_off_replanning_on': os.path.join(os.path.dirname(RUN_SCRIPT), 'settings_test_landuse_off_replanning_on.yaml'),
    'landuse_on_replanning_on': os.path.join(os.path.dirname(RUN_SCRIPT), 'settings_test_landuse_on_replanning_on.yaml'),
    'landuse_on_replanning_off': os.path.join(os.path.dirname(RUN_SCRIPT), 'settings_test_landuse_on_replanning_off.yaml'),
}

# Define expected dummy output files for each model stub
EXPECTED_STUB_OUTPUTS = {
    'urbansim': 'urbansim_output.h5',
    'atlas': 'atlas_output.csv',
    'activitysim': 'activitysim_output.parquet',
    'beam': 'beam_output.csv.gz',
}

os.chdir("../")

def run_pilates_with_settings(settings_file):
    """Runs the pilates run.py script with the given settings file."""
    print(f"\nRunning run.py with settings: {settings_file}")
    result = subprocess.run(
        ['python', RUN_SCRIPT, '-c', settings_file],
        capture_output=True,
        text=True,
        check=False # Don't raise exception on non-zero exit code yet
    )
    print("Stdout:")
    print(result.stdout)
    print("Stderr:")
    print(result.stderr)
    return result

def check_dummy_outputs(output_dir, enabled_models):
    """Checks for the existence of expected dummy output files."""
    print(f"Checking for dummy outputs in: {output_dir}")
    missing_files = []
    for model, filename in EXPECTED_STUB_OUTPUTS.items():
        if model in enabled_models:
            expected_path = os.path.join(output_dir, filename)
            print(f"Checking for {expected_path}...")
            if not os.path.exists(expected_path):
                missing_files.append(expected_path)
            else:
                print(f"Found {expected_path}")
    return missing_files

@pytest.mark.parametrize("config_name, settings_file", SETTINGS_FILES.items())
def test_orchestration_with_stubs(config_name, settings_file):
    """Tests the high-level orchestration with different stub configurations."""
    assert os.path.exists(settings_file), f"Settings file not found: {settings_file}"

    # Read settings to determine expected behavior
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    # Determine which models are expected to run based on settings (simplified)
    # This logic might need refinement based on actual run.py model execution flow
    enabled_models = set()
    if settings.get('land_use_enabled'):
        enabled_models.add('urbansim')
        enabled_models.add('atlas') # Atlas might run even if land_use_enabled is false depending on config
    if settings.get('replanning_enabled'):
         enabled_models.add('activitysim')
         enabled_models.add('beam') # BEAM might run even if replanning_enabled is false depending on config

    # Run the main script
    result = run_pilates_with_settings(settings_file)

    # Assert that the script ran successfully
    assert result.returncode == 0, f"run.py failed with settings {settings_file}. Stderr: {result.stderr}"

    # Determine the expected output directory from settings (assuming output_dir is specified)
    # You might need to adjust this based on how output_dir is structured/determined in run.py
    output_dir_setting = settings.get('output_dir')
    assert output_dir_setting, f"output_dir not specified in {settings_file}"
    full_output_dir = os.path.abspath(output_dir_setting) # Assuming output_dir is relative to workspace root

    # Check for expected dummy output files
    missing = check_dummy_outputs(full_output_dir, enabled_models)
    assert not missing, f"Missing expected dummy output files: {missing}"

    # Optional: Add assertions to check for specific messages in stdout/stderr
    # assert "Using stub for urbansim" in result.stdout
    # assert "Finished stub for urbansim" in result.stdout