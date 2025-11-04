import os
from pilates.utils.provenance import find_project_root
from pilates.utils.settings_helper import get as get_setting


def get_beam_source_dir(settings):
    """
    Robustly resolve the BEAM source directory, matching logic in pilates/beam/preprocessor.py.
    Returns the absolute path to the BEAM source directory for the given region.
    """
    region = get_setting(settings, "run.region")
    # Find the project root by searching upwards for 'pilates' or '.git'
    pilates_root = find_project_root()
    if pilates_root is None:
        pilates_root = os.path.realpath(os.getcwd())
    beam_source_dir = os.path.abspath(
        os.path.join(
            pilates_root,
            "pilates",
            "beam",
            "production",
            region,
        )
    )
    # If not found, try with "sources/PILATES" in the path (for symlinked or alternate layouts)
    if not os.path.exists(beam_source_dir):
        alt_root = os.path.join(pilates_root, "sources", "PILATES")
        alt_beam_source_dir = os.path.abspath(
            os.path.join(
                alt_root,
                "pilates",
                "beam",
                "production",
                region,
            )
        )
        if os.path.exists(alt_beam_source_dir):
            beam_source_dir = alt_beam_source_dir
    return beam_source_dir
