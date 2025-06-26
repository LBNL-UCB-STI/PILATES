# Robustly resolve the BEAM source directory, matching logic in pilates/beam/preprocessor.py
import os

def find_project_root(start_path=None, markers=("pilates", ".git")):
    """
    Search upwards from start_path for a directory containing one of the marker directories/files.
    Returns the absolute path to the project root, or None if not found.
    """
    if start_path is None:
        start_path = os.getcwd()
    current = os.path.abspath(start_path)
    while True:
        for marker in markers:
            if os.path.exists(os.path.join(current, marker)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None

def get_beam_source_dir(settings):
    """
    Robustly resolve the BEAM source directory, matching logic in pilates/beam/preprocessor.py.
    Returns the absolute path to the BEAM source directory for the given region.
    """
    region = settings["region"]
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
