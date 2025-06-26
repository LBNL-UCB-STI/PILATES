# Robustly resolve the BEAM source directory, matching logic in pilates/beam/preprocessor.py
import os

# This file expects a 'settings' dict to be provided in the calling context.
# Example usage:
#   from workflow_state import get_beam_source_dir
#   beam_source_dir = get_beam_source_dir(settings)

def get_beam_source_dir(settings):
    """
    Robustly resolve the BEAM source directory, matching logic in pilates/beam/preprocessor.py.
    Returns the absolute path to the BEAM source directory for the given region.
    """
    import os

    region = settings["region"]
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
