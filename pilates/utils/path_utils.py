import os
from typing import Iterable, Optional, Tuple


def find_project_root(
    start_path: Optional[str] = None,
    markers: Iterable[str] = ("pilates", ".git"),
) -> Optional[str]:
    """
    Search upwards from start_path for a directory containing one of the marker
    directories/files. Returns the absolute path to the project root, or None.
    """
    if start_path is None:
        start_path = os.getcwd()
    current = os.path.abspath(start_path)
    marker_tuple: Tuple[str, ...] = tuple(markers)

    while True:
        for marker in marker_tuple:
            if os.path.exists(os.path.join(current, marker)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None
