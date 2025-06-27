import os
import hashlib
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def _validate_file_path(file_path: str) -> Optional[str]:
    """
    Validate and normalize file path.

    Args:
        file_path: Path to validate

    Returns:
        Absolute path if file exists, None otherwise
    """
    if not file_path:
        logger.warning("Empty file path provided for validation")
        return None

    abs_path = os.path.abspath(file_path)
    logger.debug(f"Validating file path: {abs_path}")
    if not os.path.exists(abs_path):
        logger.warning(f"File does not exist: {abs_path}")
        return None

    return abs_path

def _get_relative_path(file_path: str, base_path: str = None) -> str:
    """Get path relative to base directory for consistent storage."""
    abs_path = os.path.abspath(file_path)
    base_path = base_path or os.getcwd()
    try:
        return os.path.relpath(abs_path, base_path)
    except ValueError:
        logger.warning(
            f"Could not create relative path for {abs_path} relative to {base_path}"
        )
        return abs_path

def _calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA256 hash of a file with improved error handling."""
    abs_file_path = _validate_file_path(file_path)
    if not abs_file_path:
        return None

    try:
        sha256_hash = hashlib.sha256()
        with open(abs_file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except (IOError, OSError) as e:
        logger.warning(f"Could not calculate hash for {abs_file_path}: {e}")
        return None

def _load_metadata(file_path: str) -> Dict[str, Any]:
    """Load metadata from a JSON file located in the same directory as the file."""
    metadata_file = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}.metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load metadata from {metadata_file}: {e}")
    return {}
