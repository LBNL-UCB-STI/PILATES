"""
Utility functions for handling different data artifact formats (e.g., Zarr, NetCDF).
Includes helpers for computing hashes, extracting metadata, and copying artifacts.
"""

import hashlib
import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Optional, Any

import xarray as xr
import zarr

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> Optional[str]:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest of file hash, or None if file doesn't exist.
    """
    if not file_path.exists() or not file_path.is_file():
        return None

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64k chunks to handle large files efficiently
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compute_zarr_chunk_manifest(zarr_path: Path) -> Dict[str, str]:
    """
    Compute manifest of all chunk files in a Zarr store, mapping relative paths to their SHA-256 hashes.

    Args:
        zarr_path: Path to the Zarr store (directory).

    Returns:
        Dict mapping relative chunk paths to their hashes.
    """
    chunk_manifest = {}
    if not zarr_path.is_dir():
        logger.warning(
            f"Zarr path {zarr_path} is not a directory. Cannot compute chunk manifest."
        )
        return chunk_manifest

    # Find all chunk files (e.g., 0.0.0, 1.2.3, etc.)
    # Zarr chunk files typically have names consisting only of digits and dots.
    for chunk_file in zarr_path.rglob("*"):
        if chunk_file.is_file() and not chunk_file.name.startswith("."):
            name_parts = chunk_file.name.split(".")
            # A heuristic to identify Zarr chunk files
            if all(part.isdigit() for part in name_parts):
                relative_path = str(chunk_file.relative_to(zarr_path))
                chunk_hash = compute_file_hash(chunk_file)
                if chunk_hash:
                    chunk_manifest[relative_path] = chunk_hash
    return chunk_manifest


def get_zarr_metadata(zarr_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a Zarr store.

    Args:
        zarr_path: Path to the Zarr store.

    Returns:
        Dict with 'n_variables', 'n_chunks', 'total_size_mb'.
    """
    if not zarr_path.is_dir():
        logger.warning(
            f"Zarr path {zarr_path} is not a directory. Cannot get Zarr metadata."
        )
        return {"n_variables": 0, "n_chunks": 0, "total_size_mb": 0.0}

    try:
        with xr.open_zarr(zarr_path) as ds:  # Open with xarray
            n_variables = len(ds.data_vars)  # Count xarray data variables

        # Count chunk files using the heuristic from compute_zarr_chunk_manifest
        n_chunks = sum(
            1
            for f in zarr_path.rglob("*")
            if f.is_file()
            and not f.name.startswith(".")
            and all(part.isdigit() for part in f.name.split("."))
        )

        # Calculate total size
        total_size_mb = sum(
            f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)

        return {
            "n_variables": n_variables,
            "n_chunks": n_chunks,
            "total_size_mb": round(total_size_mb, 4),
        }
    except Exception as e:
        logger.error(f"Error reading Zarr metadata from {zarr_path}: {e}")
        return {"n_variables": 0, "n_chunks": 0, "total_size_mb": 0.0}


def get_netcdf_metadata(netcdf_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a NetCDF file.

    Args:
        netcdf_path: Path to the NetCDF file.

    Returns:
        Dict with 'n_variables', 'file_size_mb'.
    """
    if not netcdf_path.is_file():
        logger.warning(
            f"NetCDF path {netcdf_path} is not a file. Cannot get NetCDF metadata."
        )
        return {"n_variables": 0, "file_size_mb": 0.0}

    try:
        with xr.open_dataset(netcdf_path, decode_times=False) as ds:
            n_variables = len(ds.data_vars)
            file_size_mb = netcdf_path.stat().st_size / (1024 * 1024)
            return {
                "n_variables": n_variables,
                "file_size_mb": round(file_size_mb, 4),
            }
    except Exception as e:
        logger.error(f"Error reading NetCDF metadata from {netcdf_path}: {e}")
        return {"n_variables": 0, "file_size_mb": 0.0}


def get_artifact_metadata(path: Path, artifact_format: str) -> Dict[str, Any]:
    """
    Get metadata for a data artifact based on its format.

    Args:
        path: Path to the data artifact (file or directory).
        artifact_format: The format of the artifact ('zarr', 'netcdf').

    Returns:
        A dictionary containing relevant metadata.
    """
    if artifact_format.lower() == "zarr":
        return get_zarr_metadata(path)
    elif artifact_format.lower() == "netcdf":
        return get_netcdf_metadata(path)
    else:
        logger.warning(
            f"Unsupported artifact format: {artifact_format}. Returning empty metadata."
        )
        return {}


def copy_artifact(source_path: Path, destination_path: Path, artifact_format: str):
    """
    Copies a data artifact from source to destination based on its format.

    For 'zarr' format, it performs a directory copy. For all other formats,
    it performs a single-file copy.

    Args:
        source_path: The path to the source artifact.
        destination_path: The path where the artifact should be copied.
        artifact_format: The format of the artifact ('zarr', 'netcdf', etc.).

    Raises:
        FileNotFoundError: If the source path does not exist.
        TypeError: If the source path type (file/dir) does not match the artifact format.
        Exception: For other copying errors.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source artifact not found: {source_path}")

    # Ensure parent directory exists for the destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if artifact_format.lower() == "zarr":
        if not source_path.is_dir():
            raise TypeError(
                f"Source path for Zarr artifact must be a directory: {source_path}"
            )
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        logger.info(f"Copied Zarr store from {source_path} to {destination_path}")
    else:  # Assume any other format is a single file
        if not source_path.is_file():
            raise TypeError(
                f"Source path for file-based artifact '{artifact_format}' must be a file: {source_path}"
            )
        shutil.copy2(source_path, destination_path)
        logger.info(
            f"Copied {artifact_format.upper()} file from {source_path} to {destination_path}"
        )
