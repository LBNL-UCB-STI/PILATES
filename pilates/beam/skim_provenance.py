"""Skim artifact inspection and portable provenance sidecar helpers.

Provides HDF5/OMX and Zarr skim inspection plus portable JSON sidecar
generation (`<skims>.provenance.json`) so skim lineage and structural metadata
can travel across Consist databases or standalone files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from consist.core.admission import admission_file_identity

logger = logging.getLogger(__name__)

SKIM_PROVENANCE_SCHEMA_VERSION = "pilates.skim_provenance.v1"

_KNOWN_TIME_PERIODS = ("EA", "AM", "MD", "PM", "EV")


def skim_provenance_sidecar_path(skim_path: Path | str) -> Path:
    """Return the co-located provenance sidecar path for ``skim_path``."""
    p = Path(skim_path)
    return p.parent / f"{p.name}.provenance.json"


def _decode_attr_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if hasattr(val, "item") and callable(val.item):
        try:
            item_val = val.item()
            if isinstance(item_val, (bytes, bytearray)):
                return item_val.decode("utf-8", errors="replace")
            return item_val
        except Exception:
            pass
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def _compute_zone_order_sha256(zone_ids: list[Any]) -> str | None:
    if not zone_ids:
        return None
    canonical = ",".join(
        _decode_attr_value(z) if isinstance(z, (bytes, bytearray)) else str(z)
        for z in zone_ids
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _infer_time_periods_from_matrix_names(matrix_names: list[str]) -> list[str]:
    found: set[str] = set()
    for name in matrix_names:
        upper = name.upper()
        for period in _KNOWN_TIME_PERIODS:
            if upper.endswith(f"__{period}") or upper.endswith(f"_{period}"):
                found.add(period)
    return [p for p in _KNOWN_TIME_PERIODS if p in found]


def _compute_directory_identity(directory: Path) -> tuple[str, int]:
    """Compute a deterministic manifest digest and byte size for a directory."""
    manifest_hasher = hashlib.sha256()
    total_bytes = 0
    files = sorted(
        (p for p in directory.rglob("*") if p.is_file()),
        key=lambda item: item.relative_to(directory).as_posix(),
    )
    for file_path in files:
        rel = file_path.relative_to(directory).as_posix()
        file_hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                file_hasher.update(chunk)
                total_bytes += len(chunk)
        manifest_hasher.update(f"{rel}:{file_hasher.hexdigest()}\n".encode("utf-8"))
    return f"sha256:dir_manifest:{manifest_hasher.hexdigest()}", total_bytes


def _extract_root_attr(attrs: Mapping[str, Any], key: str) -> str | None:
    if key in attrs:
        decoded = _decode_attr_value(attrs[key])
        return str(decoded) if decoded is not None else None
    return None


def _inspect_omx_artifact(path: Path) -> dict[str, Any]:
    data_model_version: str | None = None
    skim_matrix_version: str | None = None
    beam_run_id: str | None = None
    created_at: str | None = None
    zone_ids: list[Any] = []
    zone_count: int | None = None
    matrix_names: list[str] = []

    try:
        import h5py

        with h5py.File(path, "r") as h5f:
            attrs = dict(h5f.attrs)
            data_model_version = _extract_root_attr(attrs, "data_model_version")
            skim_matrix_version = _extract_root_attr(attrs, "skim_matrix_version")
            beam_run_id = _extract_root_attr(attrs, "beam_run_id")
            created_at = _extract_root_attr(attrs, "created_at")

            data_group = h5f.get("data") or h5f.get("DATA")
            if data_group is not None and hasattr(data_group, "keys"):
                matrix_names = sorted(str(k) for k in data_group.keys())
                if matrix_names:
                    first_ds = data_group[matrix_names[0]]
                    if hasattr(first_ds, "shape") and len(first_ds.shape) >= 1:
                        zone_count = int(first_ds.shape[0])

            lookup_group = h5f.get("lookup") or h5f.get("LOOKUP")
            if lookup_group is not None and hasattr(lookup_group, "keys"):
                lookup_ds = lookup_group.get("zone_number")
                if lookup_ds is None:
                    lookup_keys = sorted(str(k) for k in lookup_group.keys())
                    if lookup_keys:
                        lookup_ds = lookup_group[lookup_keys[0]]
                if lookup_ds is not None:
                    zone_ids = [_decode_attr_value(v) for v in lookup_ds[...].tolist()]
            elif "original_zone_ids" in attrs:
                raw_zones = attrs["original_zone_ids"]
                if hasattr(raw_zones, "tolist"):
                    zone_ids = [_decode_attr_value(v) for v in raw_zones.tolist()]
                elif isinstance(raw_zones, (list, tuple)):
                    zone_ids = [_decode_attr_value(v) for v in raw_zones]
    except Exception as exc:
        logger.debug("Unable to inspect OMX HDF5 structure at %s: %s", path, exc)

    if not zone_ids:
        mapping_file = Path(f"{path}.mapping")
        if mapping_file.is_file():
            try:
                mapping_data = json.loads(mapping_file.read_text(encoding="utf-8"))
                if isinstance(mapping_data, dict):
                    sorted_pairs = sorted(
                        mapping_data.items(), key=lambda kv: int(kv[1])
                    )
                    zone_ids = [str(k) for k, _ in sorted_pairs]
            except Exception as exc:
                logger.debug(
                    "Unable to parse OMX mapping sidecar at %s: %s", mapping_file, exc
                )

    if zone_ids:
        zone_count = len(zone_ids)

    return {
        "data_model_version": data_model_version,
        "skim_matrix_version": skim_matrix_version,
        "beam_run_id": beam_run_id,
        "created_at": created_at,
        "zone_count": zone_count,
        "zone_order_sha256": _compute_zone_order_sha256(zone_ids),
        "matrix_count": len(matrix_names),
        "matrix_names": matrix_names,
        "time_periods": _infer_time_periods_from_matrix_names(matrix_names),
    }


def _inspect_zarr_artifact(path: Path) -> dict[str, Any]:
    data_model_version: str | None = None
    skim_matrix_version: str | None = None
    beam_run_id: str | None = None
    created_at: str | None = None
    zone_ids: list[Any] = []
    zone_count: int | None = None
    matrix_names: list[str] = []
    time_periods: list[str] = []

    try:
        import xarray as xr

        with xr.open_zarr(path) as ds:
            attrs = dict(ds.attrs)
            data_model_version = _extract_root_attr(attrs, "data_model_version")
            skim_matrix_version = _extract_root_attr(attrs, "skim_matrix_version")
            beam_run_id = _extract_root_attr(attrs, "beam_run_id")
            created_at = _extract_root_attr(attrs, "created_at")

            matrix_names = sorted(str(k) for k in ds.data_vars.keys())
            if "time_period" in ds.coords:
                time_periods = [
                    str(_decode_attr_value(v))
                    for v in ds.coords["time_period"].values.tolist()
                ]
            if "original_zone_ids" in attrs:
                raw_zones = attrs["original_zone_ids"]
                if hasattr(raw_zones, "tolist"):
                    zone_ids = [_decode_attr_value(v) for v in raw_zones.tolist()]
                elif isinstance(raw_zones, (list, tuple)):
                    zone_ids = [_decode_attr_value(v) for v in raw_zones]
            elif "otaz" in ds.coords:
                zone_ids = [
                    _decode_attr_value(v) for v in ds.coords["otaz"].values.tolist()
                ]
            elif "origin" in ds.coords:
                zone_ids = [
                    _decode_attr_value(v) for v in ds.coords["origin"].values.tolist()
                ]
            if zone_ids:
                zone_count = len(zone_ids)
    except Exception as exc:
        logger.debug("Unable to inspect Zarr dataset at %s via xarray: %s", path, exc)
        zattrs_path = path / ".zattrs"
        if zattrs_path.is_file():
            try:
                attrs = json.loads(zattrs_path.read_text(encoding="utf-8"))
                if isinstance(attrs, dict):
                    data_model_version = _extract_root_attr(attrs, "data_model_version")
                    skim_matrix_version = _extract_root_attr(
                        attrs, "skim_matrix_version"
                    )
                    beam_run_id = _extract_root_attr(attrs, "beam_run_id")
                    created_at = _extract_root_attr(attrs, "created_at")
                    raw_zones = attrs.get("original_zone_ids")
                    if isinstance(raw_zones, list):
                        zone_ids = [_decode_attr_value(v) for v in raw_zones]
                        zone_count = len(zone_ids)
            except Exception:
                pass

    if not time_periods and matrix_names:
        time_periods = _infer_time_periods_from_matrix_names(matrix_names)

    return {
        "data_model_version": data_model_version,
        "skim_matrix_version": skim_matrix_version,
        "beam_run_id": beam_run_id,
        "created_at": created_at,
        "zone_count": zone_count,
        "zone_order_sha256": _compute_zone_order_sha256(zone_ids),
        "matrix_count": len(matrix_names),
        "matrix_names": matrix_names,
        "time_periods": time_periods,
    }


def inspect_skim_artifact(path: Path | str) -> dict[str, Any]:
    """Inspect a skim artifact (.omx, .zarr, or .csv.gz) and return metadata + identity.

    Gracefully handles legacy OMX/Zarr files that lack HDF5/Zarr root provenance
    attributes by returning ``None`` for absent root attributes while still
    extracting structural zone/matrix metadata and cryptographic identity.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Skim artifact does not exist: {p}")

    if p.is_dir() or p.suffix.lower() == ".zarr":
        skim_format = "zarr"
        identity_digest, size_bytes = _compute_directory_identity(p)
        details = _inspect_zarr_artifact(p)
    elif p.name.lower().endswith(".csv.gz"):
        skim_format = "csv_gz"
        identity_digest = str(admission_file_identity(p))
        size_bytes = p.stat().st_size
        details = {
            "data_model_version": None,
            "skim_matrix_version": None,
            "beam_run_id": None,
            "created_at": None,
            "zone_count": None,
            "zone_order_sha256": None,
            "matrix_count": None,
            "matrix_names": [],
            "time_periods": [],
        }
    else:
        skim_format = "omx"
        identity_digest = str(admission_file_identity(p))
        size_bytes = p.stat().st_size
        details = _inspect_omx_artifact(p)

    has_embedded_provenance = any(
        details.get(k) is not None
        for k in (
            "data_model_version",
            "skim_matrix_version",
            "beam_run_id",
            "created_at",
        )
    )

    return {
        "skim_format": skim_format,
        "identity": identity_digest,
        "size_bytes": size_bytes,
        "has_embedded_provenance": has_embedded_provenance,
        **details,
    }


def write_skim_provenance_sidecar(
    skim_path: Path | str,
    *,
    producer_step: str,
    run_id: str | None = None,
    scenario: str | None = None,
    region: str | None = None,
    year: int | None = None,
    iteration: int | None = None,
    tracker_db_path: Path | str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Inspect ``skim_path`` and write ``<skim_path>.provenance.json``."""
    p = Path(skim_path)
    inspection = inspect_skim_artifact(p)
    skim_metadata: dict[str, Any] = {
        "data_model_version": inspection["data_model_version"],
        "skim_matrix_version": inspection["skim_matrix_version"],
        "beam_run_id": inspection["beam_run_id"],
        "created_at": inspection["created_at"],
        "zone_count": inspection["zone_count"],
        "zone_order_sha256": inspection["zone_order_sha256"],
        "matrix_count": inspection["matrix_count"],
        "matrix_names": inspection["matrix_names"],
        "time_periods": inspection["time_periods"],
    }
    if extra_metadata:
        skim_metadata.update(dict(extra_metadata))

    payload: dict[str, Any] = {
        "schema_version": SKIM_PROVENANCE_SCHEMA_VERSION,
        "artifact_name": p.name,
        "skim_format": inspection["skim_format"],
        "identity": inspection["identity"],
        "size_bytes": inspection["size_bytes"],
        "producer": {
            "step": producer_step,
            "run_id": run_id,
            "scenario": scenario,
            "region": region,
            "year": year,
            "iteration": iteration,
            "tracker_db_path": (
                str(tracker_db_path) if tracker_db_path is not None else None
            ),
        },
        "skim_metadata": skim_metadata,
    }

    sidecar_path = skim_provenance_sidecar_path(p)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sidecar_path.parent / f".tmp_{sidecar_path.name}.{os.getpid()}"
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(sidecar_path)
    return sidecar_path, payload


def read_skim_provenance_sidecar(sidecar_path: Path | str) -> dict[str, Any]:
    """Read and validate a ``pilates.skim_provenance.v1`` JSON sidecar."""
    p = Path(sidecar_path)
    if not p.is_file():
        raise FileNotFoundError(f"Skim provenance sidecar does not exist: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Skim provenance sidecar must be a JSON object: {p}")
    schema_version = raw.get("schema_version")
    if schema_version != SKIM_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported skim provenance schema_version {schema_version!r} in {p}; "
            f"expected {SKIM_PROVENANCE_SCHEMA_VERSION!r}."
        )
    return raw
