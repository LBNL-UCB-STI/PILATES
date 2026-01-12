import os
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from pilates.utils import consist_runtime as cr

if TYPE_CHECKING:
    from pilates.generic.records import RecordStore
    from pilates.workspace import Workspace


def artifact_to_path(
    value: Any, workspace: Optional["Workspace"] = None
) -> Optional[str]:
    """
    Resolve an artifact-like object or path into a concrete filesystem path.

    Parameters
    ----------
    value : object
        Artifact-like object or path value to resolve.
    workspace : Workspace, optional
        Workspace used to resolve relative paths.

    Returns
    -------
    str or None
        Resolved filesystem path, or None if value is None.
    """
    if value is None:
        return None
    path = getattr(value, "path", None) or getattr(value, "uri", None) or value
    if isinstance(path, Path):
        path = os.fspath(path)
    elif isinstance(path, os.PathLike):
        path = os.fspath(path)
    if isinstance(path, str) and workspace is not None and not os.path.isabs(path):
        if "://" not in path:
            return os.path.join(workspace.full_path, path)
    return path


def update_coupler_from_beam_outputs(
    output_store: Optional["RecordStore"],
    coupler: Any,
    workspace: "Workspace",
) -> None:
    """
    Log and propagate BEAM output artifacts into the workflow coupler.

    Parameters
    ----------
    output_store : RecordStore
        Output store from BEAM postprocessing.
    coupler : object
        Consist coupler or compatible interface.
    workspace : Workspace
        Workspace used to resolve output paths.
    """
    if not output_store:
        return
    linkstats_record = None
    beam_plans_record = None
    for record in output_store.all_records():
        if record.short_name == "zarr_skims":
            zarr_path = artifact_to_path(record.file_path, workspace)
            if zarr_path and os.path.exists(zarr_path):
                log_and_set_output(
                    key="zarr_skims",
                    path=zarr_path,
                    description="Zarr skims updated with BEAM outputs",
                    coupler=coupler,
                )
        elif record.short_name == "final_skims_omx":
            omx_path = artifact_to_path(record.file_path, workspace)
            if omx_path and os.path.exists(omx_path):
                log_and_set_output(
                    key="final_skims_omx",
                    path=omx_path,
                    description="Final skims OMX for downstream models",
                    coupler=coupler,
                )
        elif record.short_name and record.short_name.startswith("linkstats"):
            linkstats_record = _select_beam_output_record(
                linkstats_record, record, "linkstats"
            )
        elif record.short_name and record.short_name.startswith("beam_plans_out"):
            beam_plans_record = _select_beam_output_record(
                beam_plans_record, record, "beam_plans_out"
            )

    _log_and_set_beam_record(
        linkstats_record,
        key="linkstats",
        description="BEAM linkstats output for downstream runs",
        coupler=coupler,
        workspace=workspace,
    )
    _log_and_set_beam_record(
        beam_plans_record,
        key="beam_plans_out",
        description="BEAM plans output for downstream runs",
        coupler=coupler,
        workspace=workspace,
    )


def _select_beam_output_record(
    current: Optional[Any],
    candidate: Any,
    base_key: str,
) -> Optional[Any]:
    """
    Choose the most recent BEAM output record for a base key.

    Parameters
    ----------
    current : object, optional
        Currently selected record.
    candidate : object
        Candidate record to compare.
    base_key : str
        Base short_name prefix (e.g., ``linkstats``).

    Returns
    -------
    object or None
        Selected record with the highest (year, iteration) rank.
    """
    candidate_rank = _beam_record_rank(candidate, base_key)
    if candidate_rank is None:
        return current
    if current is None:
        return candidate
    current_rank = _beam_record_rank(current, base_key)
    if current_rank is None or candidate_rank > current_rank:
        return candidate
    return current


def _beam_record_rank(record: Any, base_key: str) -> Optional[tuple]:
    """
    Return a sortable (year, iteration) rank for a BEAM record short_name.

    Parameters
    ----------
    record : object
        Record with a ``short_name`` attribute.
    base_key : str
        Base short_name prefix.

    Returns
    -------
    tuple or None
        (year, iteration) rank, or None when the record cannot be ranked.
    """
    short_name = getattr(record, "short_name", None)
    if not short_name:
        return None
    if short_name == base_key:
        return (0, 0)
    if not short_name.startswith(f"{base_key}_"):
        return None
    suffix = short_name[len(base_key) + 1 :]
    parts = suffix.split("_")
    if len(parts) < 2:
        return None
    if parts[-1].startswith("sub"):
        return None
    try:
        year = int(parts[-2])
        iteration = int(parts[-1])
    except ValueError:
        return None
    return (year, iteration)


def _log_and_set_beam_record(
    record: Optional[Any],
    *,
    key: str,
    description: str,
    coupler: Any,
    workspace: "Workspace",
) -> None:
    """
    Log a BEAM output record and set it on the coupler.

    Parameters
    ----------
    record : object, optional
        Record describing the output artifact.
    key : str
        Coupler key to set.
    description : str
        Description used in provenance logging.
    coupler : object
        Consist coupler or compatible interface.
    workspace : Workspace
        Workspace used to resolve output paths.
    """
    if record is None:
        return
    output_path = artifact_to_path(record.file_path, workspace)
    if not output_path or not os.path.exists(output_path):
        return
    log_and_set_output(
        key=key,
        path=output_path,
        description=description,
        coupler=coupler,
    )


def clean_expected_outputs(outputs: dict) -> dict:
    """
    Filter out expected outputs that resolved to None.

    Parameters
    ----------
    outputs : dict
        Mapping of expected outputs to filter.

    Returns
    -------
    dict
        Outputs with None-valued entries removed.
    """
    return {key: value for key, value in outputs.items() if value is not None}


def set_coupler_from_artifact(
    coupler: Any,
    key: str,
    artifact: Optional[Any],
    fallback: Optional[str] = None,
) -> None:
    """
    Set a coupler key from an artifact, falling back to a raw path if needed.

    Parameters
    ----------
    coupler : object
        Consist coupler or compatible interface.
    key : str
        Coupler key to set.
    artifact : object
        Artifact-like object returned by logging helpers.
    fallback : str, optional
        Path to set if artifact is None.
    """
    if artifact is None and fallback is None:
        return
    set_from_artifact = getattr(coupler, "set_from_artifact", None)
    if callable(set_from_artifact):
        if artifact is None:
            set_from_artifact(key, fallback)
        else:
            set_from_artifact(key, artifact)
        return
    coupler.set(key, artifact or fallback)


def log_and_set_output(
    *,
    key: str,
    path: str,
    description: str,
    coupler: Any,
) -> None:
    """
    Log an output path and set it on the coupler.

    Parameters
    ----------
    key : str
        Coupler key to set.
    path : str
        Output path to log.
    description : str
        Description used in provenance logging.
    coupler : object
        Consist coupler or compatible interface.
    """
    artifact = cr.log_output(path, key=key, description=description)
    set_coupler_from_artifact(coupler, key, artifact, fallback=path)
