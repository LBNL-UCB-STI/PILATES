from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

from pilates.generic.records import FileRecord, RecordStore
from pilates.utils.io import locate_beam_file
from pilates.workflows.artifact_keys import (
    BEAM_HOUSEHOLDS_IN,
    BEAM_PERSONS_IN,
    BEAM_PLANS_IN,
)

logger = logging.getLogger(__name__)


def default_beam_exchange_scenario_folder(settings: Any, workspace: Any) -> str:
    base_input_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
    )
    return os.path.join(base_input_dir, settings.beam.scenario_folder)


def config_beam_exchange_scenario_folder(
    settings: Any, workspace: Any
) -> Optional[str]:
    """
    Parse the generated BEAM config for an exchange.scenario.folder override.
    """
    base_input_dir = os.path.join(
        workspace.get_beam_mutable_data_dir(),
        settings.run.region,
    )
    default_folder = default_beam_exchange_scenario_folder(settings, workspace)
    config_path = os.path.join(base_input_dir, settings.beam.config)
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r") as config_file:
            for raw_line in config_file:
                line = raw_line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                key, value = [part.strip() for part in line.split("=", 1)]
                # In BEAM HOCON this is typically nested under beam.exchange.scenario:
                #   folder = ${beam.inputDirectory}"/urbansim/2018"
                if key != "folder":
                    continue
                if "${beam.inputDirectory}" not in value:
                    continue

                resolved = value.replace("${beam.inputDirectory}", base_input_dir)
                resolved = resolved.replace('"', "")
                resolved = os.path.normpath(resolved)
                if resolved and os.path.normpath(resolved) != os.path.normpath(
                    default_folder
                ):
                    return resolved
    except Exception as exc:
        logger.warning(
            "[BEAM Preprocessor] Could not parse exchange.scenario.folder from %s: %s. "
            "Falling back to settings.beam.scenario_folder.",
            config_path,
            exc,
        )

    return None


def resolve_beam_exchange_scenario_folder(settings: Any, workspace: Any) -> str:
    """
    Resolve the effective BEAM exchange folder for staged write operations.
    """
    return config_beam_exchange_scenario_folder(
        settings, workspace
    ) or default_beam_exchange_scenario_folder(settings, workspace)


def beam_exchange_scenario_folder_candidates(
    settings: Any, workspace: Any
) -> List[str]:
    """
    Return candidate exchange folders in operator-facing precedence order.
    """
    default_folder = default_beam_exchange_scenario_folder(settings, workspace)
    candidates = [default_folder]
    config_folder = config_beam_exchange_scenario_folder(settings, workspace)
    if config_folder is not None and all(
        os.path.normpath(config_folder) != os.path.normpath(existing)
        for existing in candidates
    ):
        candidates.append(config_folder)
    return candidates


def beam_exchange_format_candidates(preferred_format: Optional[str]) -> List[str]:
    candidates: List[str] = []
    if preferred_format:
        candidates.append(preferred_format)
    for fallback in ("parquet", "csv", "csv.gz"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def locate_existing_beam_exchange_input(
    beam_scenario_folder: str,
    stem: str,
    preferred_format: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    for candidate_format in beam_exchange_format_candidates(preferred_format):
        if candidate_format == "csv.gz":
            path = os.path.join(beam_scenario_folder, f"{stem}.csv.gz")
        elif candidate_format == "csv":
            path = os.path.join(beam_scenario_folder, f"{stem}.csv")
        else:
            path = locate_beam_file(beam_scenario_folder, stem, candidate_format)
        if path and os.path.exists(path):
            return path, candidate_format
    return None, None


def register_existing_beam_exchange_inputs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    required_keys: Optional[Dict[str, str]] = None,
) -> RecordStore:
    """
    Register canonical BEAM scenario inputs already present in exchange folders.
    """
    keys = required_keys or {
        BEAM_PLANS_IN: "plans",
        BEAM_HOUSEHOLDS_IN: "households",
        BEAM_PERSONS_IN: "persons",
    }

    file_format = getattr(getattr(settings, "activitysim", None), "file_format", None)
    if not file_format:
        file_format = "parquet"

    candidate_folders = beam_exchange_scenario_folder_candidates(settings, workspace)
    current_year = getattr(state, "current_year", None)
    current_inner_iter = getattr(state, "current_inner_iter", None)
    folder_errors: List[str] = []

    for idx, beam_scenario_folder in enumerate(candidate_folders):
        records: List[FileRecord] = []
        unresolved: List[str] = []
        for short_name, stem in keys.items():
            path, resolved_format = locate_existing_beam_exchange_input(
                beam_scenario_folder,
                stem,
                file_format,
            )
            if path and resolved_format:
                records.append(
                    FileRecord(
                        file_path=path,
                        short_name=short_name,
                        description=f"Existing BEAM scenario input: {stem} ({resolved_format})",
                        year=current_year,
                        iteration=current_inner_iter,
                    ),
                )
                continue
            unresolved.append(
                f"{stem}.[{'|'.join(beam_exchange_format_candidates(file_format))}]"
            )

        if not unresolved:
            if idx > 0:
                logger.info(
                    "[BEAM Preprocessor] YAML scenario_folder was missing required inputs; "
                    "using fallback exchange.scenario.folder from BEAM config: %s",
                    beam_scenario_folder,
                )
            return RecordStore(recordList=records)

        folder_errors.append(f"{beam_scenario_folder}: {', '.join(unresolved)}")

    raise FileNotFoundError(
        "Missing default BEAM scenario inputs in exchange folders "
        f"{'; '.join(folder_errors)}"
    )
