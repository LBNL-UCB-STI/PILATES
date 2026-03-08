"""
run.py

Main entrypoint and workflow orchestrator for PILATES simulations.

This module:
- Parses settings and initializes workflow state.
- Initializes the Consist Tracker and Scenario Context.
- Executes the multi-stage simulation loop using the Scenario/Step API.
- Manages provenance for the critical "Data Initialization" step to link
  immutable inputs to the mutable workspace.
"""

import warnings
from datetime import datetime
import os
import logging
import sys
import shutil
import socket
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple, Mapping, Sequence

from pilates.workspace import Workspace
from pilates.generic.records import FileRecord, RecordStore, sanitize_artifact_key
from pilates.generic.initialization import (
    Initialization,
    build_bootstrap_artifact_summary,
)
from pilates.utils.formatting import formatted_print
from pilates.utils.io import parse_args_and_settings
from pilates.utils import consist_runtime as cr
from pilates.utils.consist_config import (
    build_scenario_consist_kwargs,
    build_step_consist_kwargs,
)
from pilates.utils.consist_db_snapshot import (
    ConsistDbSnapshotManager,
    mirror_consist_db_to_archive,
    resolve_consist_db_paths,
    restore_local_consist_db_from_snapshot,
    seed_local_consist_db_from_shared,
    snapshot_latest_dir,
)
from pilates.utils.coupler_helpers import flush_archive_queue, stop_archive_worker
from pilates.utils.restart_bundle import (
    build_restart_bundle_manifest,
    manifest_entries_to_local_artifacts,
    write_restart_bundle_manifest,
)
from pilates.atlas.inputs import atlas_static_input_relpaths
from pilates.activitysim.preprocessor import required_asim_config_dirs
from pilates.urbansim.postprocessor import get_usim_datastore_fname
from pilates.workflows.coupler_schema import build_coupler_schema
from pilates.workflows.catalog import schema_step_names, enabled_schema_step_models
from pilates.workflows.stages import (
    run_land_use_stage,
    run_postprocessing_stage,
    run_supply_demand_stage,
    run_vehicle_ownership_stage,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
from workflow_state import WorkflowState

from pilates.workflows.steps import (
    StepOutputsHolder,
    make_activitysim_compile_step,
    make_activitysim_postprocess_step,
    make_activitysim_preprocess_step,
    make_activitysim_run_step,
    make_atlas_postprocess_step,
    make_atlas_preprocess_step,
    make_atlas_run_step,
    make_beam_postprocess_step,
    make_beam_full_skim_step,
    make_beam_preprocess_step,
    make_beam_run_step,
    make_urbansim_postprocess_step,
    make_urbansim_preprocess_step,
    make_urbansim_run_step,
    validate_workflow_step_contracts,
)
from consist.types import CacheOptions

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_SCENARIO_NAME_TEMPLATE = "{func_name}__y{year}__i{iteration}__phase_{phase}"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_scenario_id(settings: Any) -> str:
    run_cfg = getattr(settings, "run", None)
    candidates = [
        getattr(run_cfg, "scenario", None),
        getattr(run_cfg, "output_run_name", None),
    ]
    for candidate in candidates:
        text = str(candidate).strip() if candidate is not None else ""
        if text:
            return text
    return "unknown_scenario"


def _resolve_seed(settings: Any) -> Optional[int]:
    candidates = [
        getattr(getattr(settings, "activitysim", None), "random_seed", None),
        getattr(getattr(settings, "run", None), "seed", None),
        getattr(settings, "seed", None),
    ]
    for candidate in candidates:
        value = _coerce_int(candidate)
        if value is not None:
            return value
    return None


def _merge_tag_list(existing: Any, additions: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        for value in existing:
            text = str(value).strip()
            if text and text not in seen:
                merged.append(text)
                seen.add(text)
    for value in additions:
        text = str(value).strip()
        if text and text not in seen:
            merged.append(text)
            seen.add(text)
    return merged


def _facet_to_mapping(facet: Any) -> Dict[str, Any]:
    if isinstance(facet, Mapping):
        return dict(facet)
    model_dump = getattr(facet, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _merge_epoch_facet(
    *,
    existing: Any,
    scenario_id: str,
    seed: Optional[int],
    model: Optional[str],
    year: Optional[int],
    iteration: Optional[int],
) -> Dict[str, Any]:
    merged = _facet_to_mapping(existing)
    merged["scenario_id"] = scenario_id
    if seed is not None:
        merged["seed"] = seed
    if model:
        merged["model"] = model
    if year is not None:
        merged["year"] = year
    if iteration is not None:
        merged["iteration"] = iteration
    return merged


class _EpochTaggingScenarioProxy:
    """
    Wrapper around a Consist scenario that injects epoch metadata and parent linkage.
    """

    def __init__(self, scenario: Any, *, scenario_id: str, seed: Optional[int]) -> None:
        self._scenario = scenario
        self._scenario_id = scenario_id
        self._seed = seed
        self._activitysim_run_ids: Dict[Tuple[int, int], str] = {}
        self._activitysim_step_ids: Dict[Tuple[int, int], str] = {}
        self._beam_run_ids: Dict[Tuple[int, int], str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scenario, name)

    def _infer_model_name(self, run_kwargs: Mapping[str, Any]) -> Optional[str]:
        explicit = run_kwargs.get("model")
        if explicit:
            return str(explicit)
        fn = run_kwargs.get("fn")
        meta = getattr(fn, "__consist_step__", None)
        model_name = getattr(meta, "model", None)
        if model_name:
            return str(model_name)
        return None

    def _resolve_parent_run_id(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
    ) -> Optional[str]:
        if model_name is None or year is None or iteration is None:
            return None
        model_norm = model_name.lower()
        if model_norm == "beam" or model_norm.startswith("beam_"):
            key = (year, iteration)
            return self._activitysim_run_ids.get(key) or self._activitysim_step_ids.get(
                key
            )
        if (model_norm == "activitysim" or model_norm.startswith("activitysim_")) and iteration > 0:
            return self._beam_run_ids.get((year, iteration - 1))
        return None

    def _should_expect_parent(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
    ) -> bool:
        if model_name is None or year is None or iteration is None:
            return False
        model_norm = model_name.lower()
        if model_norm == "beam" or model_norm.startswith("beam_"):
            return True
        if (model_norm == "activitysim" or model_norm.startswith("activitysim_")) and iteration > 0:
            return True
        return False

    def _remember_run_id(
        self,
        *,
        model_name: Optional[str],
        year: Optional[int],
        iteration: Optional[int],
        run_id: Optional[str],
    ) -> None:
        if model_name is None or year is None or iteration is None or not run_id:
            return
        key = (year, iteration)
        model_norm = model_name.lower()
        if model_norm == "activitysim" or model_norm.startswith("activitysim_"):
            self._activitysim_step_ids[key] = run_id
            if model_norm in {"activitysim", "activitysim_run"}:
                self._activitysim_run_ids[key] = run_id
        elif model_norm in {"beam", "beam_run", "beam_full_skim"}:
            self._beam_run_ids[key] = run_id

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) > 2:
            return self._scenario.run(*args, **kwargs)

        run_kwargs: Dict[str, Any] = dict(kwargs)
        if len(args) >= 1 and "fn" not in run_kwargs:
            run_kwargs["fn"] = args[0]
        if len(args) == 2 and "name" not in run_kwargs:
            run_kwargs["name"] = args[1]

        model_name = self._infer_model_name(run_kwargs)
        year = _coerce_int(run_kwargs.get("year"))
        iteration = _coerce_int(run_kwargs.get("iteration"))
        if model_name and "model" not in run_kwargs:
            run_kwargs["model"] = model_name

        if not run_kwargs.get("parent_run_id"):
            resolved_parent = self._resolve_parent_run_id(
                model_name=model_name,
                year=year,
                iteration=iteration,
            )
            if resolved_parent:
                run_kwargs["parent_run_id"] = resolved_parent
            elif self._should_expect_parent(
                model_name=model_name,
                year=year,
                iteration=iteration,
            ):
                # Keep execution behavior unchanged when lineage hints are unavailable.
                logger.debug(
                    "[RunTagging] parent_run_id unavailable for model=%s year=%s iteration=%s; leaving unset.",
                    model_name,
                    year,
                    iteration,
                )

        tag_additions = [f"scenario_id:{self._scenario_id}"]
        if self._seed is not None:
            tag_additions.append(f"seed:{self._seed}")
        if model_name:
            tag_additions.append(f"model:{model_name}")
        if year is not None:
            tag_additions.append(f"year:{year}")
        if iteration is not None:
            tag_additions.append(f"iteration:{iteration}")
        run_kwargs["tags"] = _merge_tag_list(run_kwargs.get("tags"), tag_additions)
        run_kwargs["facet"] = _merge_epoch_facet(
            existing=run_kwargs.get("facet"),
            scenario_id=self._scenario_id,
            seed=self._seed,
            model=model_name,
            year=year,
            iteration=iteration,
        )

        result = self._scenario.run(**run_kwargs)
        run_id = str(getattr(getattr(result, "run", None), "id", "")).strip() or None
        self._remember_run_id(
            model_name=model_name,
            year=year,
            iteration=iteration,
            run_id=run_id,
        )
        return result


class _SchemaCoupler:
    """No-op coupler used to construct decorated step callables for schema introspection."""

    def get(self, _key: str, default: Optional[Any] = None) -> Any:
        return default

    def set(self, _key: str, _value: Any) -> None:
        return None

    def update(self, _mapping: Dict[str, Any]) -> None:
        return None

    def view(self, _namespace: str) -> "_SchemaCoupler":
        return self

    def declare_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None


def _resolve_cache_epoch(settings: Any) -> int:
    value = getattr(getattr(settings, "run", None), "cache_epoch", 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _build_schema_steps() -> List[Callable[..., Any]]:
    coupler = _SchemaCoupler()
    outputs_holder = StepOutputsHolder()
    step_factories: Dict[str, Callable[..., Any]] = {
        "urbansim_preprocess": make_urbansim_preprocess_step,
        "urbansim_run": make_urbansim_run_step,
        "urbansim_postprocess": make_urbansim_postprocess_step,
        "atlas_preprocess": make_atlas_preprocess_step,
        "atlas_run": make_atlas_run_step,
        "atlas_postprocess": make_atlas_postprocess_step,
        "activitysim_preprocess": make_activitysim_preprocess_step,
        "activitysim_compile": make_activitysim_compile_step,
        "activitysim_run": make_activitysim_run_step,
        "activitysim_postprocess": make_activitysim_postprocess_step,
        "beam_preprocess": make_beam_preprocess_step,
        "beam_run": make_beam_run_step,
        "beam_postprocess": make_beam_postprocess_step,
        "beam_full_skim": make_beam_full_skim_step,
    }
    ordered_steps = schema_step_names()
    missing_factories = [name for name in ordered_steps if name not in step_factories]
    if missing_factories:
        raise RuntimeError(
            "Missing schema step factories for: " + ", ".join(missing_factories)
        )
    return [
        step_factories[step_name](coupler=coupler, outputs_holder=outputs_holder)
        for step_name in ordered_steps
    ]


def _is_model_enabled(settings: Any, *, flag_attr: str, model_attr: str) -> bool:
    """
    Resolve whether a workflow model is enabled.

    Prefers precomputed flags from ``parse_args_and_settings`` and falls back to
    ``settings.run.models`` when flags are not present.
    """
    explicit_flag = getattr(settings, flag_attr, None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    run_cfg = getattr(settings, "run", None)
    model_cfg = getattr(run_cfg, "models", None) if run_cfg is not None else None
    return bool(getattr(model_cfg, model_attr, None))


def _filter_schema_steps_for_enabled_models(
    steps: List[Callable[..., Any]],
    settings: Any,
    *,
    include_optional: bool = True,
) -> List[Callable[..., Any]]:
    """
    Keep only step definitions for models enabled in the active run settings.

    Parameters
    ----------
    steps : list of callables
        Step functions decorated with ``@define_step`` metadata.
    settings : Any
        Runtime settings object used to resolve enabled model flags.
    include_optional : bool, default True
        Whether optional steps (currently ``beam_full_skim``) should be included.
    """
    enabled_models = enabled_schema_step_models(
        settings,
        is_model_enabled=_is_model_enabled,
        include_optional=include_optional,
    )

    filtered: List[Callable[..., Any]] = []
    for step_func in steps:
        meta = getattr(step_func, "__consist_step__", None)
        model_name = getattr(meta, "model", "") if meta is not None else ""
        if model_name not in enabled_models:
            continue
        filtered.append(step_func)
    return filtered


def _get_consist_schemas() -> Optional[list[type]]:
    try:
        from pilates.database.schema.registry import get_consist_schemas

        return get_consist_schemas()
    except Exception:
        return None


def build_manifest_path(workspace: Workspace, year: int, iteration: int) -> Path:
    return (
        Path(workspace.full_path)
        / ".workflow"
        / f"year_{year}_iteration_{iteration}.yaml"
    )


def build_atlas_static_inputs_fallback(workspace: Workspace) -> Dict[str, str]:
    """
    Enumerate static ATLAS inputs from the mutable input directory.

    This fallback is used when Initialization was skipped (e.g., restart) and the
    in-memory RecordStore of copied inputs is unavailable. It may include files
    produced by prior ATLAS preprocess runs.
    """
    atlas_input_dir = workspace.get_atlas_mutable_input_dir()
    if not os.path.exists(atlas_input_dir):
        return {}

    settings = getattr(workspace, "settings", None)
    if settings is not None:
        inputs: Dict[str, str] = {}
        for relpath in atlas_static_input_relpaths(settings):
            normalized_relpath = relpath.replace("\\", "/")
            path = os.path.join(atlas_input_dir, normalized_relpath)
            if not os.path.exists(path):
                continue
            rel_no_ext = os.path.splitext(normalized_relpath)[0]
            key = sanitize_artifact_key(rel_no_ext) or rel_no_ext
            inputs.setdefault(key, path)
        if inputs:
            return inputs

    inputs: Dict[str, str] = {}
    for root, _, files in os.walk(atlas_input_dir):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            relpath = os.path.relpath(path, atlas_input_dir)
            rel_no_ext = os.path.splitext(relpath.replace("\\", "/"))[0]
            key = sanitize_artifact_key(rel_no_ext) or rel_no_ext
            inputs.setdefault(key, path)
    return inputs


def _restore_restart_workspace_atlas_registry(
    *,
    settings: Any,
    workspace: Workspace,
) -> int:
    """
    Rebuild the in-memory ATLAS static-input registry from local mutable inputs.

    On restart, ``workspace.input_data`` starts empty even if the local ATLAS
    input tree was rehydrated successfully. Reconstruct the registry so stage
    code can keep using the same authoritative input-source contract.
    """
    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    if getattr(model_cfg, "vehicle_ownership", None) != "atlas":
        return 0

    atlas_input_dir = workspace.get_atlas_mutable_input_dir()
    if not os.path.exists(atlas_input_dir):
        return 0

    records = []
    for relpath in atlas_static_input_relpaths(settings):
        normalized_relpath = relpath.replace("\\", "/")
        local_path = os.path.join(atlas_input_dir, normalized_relpath)
        if not os.path.exists(local_path):
            continue
        rel_no_ext = os.path.splitext(normalized_relpath)[0]
        short_name = sanitize_artifact_key(rel_no_ext) or rel_no_ext
        metadata = {}
        if local_path.lower().endswith(".csv"):
            metadata["profile_file_schema"] = True
        records.append(
            FileRecord(
                file_path=local_path,
                short_name=short_name,
                description=f"Restart-local ATLAS static input: {os.path.basename(local_path)}",
                metadata=metadata,
            )
        )

    if not records:
        return 0

    workspace.input_data["atlas"] = RecordStore(recordList=records)
    return len(records)


def _read_mount_table() -> Dict[str, str]:
    mounts: Dict[str, str] = {}
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    mountpoint = parts[1]
                    fstype = parts[2]
                    mounts[mountpoint] = fstype
    except OSError:
        return {}
    return mounts


def _mount_for_path(path: str, mounts: Dict[str, str]) -> str:
    path = os.path.realpath(path)
    best_match = ""
    for mountpoint in mounts:
        if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) > len(best_match):
                best_match = mountpoint
    return best_match


def _format_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}EiB"


def _log_local_storage_info() -> None:
    mounts = _read_mount_table()
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_NODELIST")
    logger.info(
        "Storage probe: host=%s job_id=%s nodelist=%s",
        hostname,
        job_id or "n/a",
        node_list or "n/a",
    )

    candidates = []
    for var in ("SLURM_TMPDIR", "TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(var)
        if value:
            candidates.append(value)
    candidates += [
        "/tmp",
        "/var/tmp",
        "/dev/shm",
        "/scratch",
        "/local",
        "/local_scratch",
        "/lscratch",
        "/mnt",
    ]

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        mountpoint = _mount_for_path(path, mounts)
        fstype = mounts.get(mountpoint, "unknown")
        logger.info(
            "Storage candidate: path=%s mount=%s fstype=%s free=%s total=%s",
            os.path.realpath(path),
            mountpoint or "unknown",
            fstype,
            _format_bytes(usage.free),
            _format_bytes(usage.total),
        )


def _is_bootstrap_cache_enabled(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "bootstrap_cache_enabled", True))


def _build_bootstrap_manifest_reference(
    *,
    probe_run_id: Optional[str] = None,
    materialization_run_id: Optional[str] = None,
) -> Dict[str, str]:
    reference: Dict[str, str] = {}
    if probe_run_id:
        reference["probe_run_id"] = probe_run_id
    if materialization_run_id:
        reference["materialization_run_id"] = materialization_run_id
    return reference


def run_bootstrap_phase(
    *,
    tracker: Any,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
    scenario_id: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    Execute initialization in a dedicated pre-scenario bootstrap phase.

    Phase 1 behavior:
    - probe cache via tracker.run(...) before scenario starts;
    - if cache hit, force one overwrite run to materialize workspace safely;
    - return cache status plus lightweight artifact summary metadata.
    """
    staged_artifact_summary: Dict[str, Any] = {}

    def _execute_initialization() -> None:
        nonlocal staged_artifact_summary
        init_model = Initialization("initialization", state)
        copied_records = init_model.run(settings, workspace)
        staged_artifact_summary = build_bootstrap_artifact_summary(
            workspace,
            copied_records,
        )

    run_kwargs: Dict[str, Any] = {
        "fn": _execute_initialization,
        "name": "bootstrap_initialization",
        "model": "initialization",
        "year": state.start_year,
        "iteration": 0,
        "phase": "bootstrap",
        "stage": "bootstrap",
        **build_step_consist_kwargs(
            "initialization",
            settings,
            workspace_path=workspace.full_path,
        ),
    }
    run_kwargs["tags"] = _merge_tag_list(
        run_kwargs.get("tags"),
        [
            "bootstrap",
            "init",
            f"scenario_id:{scenario_id}",
            "model:initialization",
            f"year:{state.start_year}",
            "iteration:0",
        ]
        + ([f"seed:{seed}"] if seed is not None else []),
    )
    run_kwargs["facet"] = _merge_epoch_facet(
        existing=run_kwargs.get("facet"),
        scenario_id=scenario_id,
        seed=seed,
        model="initialization",
        year=state.start_year,
        iteration=0,
    )

    if not _is_bootstrap_cache_enabled(settings):
        logger.info("Bootstrap cache disabled; running initialization once.")
        run_result = tracker.run(
            **run_kwargs,
            cache_options=CacheOptions(cache_mode="off"),
        )
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary(workspace)
        return {
            "bootstrap_cache_hit": False,
            "staged_artifact_summary": staged_artifact_summary,
            "manifest_reference": _build_bootstrap_manifest_reference(
                probe_run_id=getattr(getattr(run_result, "run", None), "id", None)
            ),
        }

    probe_result = tracker.run(**run_kwargs)
    probe_run_id = getattr(getattr(probe_result, "run", None), "id", None)
    cache_hit = bool(getattr(probe_result, "cache_hit", False))

    if cache_hit:
        logger.info(
            "BOOTSTRAP CACHE HIT. Running Phase 1 materialization pass to keep workspace safe."
        )
        materialized_result = tracker.run(
            **run_kwargs,
            cache_options=CacheOptions(cache_mode="overwrite"),
        )
        if not staged_artifact_summary:
            staged_artifact_summary = build_bootstrap_artifact_summary(workspace)
        return {
            "bootstrap_cache_hit": True,
            "staged_artifact_summary": staged_artifact_summary,
            "manifest_reference": _build_bootstrap_manifest_reference(
                probe_run_id=probe_run_id,
                materialization_run_id=getattr(
                    getattr(materialized_result, "run", None), "id", None
                ),
            ),
        }

    logger.info("BOOTSTRAP CACHE MISS. Initialization executed for this workspace.")
    if not staged_artifact_summary:
        staged_artifact_summary = build_bootstrap_artifact_summary(workspace)
    return {
        "bootstrap_cache_hit": False,
        "staged_artifact_summary": staged_artifact_summary,
        "manifest_reference": _build_bootstrap_manifest_reference(
            probe_run_id=probe_run_id
        ),
    }


def _assert_bootstrap_output_invariant(
    bootstrap_result: Optional[Dict[str, Any]],
) -> None:
    """
    Ensure bootstrap produced a non-empty artifact summary before state mutation.
    """
    summary = (
        bootstrap_result.get("staged_artifact_summary")
        if isinstance(bootstrap_result, dict)
        else None
    )
    copied_total = (
        summary.get("copied_records_total") if isinstance(summary, dict) else None
    )
    if isinstance(copied_total, int) and copied_total > 0:
        return

    diagnostics = {
        "bootstrap_result_type": type(bootstrap_result).__name__,
        "bootstrap_cache_hit": (
            bootstrap_result.get("bootstrap_cache_hit")
            if isinstance(bootstrap_result, dict)
            else None
        ),
        "manifest_reference": (
            bootstrap_result.get("manifest_reference")
            if isinstance(bootstrap_result, dict)
            else None
        ),
        "staged_artifact_summary": summary,
    }
    raise RuntimeError(
        "Bootstrap initialization invariant failed: expected "
        "'staged_artifact_summary.copied_records_total' > 0 before setting "
        f"data_initialized=True. diagnostics={diagnostics}"
    )


def _restart_required_local_artifacts(
    *,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
) -> List[Dict[str, str]]:
    """
    Build a pragmatic set of local artifacts that must exist to safely skip bootstrap.

    These checks intentionally focus on common restart failures on ephemeral local
    storage (UrbanSim base datastore and ActivitySim mutable inputs).
    """
    required: List[Dict[str, str]] = []

    usim_base_fname = get_usim_datastore_fname(settings, io="input")
    required.append(
        {
            "key": "usim_datastore_base_h5",
            "path": os.path.join(workspace.get_usim_mutable_data_dir(), usim_base_fname),
            "reason": "UrbanSim base datastore required for downstream restart inputs",
        }
    )

    model_cfg = getattr(getattr(settings, "run", None), "models", None)
    current_stage = getattr(state, "current_major_stage", None)
    requires_activitysim_locals = (
        getattr(model_cfg, "activity_demand", None) == "activitysim"
        and (
            current_stage is None
            or current_stage
            in {
                WorkflowState.Stage.supply_demand_loop,
                WorkflowState.Stage.activity_demand,
                WorkflowState.Stage.activity_demand_directly_from_land_use,
            }
        )
    )
    if requires_activitysim_locals:
        asim_data_dir = workspace.get_asim_mutable_data_dir()
        for filename in ("households.csv", "persons.csv", "land_use.csv"):
            required.append(
                {
                    "key": f"activitysim_input_{filename}",
                    "path": os.path.join(asim_data_dir, filename),
                    "reason": "ActivitySim mutable input required on restart",
                }
            )
        asim_configs_dir = workspace.get_asim_mutable_configs_dir()
        main_configs_dir = (
            getattr(getattr(settings, "activitysim", None), "main_configs_dir", None)
            or "configs"
        )
        for dirname in required_asim_config_dirs(main_configs_dir):
            required.append(
                {
                    "key": f"activitysim_config_settings_yaml_{dirname}",
                    "path": os.path.join(asim_configs_dir, dirname, "settings.yaml"),
                    "reason": (
                        "ActivitySim mutable config tree required on restart "
                        f"(config_dir={dirname})"
                    ),
                }
            )

    requires_activitysim_zarr = (
        getattr(model_cfg, "activity_demand", None) == "activitysim"
        and bool(getattr(state, "asim_compiled", False))
        and current_stage
        in {
            WorkflowState.Stage.supply_demand_loop,
            WorkflowState.Stage.activity_demand,
            WorkflowState.Stage.activity_demand_directly_from_land_use,
            WorkflowState.Stage.traffic_assignment,
        }
    )
    get_asim_output_dir = getattr(workspace, "get_asim_output_dir", None)
    if requires_activitysim_zarr and callable(get_asim_output_dir):
        required.append(
            {
                "key": "zarr_skims",
                "path": os.path.join(
                    get_asim_output_dir(),
                    "cache",
                    "skims.zarr",
                ),
                "reason": "ActivitySim compiled skims required for resumed supply-demand loop",
            }
        )

    requires_atlas_locals = (
        getattr(model_cfg, "vehicle_ownership", None) == "atlas"
        and current_stage == WorkflowState.Stage.vehicle_ownership_model
    )
    get_atlas_input_dir = getattr(workspace, "get_atlas_mutable_input_dir", None)
    if requires_atlas_locals and callable(get_atlas_input_dir):
        atlas_input_dir = get_atlas_input_dir()
        for relpath in atlas_static_input_relpaths(settings):
            required.append(
                {
                    "key": f"atlas_static::{relpath}",
                    "path": os.path.join(atlas_input_dir, relpath),
                    "reason": "ATLAS static input required during vehicle ownership restart",
                }
            )

    return required


def _find_missing_restart_local_artifacts(
    *,
    settings: Any,
    state: WorkflowState,
    workspace: Workspace,
) -> List[Dict[str, str]]:
    missing: List[Dict[str, str]] = []
    for artifact in _restart_required_local_artifacts(
        settings=settings, state=state, workspace=workspace
    ):
        path = os.path.realpath(artifact["path"])
        if not os.path.exists(path):
            missing.append(
                {
                    "key": artifact["key"],
                    "path": path,
                    "reason": artifact["reason"],
                }
            )
    return missing


def _format_missing_artifact_summary(artifacts: List[Dict[str, str]]) -> str:
    if not artifacts:
        return "none"
    return ", ".join(
        f"{item.get('key')}:{item.get('path')}" for item in artifacts
    )


def _resolve_restart_rehydrate_mode(settings: Any) -> str:
    run_cfg = getattr(settings, "run", None)
    raw = getattr(run_cfg, "restart_rehydrate_mode", "bundle")
    mode = str(raw).strip().lower() if raw is not None else "bundle"
    if mode in {"bundle", "full", "off"}:
        return mode
    logger.warning(
        "Unknown run.restart_rehydrate_mode=%r; defaulting to 'bundle'.",
        raw,
    )
    return "bundle"


def _is_restart_strict(settings: Any) -> bool:
    run_cfg = getattr(settings, "run", None)
    return bool(getattr(run_cfg, "restart_strict", False))


def _read_archive_run_state_year(state_path: str) -> Optional[int]:
    if not state_path:
        return None
    try:
        year, *_ = WorkflowState.read_current_stage(state_path)
    except Exception as exc:
        logger.warning("Failed reading archive run_state year from %s: %s", state_path, exc)
        return None
    return _coerce_int(year)


def _enforce_resume_rewind_guardrail(
    *,
    state: WorkflowState,
    archive_state_path: str,
    allow_rewind_resume: bool,
) -> None:
    resume_year = _coerce_int(getattr(state, "current_year", None))
    archive_year = _read_archive_run_state_year(archive_state_path)
    if resume_year is None or archive_year is None:
        return
    if resume_year >= archive_year:
        return

    message = (
        "Refusing rewind resume: requested resume year "
        f"{resume_year} is lower than archive run_state year {archive_year} "
        f"(archive={os.path.realpath(archive_state_path)})."
    )
    if allow_rewind_resume:
        logger.warning("%s Proceeding because --allow-rewind-resume was set.", message)
        return
    raise RuntimeError(message + " Use --allow-rewind-resume to override.")


def _map_local_path_to_archive(
    *,
    local_path: str,
    local_run_dir: str,
    archive_run_dir: str,
) -> Optional[str]:
    local_abs = os.path.realpath(local_path)
    local_root = os.path.realpath(local_run_dir)
    archive_root = os.path.realpath(archive_run_dir)
    try:
        if os.path.commonpath([local_abs, local_root]) != local_root:
            return None
    except ValueError:
        return None
    rel = os.path.relpath(local_abs, local_root)
    return os.path.join(archive_root, rel)


def _copy_archive_entry_preserve_existing(
    *,
    archive_path: str,
    local_path: str,
) -> Tuple[int, int]:
    """
    Copy an archive entry to local without overwriting existing files.

    Returns
    -------
    tuple(int, int)
        Number of copied files and number of skipped existing files.
    """
    copied = 0
    skipped_existing = 0

    if os.path.isdir(archive_path):
        for root, _, files in os.walk(archive_path):
            rel_root = os.path.relpath(root, archive_path)
            dest_root = local_path if rel_root == "." else os.path.join(local_path, rel_root)
            os.makedirs(dest_root, exist_ok=True)
            for filename in files:
                src = os.path.join(root, filename)
                dest = os.path.join(dest_root, filename)
                if os.path.exists(dest):
                    skipped_existing += 1
                    continue
                shutil.copy2(src, dest)
                copied += 1
        return copied, skipped_existing

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return 0, 1
    shutil.copy2(archive_path, local_path)
    return 1, 0


def _rehydrate_missing_local_artifacts_from_archive(
    *,
    missing_artifacts: List[Dict[str, str]],
    local_run_dir: str,
    archive_run_dir: str,
) -> Dict[str, int]:
    """
    Rehydrate missing restart artifacts from archive into node-local workspace.

    This is idempotent: existing local files are preserved and never overwritten.
    """
    summary = {
        "copied": 0,
        "skipped_existing": 0,
        "skipped_missing_archive": 0,
        "skipped_unmapped": 0,
        "copy_errors": 0,
    }
    for artifact in missing_artifacts:
        local_path = os.path.realpath(artifact["path"])
        key = artifact.get("key", "unknown")
        kind = artifact.get("kind", "file")

        if os.path.exists(local_path) and not (kind == "dir" and os.path.isdir(local_path)):
            summary["skipped_existing"] += 1
            logger.info(
                "[RestartRehydrate] Skip existing local artifact key=%s path=%s",
                key,
                local_path,
            )
            continue

        archive_path = _map_local_path_to_archive(
            local_path=local_path,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )
        if archive_path is None:
            summary["skipped_unmapped"] += 1
            logger.warning(
                "[RestartRehydrate] Cannot map local path to archive key=%s path=%s",
                key,
                local_path,
            )
            continue
        archive_path = os.path.realpath(archive_path)
        if not os.path.exists(archive_path):
            summary["skipped_missing_archive"] += 1
            logger.warning(
                "[RestartRehydrate] Archive source missing key=%s archive_path=%s",
                key,
                archive_path,
            )
            continue

        try:
            copied, skipped_existing = _copy_archive_entry_preserve_existing(
                archive_path=archive_path,
                local_path=local_path,
            )
            summary["copied"] += copied
            summary["skipped_existing"] += skipped_existing
            logger.info(
                "[RestartRehydrate] key=%s copied=%s skipped_existing=%s source=%s dest=%s",
                key,
                copied,
                skipped_existing,
                archive_path,
                local_path,
            )
        except Exception as exc:
            summary["copy_errors"] += 1
            logger.warning(
                "[RestartRehydrate] Failed copy key=%s source=%s dest=%s error=%s",
                key,
                archive_path,
                local_path,
                exc,
            )

    logger.info(
        "[RestartRehydrate] Summary copied=%s skipped_existing=%s "
        "skipped_missing_archive=%s skipped_unmapped=%s copy_errors=%s",
        summary["copied"],
        summary["skipped_existing"],
        summary["skipped_missing_archive"],
        summary["skipped_unmapped"],
        summary["copy_errors"],
    )
    return summary


def _rehydrate_full_local_run_from_archive(
    *,
    local_run_dir: str,
    archive_run_dir: str,
) -> Dict[str, int]:
    summary = {
        "copied": 0,
        "skipped_existing": 0,
        "skipped_missing_archive": 0,
        "skipped_unmapped": 0,
        "copy_errors": 0,
    }
    archive_root = os.path.realpath(archive_run_dir)
    if not os.path.exists(archive_root):
        summary["skipped_missing_archive"] = 1
        logger.warning(
            "[RestartRehydrate] Full mode archive root missing: %s",
            archive_root,
        )
        return summary

    try:
        copied, skipped_existing = _copy_archive_entry_preserve_existing(
            archive_path=archive_root,
            local_path=os.path.realpath(local_run_dir),
        )
        summary["copied"] = copied
        summary["skipped_existing"] = skipped_existing
    except Exception as exc:
        summary["copy_errors"] = 1
        logger.warning(
            "[RestartRehydrate] Full mode copy failed source=%s dest=%s error=%s",
            archive_root,
            os.path.realpath(local_run_dir),
            exc,
        )
    return summary


def _rehydrate_bundle_local_artifacts_from_archive(
    *,
    bundle_manifest: Optional[Dict[str, Any]],
    local_run_dir: str,
    archive_run_dir: str,
) -> Dict[str, int]:
    bundle_artifacts = manifest_entries_to_local_artifacts(
        manifest=bundle_manifest,
        local_run_dir=local_run_dir,
    )
    if not bundle_artifacts:
        logger.warning(
            "[RestartRehydrate] Bundle mode found no manifest artifacts to hydrate."
        )
        return {
            "copied": 0,
            "skipped_existing": 0,
            "skipped_missing_archive": 0,
            "skipped_unmapped": 0,
            "copy_errors": 0,
        }
    return _rehydrate_missing_local_artifacts_from_archive(
        missing_artifacts=bundle_artifacts,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )


def _log_resume_doctor_check(
    *,
    check: str,
    ok: bool,
    detail: str,
) -> None:
    status = "ok" if ok else "missing"
    log_fn = logger.info if ok else logger.warning
    log_fn(
        "[ResumeDoctor] check=%s status=%s %s",
        check,
        status,
        detail,
    )


def _run_resume_doctor_diagnostics(
    *,
    state: WorkflowState,
    workspace: Workspace,
    local_run_dir: str,
    archive_run_dir: str,
    archive_state_path: str,
    local_state_path: str,
    local_consist_db_path: Optional[str],
    restart_missing_artifacts_initial: List[Dict[str, str]],
    restart_missing_artifacts_after_rehydrate: List[Dict[str, str]],
) -> None:
    """
    Emit startup restart diagnostics for restart readiness.

    This function is logging-only and intentionally does not mutate behavior.
    """
    degraded_checks: List[str] = []

    def record(check: str, ok: bool, detail: str, *, required: bool = True) -> None:
        _log_resume_doctor_check(check=check, ok=ok, detail=detail)
        if required and not ok:
            degraded_checks.append(check)

    logger.info(
        "[ResumeDoctor] start year=%s iteration=%s local_run_dir=%s archive_run_dir=%s",
        state.current_year,
        state.current_inner_iter,
        local_run_dir,
        archive_run_dir,
    )

    archive_state_real = os.path.realpath(archive_state_path)
    local_state_real = os.path.realpath(local_state_path)
    record(
        "archive_run_state",
        os.path.exists(archive_state_real),
        f"path={archive_state_real}",
    )
    record(
        "local_run_state_mirror",
        os.path.exists(local_state_real),
        f"path={local_state_real}",
    )

    if local_consist_db_path:
        local_consist_db_real = os.path.realpath(local_consist_db_path)
        record(
            "local_consist_db",
            os.path.exists(local_consist_db_real),
            f"path={local_consist_db_real}",
        )
        latest_snapshot_db = (
            snapshot_latest_dir(archive_run_dir) / Path(local_consist_db_real).name
        )
        latest_snapshot_real = os.path.realpath(str(latest_snapshot_db))
        record(
            "archive_latest_consist_db_snapshot",
            os.path.exists(latest_snapshot_real),
            f"path={latest_snapshot_real}",
        )
    else:
        record(
            "local_consist_db",
            True,
            "path=none reason=disabled_or_unconfigured",
            required=False,
        )
        record(
            "archive_latest_consist_db_snapshot",
            True,
            "path=none reason=disabled_or_unconfigured",
            required=False,
        )

    if state.data_initialized:
        missing_summary = _format_missing_artifact_summary(
            restart_missing_artifacts_after_rehydrate
        )
        record(
            "required_restart_local_artifacts",
            not restart_missing_artifacts_after_rehydrate,
            "data_initialized=true "
            f"initial_missing={len(restart_missing_artifacts_initial)} "
            f"remaining_missing={len(restart_missing_artifacts_after_rehydrate)} "
            f"missing={missing_summary}",
        )
    else:
        record(
            "required_restart_local_artifacts",
            True,
            "data_initialized=false reason=bootstrap_required",
            required=False,
        )

    year = state.current_year
    iteration = state.current_inner_iter
    local_manifest_path: Optional[Path] = None
    try:
        local_manifest_path = build_manifest_path(
            workspace=workspace,
            year=int(year),
            iteration=int(iteration),
        )
    except Exception as exc:
        record(
            "supply_demand_manifest_local",
            False,
            f"year={year} iteration={iteration} error={exc}",
        )
        record(
            "supply_demand_manifest_archive_mapped",
            False,
            f"year={year} iteration={iteration} error=local_manifest_path_unavailable",
        )

    if local_manifest_path is not None:
        local_manifest_real = os.path.realpath(str(local_manifest_path))
        record(
            "supply_demand_manifest_local",
            os.path.exists(local_manifest_real),
            f"year={year} iteration={iteration} path={local_manifest_real}",
        )
        archive_manifest_path = _map_local_path_to_archive(
            local_path=local_manifest_real,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
        )
        if archive_manifest_path is None:
            record(
                "supply_demand_manifest_archive_mapped",
                False,
                f"year={year} iteration={iteration} local_path={local_manifest_real} archive_path=unmapped",
            )
        else:
            archive_manifest_real = os.path.realpath(archive_manifest_path)
            record(
                "supply_demand_manifest_archive_mapped",
                os.path.exists(archive_manifest_real),
                f"year={year} iteration={iteration} local_path={local_manifest_real} archive_path={archive_manifest_real}",
            )

    if degraded_checks:
        logger.warning(
            "[ResumeDoctor] summary status=degraded reason=missing_checks:%s",
            ",".join(degraded_checks),
        )
    else:
        logger.info("[ResumeDoctor] summary status=ready reason=all_checks_ok")


def main():
    """
    Main entrypoint for PILATES simulation orchestration using Consist Scenario API.

    This workflow coordinates multiple land use and transportation microsimulation models
    across a multi-year planning horizon:

    1. **Initialization**: Copy immutable input data to mutable workspace
    2. **Land Use Forecasting**: UrbanSim predicts demographic/economic changes
    3. **Vehicle Ownership**: ATLAS models vehicle fleet evolution
    4. **Supply/Demand Loop**: Iterates between activity demand (ActivitySim) and
       traffic assignment (BEAM) until convergence
    5. **Post-Processing**: Validation and output generation

    Architecture:
    - **Consist Scenario**: Manages caching of expensive computations and provenance logging
    - **Coupler**: Passes artifacts (outputs) between models via `scenario.coupler`
    - **StepConfig**: Declarative config for each model step
    - **Step Builders**: Encapsulate model-specific execution logic

    Caching Strategy:
    - ActivitySim compilation: Cached across iterations (inputs unchanged = skip compile)
    - Model outputs: Cached per iteration (convergence check)
    - Restarting: Skips initialization if run_state.yaml exists
    """
    # 1. PARSE SETTINGS AND SET UP WORKFLOW STATE
    settings = parse_args_and_settings()
    state = WorkflowState.from_settings(settings)

    _log_local_storage_info()

    # 2. SETUP PATHS
    output_directory = settings.run.output_directory
    if not output_directory:
        raise ValueError("output_directory not found in config")
    output_path = os.path.realpath(os.path.expandvars(output_directory))
    local_workspace_root = getattr(settings.run, "local_workspace_root", None)
    if local_workspace_root:
        local_root = os.path.realpath(os.path.expandvars(local_workspace_root))
    else:
        local_root = output_path

    # Split run roots:
    # - archive_run_dir (scratch) holds Consist run metadata + archived artifacts
    # - local_run_dir (node-local) holds mutable workspace during execution

    is_restart_run = bool(state.run_info_path)
    if is_restart_run:
        run_name = os.path.basename(os.path.dirname(state.run_info_path))
        logger.info(f"Restarting run. Reusing output folder: {run_name}")
    else:
        partial_run_name = settings.run.output_run_name
        run_name = f"{partial_run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info(f"Starting fresh run. Creating new output folder: {run_name}")
    scenario_id = _resolve_scenario_id(settings)
    run_seed = _resolve_seed(settings)
    logger.info(
        "Resolved run tagging metadata: scenario_id=%s seed=%s",
        scenario_id,
        run_seed if run_seed is not None else "n/a",
    )

    archive_run_dir = os.path.join(output_path, run_name)
    local_run_dir = os.path.join(local_root, run_name)
    os.makedirs(local_run_dir, exist_ok=True)
    if archive_run_dir != local_run_dir:
        os.makedirs(archive_run_dir, exist_ok=True)

    os.environ["PILATES_LOCAL_RUN_DIR"] = local_run_dir
    os.environ["PILATES_ARCHIVE_RUN_DIR"] = archive_run_dir
    os.environ["PILATES_ENABLE_ARCHIVE_COPY"] = (
        "1" if settings.run.enable_archive_copy else "0"
    )

    # 3. INITIALIZE CONSIST TRACKER
    # Consist provides provenance tracking and computation caching.
    # It is required for PILATES execution.
    #   - Provenance: Full lineage of data transformations (OpenLineage compatible)
    #   - Caching: Skips expensive computations if inputs unchanged
    #   - Coupler: Manages artifact passing between steps
    # Mount Strategy:
    # - 'inputs': The project root. Source files resolve here.
    # - 'workspace': The mutable run dir. Destination files resolve here.
    # NOTE: Do not rely on cwd; production runs may invoke `python run.py` from elsewhere.
    # Use the directory containing `run.py` as the canonical inputs root.
    project_root_abs = str(Path(__file__).resolve().parent)

    local_consist_db_path, archive_consist_db_path = resolve_consist_db_paths(
        settings=settings,
        local_run_dir=local_run_dir,
        archive_run_dir=archive_run_dir,
    )
    if local_consist_db_path:
        os.makedirs(os.path.dirname(local_consist_db_path), exist_ok=True)
    restored_from_snapshot = restore_local_consist_db_from_snapshot(
        settings=settings,
        local_db_path=local_consist_db_path,
        archive_run_dir=archive_run_dir,
    )
    if not restored_from_snapshot:
        shared_db_path = getattr(getattr(settings, "shared", None), "database", None)
        seed_local_consist_db_from_shared(
            settings=settings,
            local_db_path=local_consist_db_path,
            shared_db_path=getattr(shared_db_path, "path", None),
        )
    logger.info(
        "Initializing Consist Tracker in %s (db_path=%s)",
        archive_run_dir,
        local_consist_db_path,
    )

    cache_epoch = _resolve_cache_epoch(settings)

    tracker = cr.create_tracker(
        settings=settings,
        run_dir=archive_run_dir,
        db_path=local_consist_db_path,
        cache_epoch=cache_epoch,
        mounts={
            "inputs": project_root_abs,  # Immutable Source
            "workspace": local_run_dir,  # Mutable Destination
            "scratch": str(Path(output_path).resolve()),  # For temp files
        },
        project_root=project_root_abs,
        schemas=_get_consist_schemas(),
    )
    if tracker is None:
        raise RuntimeError(
            "Consist tracker initialization failed (received noop/invalid tracker). "
            "Check earlier Consist logs for tracker creation errors, often caused by "
            "a PILATES/Consist API mismatch."
        )
    snapshot_manager = ConsistDbSnapshotManager(
        settings=settings,
        tracker=tracker,
        local_db_path=local_consist_db_path,
        archive_run_dir=archive_run_dir,
    )

    # 4. INITIALIZE WORKSPACE
    workspace = Workspace(
        settings,
        local_root,
        folder_name=run_name,
    )
    archive_state_path = os.path.join(archive_run_dir, "run_state.yaml")
    local_state_path = os.path.join(workspace.full_path, "run_state.yaml")
    if is_restart_run:
        _enforce_resume_rewind_guardrail(
            state=state,
            archive_state_path=archive_state_path,
            allow_rewind_resume=bool(getattr(settings, "allow_rewind_resume", False)),
        )
    state.file_loc = archive_state_path
    state.mirror_file_loc = local_state_path
    if state.run_info_path != archive_state_path:
        state.set_run_info_path(archive_state_path)

    restart_rehydrate_mode = _resolve_restart_rehydrate_mode(settings)
    restart_strict = _is_restart_strict(settings)
    restart_bundle_manifest = build_restart_bundle_manifest(
        archive_run_dir=archive_run_dir,
        local_run_dir=local_run_dir,
        settings=settings,
        workspace=workspace,
        state=state,
        local_consist_db_path=local_consist_db_path,
    )
    restart_bundle_manifest_path = write_restart_bundle_manifest(
        archive_run_dir=archive_run_dir,
        manifest=restart_bundle_manifest,
    )
    logger.info(
        "Restart bundle manifest ready: path=%s artifacts=%s mode=%s strict=%s",
        restart_bundle_manifest_path,
        len(restart_bundle_manifest.get("artifacts", [])),
        restart_rehydrate_mode,
        restart_strict,
    )

    restart_missing_artifacts_initial: List[Dict[str, str]] = []
    restart_missing_artifacts_after_rehydrate: List[Dict[str, str]] = []
    if state.data_initialized:
        restart_missing_artifacts_initial = _find_missing_restart_local_artifacts(
            settings=settings,
            state=state,
            workspace=workspace,
        )
        if restart_missing_artifacts_initial:
            logger.warning(
                "Restart preflight found missing local workspace inputs while "
                "data_initialized=True: %s",
                _format_missing_artifact_summary(restart_missing_artifacts_initial),
            )
            if restart_rehydrate_mode == "off":
                logger.info("Restart rehydration disabled (run.restart_rehydrate_mode=off).")
            elif restart_rehydrate_mode == "full":
                _rehydrate_full_local_run_from_archive(
                    local_run_dir=local_run_dir,
                    archive_run_dir=archive_run_dir,
                )
            else:
                _rehydrate_bundle_local_artifacts_from_archive(
                    bundle_manifest=restart_bundle_manifest,
                    local_run_dir=local_run_dir,
                    archive_run_dir=archive_run_dir,
                )
            restart_missing_artifacts_after_rehydrate = (
                _find_missing_restart_local_artifacts(
                    settings=settings,
                    state=state,
                    workspace=workspace,
                )
            )
            if restart_missing_artifacts_after_rehydrate:
                logger.warning(
                    "Restart preflight still missing required local workspace inputs "
                    "after archive rehydration: %s",
                    _format_missing_artifact_summary(
                        restart_missing_artifacts_after_rehydrate
                    ),
                )
                if restart_strict:
                    raise RuntimeError(
                        "Strict restart preflight failed; required restart artifacts are "
                        "still missing after hydration. missing="
                        + _format_missing_artifact_summary(
                            restart_missing_artifacts_after_rehydrate
                        )
                    )
        restored_atlas_records = _restore_restart_workspace_atlas_registry(
            settings=settings,
            workspace=workspace,
        )
        if restored_atlas_records:
            logger.info(
                "Restored ATLAS restart registry from local mutable inputs: records=%s",
                restored_atlas_records,
            )
    if is_restart_run:
        _run_resume_doctor_diagnostics(
            state=state,
            workspace=workspace,
            local_run_dir=local_run_dir,
            archive_run_dir=archive_run_dir,
            archive_state_path=archive_state_path,
            local_state_path=local_state_path,
            local_consist_db_path=local_consist_db_path,
            restart_missing_artifacts_initial=restart_missing_artifacts_initial,
            restart_missing_artifacts_after_rehydrate=restart_missing_artifacts_after_rehydrate,
        )

    # 5. BOOTSTRAP PHASE (PRE-SCENARIO)
    # Initialization runs before entering scenario step execution so bootstrap
    # lifecycle can evolve independently from normal model steps.
    cr.set_tracker(tracker)
    bootstrap_result: Optional[Dict[str, Any]] = None
    force_restart_bootstrap = bool(restart_missing_artifacts_after_rehydrate)
    if (
        restart_missing_artifacts_initial
        and not restart_missing_artifacts_after_rehydrate
    ):
        logger.info(
            "Restart preflight recovered missing local inputs from archive; "
            "continuing without forced bootstrap."
        )
    if not state.data_initialized or force_restart_bootstrap:
        if force_restart_bootstrap:
            logger.warning(
                "Forcing bootstrap initialization on restart because required local "
                "inputs were missing during preflight. initial_missing=%s "
                "remaining_after_rehydrate=%s",
                _format_missing_artifact_summary(restart_missing_artifacts_initial),
                _format_missing_artifact_summary(
                    restart_missing_artifacts_after_rehydrate
                ),
            )
        else:
            logger.info("Running bootstrap initialization phase.")
        bootstrap_result = run_bootstrap_phase(
            tracker=tracker,
            settings=settings,
            state=state,
            workspace=workspace,
            scenario_id=scenario_id,
            seed=run_seed,
        )
        _assert_bootstrap_output_invariant(bootstrap_result)
        if not state.data_initialized:
            state.set_data_initialized(True)
    else:
        logger.info("Restarting from a previous state. Skipping bootstrap phase.")
    if bootstrap_result is not None:
        logger.info(
            "Bootstrap phase complete: cache_hit=%s manifest_ref=%s summary=%s",
            bootstrap_result.get("bootstrap_cache_hit"),
            bootstrap_result.get("manifest_reference"),
            bootstrap_result.get("staged_artifact_summary"),
        )

    # 6. START SCENARIO CONTEXT
    # The scenario context is where all model execution happens. Each step runs inside
    # scenario.run(), which handles:
    #   - Caching checks (skip if inputs identical to previous run)
    #   - Provenance logging (inputs, outputs, dependencies)
    #   - Coupler coordination (step outputs → coupler → next step inputs)
    # The coupler is a shared dict-like object for passing artifacts between steps.
    scenario_kwargs = build_scenario_consist_kwargs(settings)
    scenario_kwargs["facet"] = _merge_epoch_facet(
        existing=scenario_kwargs.get("facet"),
        scenario_id=scenario_id,
        seed=run_seed,
        model="pilates_orchestrator",
        year=None,
        iteration=None,
    )
    scenario_kwargs.setdefault("name_template", _SCENARIO_NAME_TEMPLATE)
    scenario_kwargs.setdefault("cache_epoch", cache_epoch)
    schema_steps_all = _build_schema_steps()
    validate_workflow_step_contracts(declared_steps=schema_steps_all)
    schema_steps_enabled = _filter_schema_steps_for_enabled_models(
        schema_steps_all,
        settings,
        include_optional=True,
    )
    coupler_schema = build_coupler_schema(schema_steps_enabled, settings=settings)
    required_schema = build_coupler_schema(
        _filter_schema_steps_for_enabled_models(
            schema_steps_all,
            settings,
            include_optional=False,
        ),
        settings=settings,
        include_extras=False,
    )
    required_output_keys = list(required_schema.keys())
    scenario_kwargs["require_outputs"] = required_output_keys

    preview_count = 25
    logger.info(
        "Scenario output contract: declared_keys=%d require_outputs=%d "
        "(enabled_steps=%d/%d). Preview: %s",
        len(coupler_schema),
        len(required_output_keys),
        len(schema_steps_enabled),
        len(schema_steps_all),
        required_output_keys[:preview_count],
    )
    scenario_tags = _merge_tag_list(
        ["pilates_simulation"],
        [f"scenario_id:{scenario_id}"] + ([f"seed:{run_seed}"] if run_seed is not None else []),
    )
    try:
        with cr.scenario(
            run_name,
            tracker=tracker,
            tags=scenario_tags,
            model="pilates_orchestrator",
            **scenario_kwargs,
        ) as scenario:
            tagged_scenario = _EpochTaggingScenarioProxy(
                scenario,
                scenario_id=scenario_id,
                seed=run_seed,
            )
            coupler = tagged_scenario.coupler
            coupler.declare_outputs(
                *coupler_schema.keys(),
                warn_undefined=True,
                description=coupler_schema,
            )

            # 7. MAIN WORKFLOW LOOP
            # Iterates through forecast years. For each year, runs sequential stages:
            # A (Land Use) → B (Vehicle Ownership) → C (Supply/Demand Loop) → D (Post-Processing)
            #
            # Step Pattern (used for all stages):
            #   1. build_*_inputs(...)      - Collect inputs from previous outputs + coupler
            #   2. log_inputs(...)          - Log for provenance
            #   3. build_*_outputs(...)     - Declare what we expect to produce
            #   4. make_*_step(...)         - Create step function with coupler refs
            #   5. build_step_config(...)   - Create config (year, iteration, inputs, outputs, kwargs)
            #   6. scenario.run(...)        - Execute via Consist (handles caching + provenance)
            #
            for year in state:
                formatted_print(f"STARTING YEAR {year}")
                usim_inputs: Dict[str, Any] = {}
                outputs_holder_year = StepOutputsHolder()

                if state.should_run(WorkflowState.Stage.land_use):
                    usim_inputs = run_land_use_stage(
                        scenario=tagged_scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        outputs_holder_year=outputs_holder_year,
                    )
                    state.complete_step(WorkflowState.Stage.land_use)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_land_use_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.vehicle_ownership_model):
                    formatted_print(
                        f"VEHICLE OWNERSHIP MODEL (ATLAS) FOR YEAR {state.forecast_year}"
                    )
                    run_vehicle_ownership_stage(
                        scenario=tagged_scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        build_atlas_static_inputs_fallback=build_atlas_static_inputs_fallback,
                    )
                    state.complete_step(WorkflowState.Stage.vehicle_ownership_model)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_vehicle_ownership_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.supply_demand_loop):
                    run_supply_demand_stage(
                        scenario=tagged_scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                        usim_inputs=usim_inputs,
                        build_manifest_path=build_manifest_path,
                        on_iteration_boundary=(
                            lambda iteration, y=year: snapshot_manager.on_outer_iteration_boundary(
                                year=y,
                                iteration=iteration,
                            )
                        ),
                    )
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_supply_demand_y{year}"
                    )

                if state.should_run(WorkflowState.Stage.postprocessing):
                    formatted_print("POST-PROCESSING")
                    run_postprocessing_stage(
                        scenario=tagged_scenario,
                        state=state,
                        settings=settings,
                        workspace=workspace,
                        coupler=coupler,
                        year=year,
                    )
                    state.complete_step(WorkflowState.Stage.postprocessing)
                    snapshot_manager.maybe_snapshot_interval(
                        reason=f"after_postprocessing_y{year}"
                    )
                snapshot_manager.maybe_snapshot_interval(reason=f"year_boundary_y{year}")

        formatted_print("SIMULATION COMPLETE")
        logger.info("[Main] Simulation complete.")
    finally:
        snapshot_ok = snapshot_manager.final_snapshot()
        flush_archive_queue(timeout=300)
        stop_archive_worker(timeout=30)
        if not snapshot_ok:
            mirror_consist_db_to_archive(local_consist_db_path, archive_consist_db_path)


if __name__ == "__main__":
    main()
