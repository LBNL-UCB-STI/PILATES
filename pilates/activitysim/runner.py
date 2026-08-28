import logging
import os
import shutil
import inspect
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Literal, Mapping

import xarray as xr
import zarr

from pilates.config import PilatesConfig
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.zone_utils import ensure_0_based_and_flag_zarr_skims
from pilates.activitysim.outputs import (
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    write_asim_run_marker,
    clear_asim_run_marker,
    configured_asim_output_tables,
)
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ZARR_SKIMS,
)

logger = logging.getLogger(__name__)

ActivitysimSkimMode = Literal["omx", "zarr"]


@dataclass(frozen=True, slots=True)
class ActivitySimSkimDecision:
    """Immutable resolver-owned selection of one ActivitySim skim source.

    All fields are scalar JSON-compatible values so the decision can be carried
    through resolver metadata without embedding a host-specific source path.
    """

    role: str
    mode: ActivitysimSkimMode
    produces_zarr: bool
    selected_source: str


def validate_activitysim_zarr_skims(path: str | Path) -> str | None:
    """Return a read-only structural rejection reason for a Zarr store, if any."""

    zarr_path = Path(path)
    if not zarr_path.exists():
        return "path does not exist"
    if not zarr_path.is_dir():
        return "path is not a directory"

    metadata_paths = tuple(
        candidate
        for candidate in (zarr_path / ".zgroup", zarr_path / "zarr.json")
        if candidate.exists()
    )
    if not metadata_paths:
        return "missing Zarr metadata (.zgroup or zarr.json)"
    if len(metadata_paths) > 1:
        return "ambiguous Zarr root metadata (.zgroup and zarr.json)"
    metadata_path = metadata_paths[0]
    if not metadata_path.is_file():
        return f"Zarr metadata is not a file: {metadata_path.name}"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return f"invalid Zarr metadata {metadata_path.name}: {error}"
    if not isinstance(metadata, dict):
        return f"invalid Zarr metadata {metadata_path.name}: expected an object"
    if metadata_path.name == ".zgroup":
        if metadata.get("zarr_format") != 2:
            return ".zgroup must declare zarr_format 2"
    else:
        if metadata.get("zarr_format") != 3:
            return "zarr.json must declare zarr_format 3"
        if metadata.get("node_type") != "group":
            return "zarr.json root node_type must be 'group'"
    return None


@dataclass(frozen=True, slots=True)
class ActivitySimLaunchContext:
    """Resolved model-visible roots for one ActivitySim invocation."""

    workspace_root: Path
    mutable_data_dir: Path
    output_dir: Path
    compile_output_dir: Path
    mutable_configs_dir: Path
    runtime_cache_dir: Path
    runtime_zarr_path: Path
    shared_cache_dir: Path
    shared_tmp_dir: Path
    requires_staged_config_dirs: bool = False


@dataclass(frozen=True, slots=True)
class ActivitySimPreparationResult:
    """Resolved private preparation outcome supplied to one body launch."""

    launch_context: ActivitySimLaunchContext
    skim_mode: ActivitysimSkimMode
    zarr_input_path: Optional[str]
    generated_zarr_path: Optional[Path]


@dataclass(frozen=True, slots=True)
class _ActivitySimCompileEpoch:
    """One process-local private Numba cache for a live workflow state."""

    state: "WorkflowState"
    cache_dir: Path


class ActivitySimCompileEpochRegistry:
    """Keep successful private compile epochs out of workflow identity.

    Entries intentionally retain the live workflow state object and compare by
    object identity. A workspace pathname cannot prove that this Python process
    prepared the cache, while a live state object identifies one invocation.
    """

    def __init__(self) -> None:
        self._epochs: list[_ActivitySimCompileEpoch] = []

    def get(self, state: "WorkflowState") -> Path | None:
        for epoch in self._epochs:
            if epoch.state is state:
                return epoch.cache_dir
        return None

    def record(self, state: "WorkflowState", cache_dir: Path) -> None:
        if self.get(state) is not None:
            raise RuntimeError("ActivitySim compile epoch already recorded for state")
        self._epochs.append(_ActivitySimCompileEpoch(state=state, cache_dir=cache_dir))


_ACTIVITYSIM_COMPILE_EPOCHS = ActivitySimCompileEpochRegistry()


@dataclass(frozen=True, slots=True)
class ActivitySimConfigStagingPlan:
    """Runtime-only source roots copied into an ActivitySim launch tree."""

    source_roots: tuple[Path, ...]


def stage_activitysim_config_roots(
    launch_context: ActivitySimLaunchContext,
    staging_plan: ActivitySimConfigStagingPlan,
) -> None:
    """Copy the adapter-selected configuration roots into the launch tree."""
    if not staging_plan.source_roots:
        raise RuntimeError("ActivitySim launch context is missing configuration roots")
    for source in staging_plan.source_roots:
        if not source.is_dir():
            raise RuntimeError(
                f"ActivitySim staged configuration source is missing: {source}"
            )
        destination = launch_context.mutable_configs_dir / source.name
        _remove_path_if_present(str(destination))
        shutil.copytree(source, destination)


def _log_activitysim_launch_context(
    *,
    image: str,
    environment: Mapping[str, str],
    working_dir: str,
    volumes: Mapping[str, Mapping[str, str]],
    zarr_input_path: Optional[str] = None,
) -> None:
    logger.info(
        "ActivitySim launch context: image=%s working_dir=%s env=%s",
        image,
        working_dir,
        dict(environment),
    )
    logger.info(
        "ActivitySim host Python stack: xarray=%s (%s) zarr=%s (%s)",
        xr.__version__,
        getattr(xr, "__file__", "unknown"),
        getattr(zarr, "__version__", "unknown"),
        getattr(zarr, "__file__", "unknown"),
    )
    logger.info(
        "ActivitySim host to_zarr signature: %s",
        inspect.signature(xr.Dataset.to_zarr),
    )
    if zarr_input_path is not None:
        logger.info("ActivitySim zarr input path: %s", zarr_input_path)
    logger.info(
        "ActivitySim volume mounts: %s",
        {
            local: {"bind": spec.get("bind"), "mode": spec.get("mode")}
            for local, spec in sorted(volumes.items())
        },
    )


def _asim_container_environment() -> Dict[str, str]:
    """
    Environment variables passed into ActivitySim containers.

    ``PYTHONNOUSERSITE=1`` prevents host user-site packages from leaking into
    the container Python environment, which can otherwise mix incompatible
    xarray/zarr versions with the image's pinned stack.
    """
    env = {
        "NUMBA_CACHE_DIR": "/app/numba_cache/numba",
        "XDG_CACHE_HOME": "/app/numba_cache",
        "PYTHONNOUSERSITE": "1",
    }
    for key in (
        "ASIM_DEBUG_ZARR_WRITE",
        "ASIM_DEBUG_ZARR_PROBE",
        "ASIM_DEBUG_ZARR_PROBE_ONLY",
        "ASIM_DEBUG_ZARR_PROBE_DIR",
        "ASIM_DEBUG_ZARR_PROBE_LIMIT",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def persist_sharrow_cache_enabled(settings: PilatesConfig) -> bool:
    """
    Return whether ActivitySim should persist sharrow/numba compile caches.

    Backward compatibility:
    - If an explicit ``activitysim.persist_sharrow_cache`` flag is provided,
      it controls behavior.
    - Otherwise fall back to historical behavior where parquet mode persists
      sharrow cache by default.
    """
    activitysim_cfg = getattr(settings, "activitysim", None)
    if activitysim_cfg is None:
        return False
    explicit_flag = getattr(activitysim_cfg, "persist_sharrow_cache", None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    return getattr(activitysim_cfg, "file_format", None) == "parquet"


def asim_sharrow_cache_dir(workspace: Workspace) -> str:
    """
    Canonical ActivitySim sharrow/numba cache directory for compile outputs.
    """
    return os.path.join(workspace.full_path, "shared_cache", "numba")


def asim_runtime_cache_dir(workspace: Workspace) -> str:
    """
    Canonical ActivitySim runtime cache directory for skims.zarr.
    """
    get_runtime_cache_dir = getattr(workspace, "get_asim_runtime_cache_dir", None)
    if callable(get_runtime_cache_dir):
        return get_runtime_cache_dir()
    get_output_dir = getattr(workspace, "get_asim_output_dir", None)
    if callable(get_output_dir):
        return os.path.join(get_output_dir(), "cache")
    return os.path.join(getattr(workspace, "full_path", os.getcwd()), "cache")


def asim_runtime_zarr_path(workspace: Workspace) -> str:
    return os.path.join(asim_runtime_cache_dir(workspace), "skims.zarr")


def asim_compile_output_dir(workspace: Workspace) -> str:
    """Return the private output root used by the ActivitySim Numba warmup."""
    return str(Path(workspace.get_asim_output_dir()).parent / "compile-output")


def asim_staged_input_paths(workspace: Workspace) -> Dict[str, str]:
    asim_data_dir = workspace.get_asim_mutable_data_dir()
    return {
        ASIM_LAND_USE_IN: os.path.join(asim_data_dir, "land_use.csv"),
        ASIM_HOUSEHOLDS_IN: os.path.join(asim_data_dir, "households.csv"),
        ASIM_PERSONS_IN: os.path.join(asim_data_dir, "persons.csv"),
        ASIM_OMX_SKIMS: os.path.join(asim_data_dir, "skims.omx"),
    }


def asim_required_run_output_paths(
    settings: PilatesConfig, workspace: Workspace
) -> Dict[str, str]:
    final_pipeline_dir = Path(workspace.get_asim_output_dir()) / "final_pipeline"
    return {
        output_key: str(final_pipeline_dir / table_name / "final.parquet")
        for output_key, table_name in configured_asim_output_tables(settings).items()
    }


def _dir_contains_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def _remove_path_if_present(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def _stage_runtime_input_path(
    *,
    key: str,
    input_path: str,
    workspace: Optional[Workspace] = None,
    runtime_zarr_path: Optional[Path] = None,
) -> str:
    if key == "zarr_skims":
        if runtime_zarr_path is None:
            if workspace is None:
                raise RuntimeError("ActivitySim Zarr staging requires a runtime path")
            runtime_path = asim_runtime_zarr_path(workspace)
        else:
            runtime_path = str(runtime_zarr_path)
    else:
        return input_path
    if os.path.abspath(input_path) == os.path.abspath(runtime_path):
        return runtime_path

    os.makedirs(os.path.dirname(runtime_path), exist_ok=True)
    if os.path.isdir(input_path):
        _remove_path_if_present(runtime_path)
        shutil.copytree(input_path, runtime_path)
    else:
        if os.path.isdir(runtime_path):
            _remove_path_if_present(runtime_path)
        shutil.copyfile(input_path, runtime_path)
    return runtime_path


def finalize_activitysim_zarr_skims(
    path: str | Path,
    settings: PilatesConfig,
    workspace: Workspace,
) -> Path:
    """Validate and normalize a generated local Zarr skim store.

    This is deliberately local execution preparation.  Publication belongs to
    the OMX-mode primary ActivitySim step after its production invocation.
    """
    zarr_path = Path(path)
    if not zarr_path.exists():
        raise RuntimeError(
            f"ActivitySim did not create the expected Zarr skims at {zarr_path}."
        )
    try:
        ensure_0_based_and_flag_zarr_skims(str(zarr_path), settings, workspace)
    except Exception as error:
        raise RuntimeError(
            f"Failed to finalize generated ActivitySim Zarr skims at {zarr_path}."
        ) from error
    return zarr_path


class ActivitysimNumbaWarmup(GenericRunner):
    """
    Private compile-mode execution used to warm node-local Numba caches.

    It intentionally has no workflow identity, typed outputs, state mutation,
    or provenance publication. Its temporary output root is isolated from the
    production output root, including any compile-local Zarr conversion.
    """

    @staticmethod
    def get_base_asim_cmd(
        settings: PilatesConfig, household_sample_size=None, num_processes=None
    ):
        return ActivitysimRunner.get_base_asim_cmd(
            settings,
            household_sample_size=household_sample_size,
            num_processes=num_processes,
        )

    @staticmethod
    def get_asim_additional_args(settings: PilatesConfig, asim_docker_vols, compile):
        return ActivitysimRunner.get_asim_additional_args(
            settings, asim_docker_vols, compile
        )

    @staticmethod
    def get_asim_docker_vols(
        settings: PilatesConfig,
        working_dir=None,
        *,
        launch_context: Optional[ActivitySimLaunchContext] = None,
        workspace: Optional[Workspace] = None,
        output_dir: Optional[str] = None,
        runtime_cache_dir: Optional[str] = None,
        runtime_cache_mode: Literal["ro", "rw"] = "rw",
    ):
        return ActivitysimRunner.get_asim_docker_vols(
            settings,
            working_dir=working_dir,
            launch_context=launch_context,
            workspace=workspace,
            output_dir=output_dir,
            runtime_cache_dir=runtime_cache_dir,
            runtime_cache_mode=runtime_cache_mode,
        )

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_files = [
            ASIM_OMX_SKIMS,
            "asim_geoms",
            ASIM_PERSONS_IN,
            ASIM_HOUSEHOLDS_IN,
            ASIM_LAND_USE_IN,
        ]

    def run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        launch_context: ActivitySimLaunchContext,
        *,
        skim_mode: ActivitysimSkimMode = "omx",
        zarr_input_path: Optional[str] = None,
    ) -> None:
        if not isinstance(inputs, ActivitySimPreprocessOutputs):
            raise TypeError(
                "ActivitysimNumbaWarmup.run expects ActivitySimPreprocessOutputs"
            )
        self._run(
            inputs,
            launch_context,
            skim_mode=skim_mode,
            zarr_input_path=zarr_input_path,
        )

    def _run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        launch_context: ActivitySimLaunchContext,
        *,
        skim_mode: ActivitysimSkimMode = "omx",
        zarr_input_path: Optional[str] = None,
    ) -> None:
        del inputs
        settings = self.state.full_settings
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)

        shared_cache_dir = str(launch_context.shared_cache_dir)
        shared_tmp_dir = str(launch_context.shared_tmp_dir)

        os.makedirs(os.path.join(shared_cache_dir, "numba"), exist_ok=True)
        os.makedirs(shared_tmp_dir, exist_ok=True)

        compile_output_dir = str(launch_context.compile_output_dir)
        runtime_cache_dir = str(launch_context.runtime_cache_dir)
        runtime_cache_mode: Literal["ro", "rw"] = (
            "ro" if skim_mode == "zarr" else "rw"
        )
        if skim_mode == "zarr":
            if zarr_input_path is None or not os.path.isdir(zarr_input_path):
                raise RuntimeError(
                    "ActivitySim warmup requires a staged Zarr skim input directory."
                )
            if os.path.abspath(zarr_input_path) != os.path.abspath(
                str(launch_context.runtime_zarr_path)
            ):
                raise RuntimeError(
                    "ActivitySim warmup requires the staged runtime Zarr skim input."
                )
        else:
            os.makedirs(runtime_cache_dir, exist_ok=True)
        asim_docker_vols = self.get_asim_docker_vols(
            settings,
            launch_context=replace(
                launch_context,
                output_dir=Path(compile_output_dir),
            ),
            output_dir=compile_output_dir,
            runtime_cache_dir=runtime_cache_dir,
            runtime_cache_mode=runtime_cache_mode,
        )
        asim_docker_vols.update(
            {
                shared_tmp_dir: {"bind": "/tmp", "mode": "rw"},
                shared_cache_dir: {"bind": "/app/numba_cache", "mode": "rw"},
            }
        )

        _, activity_demand_image = self.get_model_and_image(
            settings, "activity_demand_model"
        )

        asim_cmd = self.get_base_asim_cmd(
            settings, household_sample_size=2500, num_processes=1
        )
        additional_args = self.get_asim_additional_args(
            settings, asim_docker_vols, True
        )
        container_environment = _asim_container_environment()
        _log_activitysim_launch_context(
            image=activity_demand_image,
            environment=container_environment,
            working_dir=asim_workdir,
            volumes=asim_docker_vols,
            zarr_input_path=zarr_input_path if skim_mode == "zarr" else None,
        )

        success = self.run_container(
            client=None,
            settings=settings,
            image=activity_demand_image,
            volumes=asim_docker_vols,
            command=asim_cmd,
            model_name="activitysim_numba_warmup",
            working_dir=asim_workdir,
            args=additional_args,
            environment=container_environment,
            output_paths=[compile_output_dir],
            lineage_mode="none",
        )

        if not success:
            raise RuntimeError("ActivitySim Numba warmup failed")

        logger.info("ActivitySim Numba warmup completed in %s", compile_output_dir)


class ActivitysimRunner(GenericRunner):
    """
    Runner for ActivitySim model.
    """

    @staticmethod
    def declared_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this runner expects without disk checks.
        """
        del settings, state
        staged_data_dir = Path(workspace.full_path) / "activitysim" / "native-launch" / "data"
        return {
            ASIM_LAND_USE_IN: staged_data_dir / "land_use.csv",
            ASIM_HOUSEHOLDS_IN: staged_data_dir / "households.csv",
            ASIM_PERSONS_IN: staged_data_dir / "persons.csv",
            ASIM_OMX_SKIMS: staged_data_dir / "skims.omx",
            ZARR_SKIMS: staged_data_dir / "skims.zarr",
        }

    @staticmethod
    def runtime_expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare runtime expected inputs, including filesystem presence checks.
        """
        inputs = ActivitysimRunner.declared_expected_inputs(settings, state, workspace)
        inputs[ZARR_SKIMS] = (
            inputs[ZARR_SKIMS]
            if inputs.get(ZARR_SKIMS) and os.path.exists(inputs[ZARR_SKIMS])
            else None
        )
        return inputs

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        return ActivitysimRunner.runtime_expected_inputs(settings, state, workspace)

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this runner produces.

        Notes
        -----
        Output keys
            - ``asim_output_dir``: ActivitySim output directory for the run.
            - required ``*_asim_out`` keys: Canonical final_pipeline parquet
              outputs for the stable ActivitySim handoff surface.
        Related docs
            - See `pilates/activitysim/inputs.py` for the corresponding input
              descriptions used by ActivitySim and downstream models.
        """
        del state
        outputs: Dict[str, Any] = {"asim_output_dir": workspace.get_asim_output_dir()}
        outputs.update(asim_required_run_output_paths(settings, workspace))
        return outputs

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_files = [
            ASIM_PERSONS_IN,
            ASIM_HOUSEHOLDS_IN,
            ASIM_LAND_USE_IN,
            ASIM_OMX_SKIMS,
            "zarr_skims",
            "asim_geoms",
        ]

    @staticmethod
    def _allocate_private_compile_epoch(
        launch_context: ActivitySimLaunchContext,
    ) -> Path:
        """Create a fresh process-private cache root below the launch cache."""
        epoch_parent = launch_context.shared_cache_dir / "numba" / "compile-epochs"
        epoch_parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="epoch-", dir=epoch_parent))

    def _prepare_body_launch(
        self,
        inputs: ActivitySimPreprocessOutputs,
        launch_context: ActivitySimLaunchContext,
        *,
        skim_mode: ActivitysimSkimMode,
        zarr_input_path: Optional[str],
        skip_numba_warmup: bool,
        workspace: Optional[Workspace],
    ) -> ActivitySimPreparationResult:
        """Prepare the private compile epoch and return the body launch choice.

        This runner-private seam has no Consist call or workflow state
        transition. It carries the already selected skim mode/path straight to
        the body while replacing only its non-identity cache root.
        """
        if skip_numba_warmup:
            logger.info(
                "ActivitySim Numba warmup: not required (explicit rewind skip)"
            )
            return ActivitySimPreparationResult(
                launch_context=launch_context,
                skim_mode=skim_mode,
                zarr_input_path=zarr_input_path,
                generated_zarr_path=None,
            )
        if not persist_sharrow_cache_enabled(self.state.full_settings):
            logger.info(
                "ActivitySim Numba warmup: not required (persistent cache disabled)"
            )
            return ActivitySimPreparationResult(
                launch_context=launch_context,
                skim_mode=skim_mode,
                zarr_input_path=zarr_input_path,
                generated_zarr_path=None,
            )
        if self.state.full_settings.activitysim.num_processes <= 1:
            logger.info(
                "ActivitySim Numba warmup: not required (single-process run)"
            )
            return ActivitySimPreparationResult(
                launch_context=launch_context,
                skim_mode=skim_mode,
                zarr_input_path=zarr_input_path,
                generated_zarr_path=None,
            )

        existing_epoch = _ACTIVITYSIM_COMPILE_EPOCHS.get(self.state)
        if existing_epoch is not None:
            logger.info("ActivitySim Numba warmup: reused private epoch")
            return ActivitySimPreparationResult(
                launch_context=replace(
                    launch_context, shared_cache_dir=existing_epoch
                ),
                skim_mode=skim_mode,
                zarr_input_path=zarr_input_path,
                generated_zarr_path=None,
            )

        epoch_context = replace(
            launch_context,
            shared_cache_dir=self._allocate_private_compile_epoch(launch_context),
        )
        logger.info("ActivitySim Numba warmup: new private epoch")
        ActivitysimNumbaWarmup("activitysim_numba_warmup", self.state).run(
            inputs,
            epoch_context,
            skim_mode=skim_mode,
            zarr_input_path=zarr_input_path,
        )
        if not _dir_contains_files(str(epoch_context.shared_cache_dir / "numba")):
            raise RuntimeError(
                "ActivitySim Numba warmup completed without a nonempty private cache."
            )
        generated_zarr_path: Optional[Path] = None
        body_skim_mode = skim_mode
        body_zarr_input_path = zarr_input_path
        if skim_mode == "omx":
            if workspace is None:
                raise RuntimeError(
                    "ActivitySim OMX preparation requires a workspace for skim finalization"
                )
            rejection = validate_activitysim_zarr_skims(
                epoch_context.runtime_zarr_path
            )
            if rejection is not None:
                raise RuntimeError(
                    "ActivitySim warmup generated invalid runtime Zarr skims: "
                    f"{rejection}"
                )
            generated_zarr_path = finalize_activitysim_zarr_skims(
                epoch_context.runtime_zarr_path,
                self.state.full_settings,
                workspace,
            )
            rejection = validate_activitysim_zarr_skims(generated_zarr_path)
            if rejection is not None:
                raise RuntimeError(
                    "ActivitySim warmup generated invalid runtime Zarr skims: "
                    f"{rejection}"
                )
            body_skim_mode = "zarr"
            body_zarr_input_path = str(generated_zarr_path)
        _ACTIVITYSIM_COMPILE_EPOCHS.record(
            self.state, epoch_context.shared_cache_dir
        )
        return ActivitySimPreparationResult(
            launch_context=epoch_context,
            skim_mode=body_skim_mode,
            zarr_input_path=body_zarr_input_path,
            generated_zarr_path=generated_zarr_path,
        )

    def run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        launch_context: ActivitySimLaunchContext,
        *,
        skim_mode: ActivitysimSkimMode = "zarr",
        extra_inputs: Optional[Mapping[str, Any]] = None,
        skip_numba_warmup: bool = False,
        workspace: Optional[Workspace] = None,
    ) -> ActivitySimRunOutputs:
        if not isinstance(inputs, ActivitySimPreprocessOutputs):
            raise TypeError(
                "ActivitysimRunner.run expects ActivitySimPreprocessOutputs"
            )
        self.state.set_sub_stage_progress("runner")
        staged_extra_inputs: Dict[str, Any] = {}
        for key, value in (extra_inputs or {}).items():
            input_path = (
                value if isinstance(value, str) else getattr(value, "path", value)
            )
            if input_path is None:
                continue
            staged_extra_inputs[key] = _stage_runtime_input_path(
                key=key,
                input_path=str(input_path),
                runtime_zarr_path=launch_context.runtime_zarr_path,
            )
        if skim_mode == "zarr":
            zarr_input_path = staged_extra_inputs.get(ZARR_SKIMS)
            if zarr_input_path is None or not os.path.exists(zarr_input_path):
                raise RuntimeError(
                    "ActivitySim Zarr mode requires a staged, existing Zarr skim input."
                )
            staged_extra_inputs[ZARR_SKIMS] = zarr_input_path
        preparation = self._prepare_body_launch(
            inputs,
            launch_context,
            skim_mode=skim_mode,
            zarr_input_path=(
                staged_extra_inputs.get(ZARR_SKIMS) if skim_mode == "zarr" else None
            ),
            skip_numba_warmup=skip_numba_warmup,
            workspace=workspace,
        )
        if preparation.zarr_input_path is not None:
            staged_extra_inputs[ZARR_SKIMS] = preparation.zarr_input_path
        outputs = self._run(
            inputs,
            preparation.launch_context,
            skim_mode=preparation.skim_mode,
            extra_inputs=staged_extra_inputs,
        )
        if preparation.generated_zarr_path is not None:
            outputs.zarr_skims = preparation.generated_zarr_path
        elif preparation.skim_mode == "omx":
            if workspace is None:
                raise RuntimeError(
                    "ActivitySim OMX mode requires a workspace for skim finalization"
                )
            outputs.zarr_skims = finalize_activitysim_zarr_skims(
                str(preparation.launch_context.runtime_zarr_path),
                self.state.full_settings,
                workspace,
            )
        return outputs

    @staticmethod
    def get_base_asim_cmd(
        settings: PilatesConfig, household_sample_size=None, num_processes=None
    ):
        formattable_asim_cmd = settings.activitysim.command_template
        if not household_sample_size:
            household_sample_size = settings.activitysim.household_sample_size
        num_processes = num_processes or settings.activitysim.num_processes
        chunk_size = settings.activitysim.chunk_size  # default no chunking
        base_asim_cmd = formattable_asim_cmd.format(
            household_sample_size, num_processes, chunk_size
        )
        return base_asim_cmd

    @staticmethod
    def get_asim_additional_args(settings: PilatesConfig, asim_docker_vols, compile):
        additional_args = []
        if settings.activitysim.file_format == "parquet":
            if persist_sharrow_cache_enabled(settings):
                additional_args.append("--persist-sharrow-cache")
            data_dirs = []
            output_dirs = []
            main_config_dirs = []
            mp_config_dirs = []
            compile_config_dirs = []
            for local, d in asim_docker_vols.items():
                if d["bind"].endswith("/output/cache"):
                    continue
                if "data" in d["bind"]:
                    data_dirs.append(d["bind"])
                elif "output" in d["bind"]:
                    output_dirs.append(d["bind"])
                elif "configs_mp" in d["bind"]:
                    mp_config_dirs.append(d["bind"])
                elif "compile" in d["bind"]:
                    compile_config_dirs.append(d["bind"])
                elif "configs" in d["bind"]:
                    main_config_dirs.append(d["bind"])
            for bind in data_dirs:
                additional_args.extend(["-d", bind])
            for bind in output_dirs:
                additional_args.extend(["-o", bind])
            if compile:
                for bind in compile_config_dirs:
                    additional_args.extend(["-c", bind])
                for bind in main_config_dirs:
                    additional_args.extend(["-c", bind])
            else:
                for bind in main_config_dirs:
                    additional_args.extend(["-c", bind])
                for bind in mp_config_dirs:
                    additional_args.extend(["-c", bind])
        return additional_args

    @staticmethod
    def get_asim_docker_vols(
        settings: PilatesConfig,
        working_dir=None,
        *,
        launch_context: Optional[ActivitySimLaunchContext] = None,
        workspace: Optional[Workspace] = None,
        output_dir: Optional[str] = None,
        runtime_cache_dir: Optional[str] = None,
        runtime_cache_mode: Literal["ro", "rw"] = "rw",
    ):
        if runtime_cache_mode not in ("ro", "rw"):
            raise ValueError(
                "ActivitySim runtime cache mount mode must be either 'ro' or 'rw'."
            )
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_remote_workdir = os.path.join("/activitysim", asim_subdir)
        configured_runtime_cache_dir = runtime_cache_dir
        if launch_context is not None:
            asim_local_mutable_data_folder = os.path.abspath(
                str(launch_context.mutable_data_dir)
            )
            asim_local_output_folder = os.path.abspath(
                output_dir if output_dir is not None else str(launch_context.output_dir)
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    str(launch_context.mutable_configs_dir),
                    settings.activitysim.main_configs_dir,
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    str(launch_context.mutable_configs_dir),
                    "configs_sh_compile",
                )
            )
            asim_local_configs_mp_folder = os.path.abspath(
                os.path.join(
                    str(launch_context.mutable_configs_dir),
                    "configs_mp",
                )
            )
            if configured_runtime_cache_dir is None:
                configured_runtime_cache_dir = os.path.abspath(
                    str(launch_context.runtime_cache_dir)
                )
            if launch_context.requires_staged_config_dirs:
                required_config_dirs = (
                    settings.activitysim.main_configs_dir,
                    "configs_mp",
                    "configs_sh_compile",
                )
                for dirname in dict.fromkeys(required_config_dirs):
                    config_dir = launch_context.mutable_configs_dir / dirname
                    if not config_dir.is_dir():
                        raise RuntimeError(
                            "ActivitySim launch context is missing staged configuration "
                            f"directory: {config_dir}"
                        )
        elif workspace is not None:
            asim_local_mutable_data_folder = os.path.abspath(
                workspace.get_asim_mutable_data_dir()
            )
            asim_local_output_folder = os.path.abspath(
                output_dir
                if output_dir is not None
                else workspace.get_asim_output_dir()
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    workspace.get_asim_mutable_configs_dir(),
                    settings.activitysim.main_configs_dir,
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    workspace.get_asim_mutable_configs_dir(),
                    "configs_sh_compile",
                )
            )
            asim_local_configs_mp_folder = os.path.abspath(
                os.path.join(
                    workspace.get_asim_mutable_configs_dir(),
                    "configs_mp",
                )
            )
            if configured_runtime_cache_dir is None:
                configured_runtime_cache_dir = os.path.abspath(
                    workspace.get_asim_runtime_cache_dir()
                )
        elif working_dir is not None:
            asim_local_mutable_data_folder = os.path.abspath(
                os.path.join(
                    working_dir, settings.activitysim.local_mutable_data_folder
                )
            )
            asim_local_output_folder = os.path.abspath(
                os.path.join(working_dir, settings.activitysim.local_output_folder)
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings.activitysim.local_mutable_configs_folder,
                    settings.activitysim.main_configs_dir,
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings.activitysim.local_mutable_configs_folder,
                    "configs_sh_compile",
                )
            )
            asim_local_configs_mp_folder = os.path.abspath(
                os.path.join(
                    working_dir,
                    settings.activitysim.local_mutable_configs_folder,
                    "configs_mp",
                )
            )
        else:
            asim_local_mutable_data_folder = os.path.abspath(
                settings.activitysim.local_mutable_data_folder
            )
            asim_local_output_folder = os.path.abspath(
                settings.activitysim.local_output_folder
            )
            asim_local_configs_folder = os.path.abspath(
                os.path.join(
                    settings.activitysim.local_configs_folder, region, "configs"
                )
            )
            asim_local_configs_compile_folder = os.path.abspath(
                os.path.join(
                    settings.activitysim.local_configs_folder,
                    region,
                    "configs_sh_compile",
                )
            )
            asim_local_configs_mp_folder = os.path.abspath(
                os.path.join(
                    settings.activitysim.local_configs_folder,
                    region,
                    "configs_mp",
                )
            )
        asim_remote_input_folder = os.path.join(asim_remote_workdir, "data")
        asim_remote_output_folder = os.path.join(asim_remote_workdir, "output")
        asim_remote_configs_folder = os.path.join(asim_remote_workdir, "configs")
        asim_remote_configs_compile_folder = os.path.join(
            asim_remote_workdir, "configs_sh_compile"
        )
        asim_remote_configs_mp_folder = os.path.join(asim_remote_workdir, "configs_mp")
        asim_remote_runtime_cache_folder = os.path.join(
            asim_remote_output_folder,
            "cache",
        )
        asim_docker_vols = {
            asim_local_mutable_data_folder: {
                "bind": asim_remote_input_folder,
                "mode": "ro" if launch_context is not None else "rw",
            },
            asim_local_output_folder: {"bind": asim_remote_output_folder, "mode": "rw"},
            asim_local_configs_mp_folder: {
                "bind": asim_remote_configs_mp_folder,
                "mode": "ro" if launch_context is not None else "rw",
            },
            asim_local_configs_compile_folder: {
                "bind": asim_remote_configs_compile_folder,
                "mode": "ro" if launch_context is not None else "rw",
            },
            asim_local_configs_folder: {
                "bind": asim_remote_configs_folder,
                "mode": "ro" if launch_context is not None else "rw",
            },
        }
        default_runtime_cache_dir = os.path.abspath(
            os.path.join(asim_local_output_folder, "cache")
        )
        if (
            configured_runtime_cache_dir
            and (
                os.path.abspath(configured_runtime_cache_dir)
                != default_runtime_cache_dir
                or runtime_cache_mode == "ro"
            )
        ):
            asim_docker_vols[os.path.abspath(configured_runtime_cache_dir)] = {
                "bind": asim_remote_runtime_cache_folder,
                "mode": runtime_cache_mode,
            }
        return asim_docker_vols

    def _parse_year_iteration_from_short_name(self, short_name: str) -> Tuple[int, int]:
        parts = short_name.split("_")
        if len(parts) >= 3 and parts[0] == "zarr" and parts[1] == "skims":
            try:
                year = int(parts[2])
                iteration = int(parts[3])
                return year, iteration
            except ValueError:
                pass
        return 0, 0  # Default or error case

    def _run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        launch_context: ActivitySimLaunchContext,
        *,
        skim_mode: ActivitysimSkimMode,
        extra_inputs: Optional[Mapping[str, Any]] = None,
    ) -> ActivitySimRunOutputs:
        """
        Do the model run

        Args:
            inputs (ActivitySimPreprocessOutputs): The typed input data generated
                by the preprocessor.
            workspace (Workspace): The workspace object for path management.
            extra_inputs (Mapping[str, Any], optional): Additional runtime inputs.

        Returns:
            ActivitySimRunOutputs: The raw output files prepared by the model run.
        """
        settings = self.state.full_settings
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)

        # Get from your config
        # Create shared cache and tmp inside the resolved launch roots.
        shared_cache_dir = str(launch_context.shared_cache_dir)
        shared_tmp_dir = str(launch_context.shared_tmp_dir)

        # Create them
        os.makedirs(os.path.join(shared_cache_dir, "numba"), exist_ok=True)
        os.makedirs(shared_tmp_dir, exist_ok=True)

        client = None  # Handled by Consist

        asim_docker_vols = self.get_asim_docker_vols(
            settings,
            launch_context=launch_context,
            runtime_cache_mode="ro" if skim_mode == "zarr" else "rw",
        )

        asim_docker_vols.update(
            {
                shared_tmp_dir: {"bind": "/tmp", "mode": "rw"},
                shared_cache_dir: {"bind": "/app/numba_cache", "mode": "rw"},
            }
        )

        activity_demand_model, activity_demand_image = self.get_model_and_image(
            settings, "activity_demand_model"
        )

        all_skims_path = str(launch_context.runtime_zarr_path)

        asim_local_output_folder = os.path.abspath(str(launch_context.output_dir))

        os.makedirs(
            os.path.join(asim_local_output_folder, "cache", "numba"), exist_ok=True
        )

        zarr_input_path = None
        if skim_mode == "zarr":
            zarr_value = (extra_inputs or {}).get(ZARR_SKIMS)
            zarr_input_path = (
                str(zarr_value) if zarr_value is not None else all_skims_path
            )
            if not os.path.exists(zarr_input_path):
                raise RuntimeError(
                    "ActivitySim Zarr mode requires a staged, existing Zarr skim input."
                )

        if zarr_input_path is None:
            logger.warning(
                "No ASIM skims cache found at: {0}. OMX skims will be used.".format(
                    all_skims_path
                )
            )

        asim_cmd = self.get_base_asim_cmd(settings)
        container_environment = _asim_container_environment()
        _log_activitysim_launch_context(
            image=activity_demand_image,
            environment=container_environment,
            working_dir=asim_workdir,
            volumes=asim_docker_vols,
            zarr_input_path=zarr_input_path,
        )

        additional_args = self.get_asim_additional_args(
            settings, asim_docker_vols, False
        )

        # Clear any stale success marker before running ActivitySim.
        clear_asim_run_marker(
            str(launch_context.output_dir),
            self.state.current_year,
            self.state.current_inner_iter,
        )

        success = self.run_container(
            client=client,
            settings=settings,
            image=activity_demand_image,
            volumes=asim_docker_vols,
            command=asim_cmd,
            model_name="activitysim",
            working_dir=asim_workdir,
            args=additional_args,
            environment=container_environment,
            output_paths=[str(launch_context.output_dir)],
            lineage_mode="none",
        )

        if not success:
            message = "ASIM run failed for year {0} iteration {1}".format(
                self.state.current_year, self.state.current_inner_iter
            )
            logger.error(message)
            raise RuntimeError(message)

        # Assemble outputs from final_pipeline parquet files.
        output_dir = os.path.join(str(launch_context.output_dir), "final_pipeline")
        raw_outputs: Dict[str, Path] = {}
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                fpath = os.path.join(output_dir, fname, "final.parquet")
                if os.path.isfile(fpath):
                    raw_outputs[fname + "_asim_out_temp"] = Path(fpath)

        if raw_outputs:
            write_asim_run_marker(
                str(launch_context.output_dir),
                self.state.current_year,
                self.state.current_inner_iter,
                meta={
                    "model": "activitysim",
                    "output_tables": list(raw_outputs),
                },
            )
        else:
            logger.warning(
                "ASIM run succeeded but no final_pipeline outputs were found; "
                "skipping success marker for year %s iteration %s.",
                self.state.current_year,
                self.state.current_inner_iter,
            )

        return ActivitySimRunOutputs(
            output_dir=launch_context.output_dir,
            raw_outputs=raw_outputs,
        )
