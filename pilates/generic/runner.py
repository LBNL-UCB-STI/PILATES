from __future__ import annotations

import abc
import contextlib
import logging
import os
import shlex
from typing import Optional, List, Union, Dict, Any, Generic, TypeVar

from pilates.config import PilatesConfig
from pilates.generic.model import Model

from abc import ABC

from pilates.utils import consist_runtime as cr
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.settings_helper import get as get_setting


logger = logging.getLogger(__name__)
CONSIST_CONTAINER_DEBUG_STREAM_ENV = "CONSIST_CONTAINER_DEBUG_STREAM"


RunnerInputsT = TypeVar("RunnerInputsT")
RunnerOutputsT = TypeVar("RunnerOutputsT")


class GenericRunner(Model, ABC, Generic[RunnerInputsT, RunnerOutputsT]):
    """
    Base class for model runners with model-specific input and output types.
    """

    def __init__(
        self,
        model_name: str,
        state: "WorkflowState",
    ):
        super().__init__(model_name, state)
        self.required_input_files = []
        self.required_output_files = []

    @staticmethod
    def get_model_and_image(settings: PilatesConfig, model_type: str):
        manager = settings.infrastructure.container_manager
        if manager == "docker":
            image_names = settings.infrastructure.docker_images
        elif manager == "singularity":
            image_names = settings.infrastructure.singularity_images
        else:
            raise ValueError(
                "Container Manager not specified (container_manager param in settings.yaml)"
            )

        # Map legacy model_type keys to their new paths under run.models
        model_name_map = {
            "land_use_model": settings.run.models.land_use,
            "travel_model": settings.run.models.travel,
            "activity_demand_model": settings.run.models.activity_demand,
            "vehicle_ownership_model": settings.run.models.vehicle_ownership,
        }

        model_name = model_name_map.get(model_type)

        # Fallback for custom or non-standard model types passed in tests
        if model_name is None:
            model_name = getattr(settings.run.models, model_type, None)

        if not model_name:
            optional_models = ["vehicle_ownership_model"]
            if model_type in optional_models:
                return None, None
            else:
                raise ValueError(f"No model {model_type} specified in settings.")

        image_name = image_names.get(model_name)
        if not image_name:
            raise ValueError(
                f"No {manager} image specified for model '{model_name}' (model type: {model_type}). Check settings for '{manager}_images'."
            )

        return model_name, image_name

    def run(
        self,
        store: RunnerInputsT,
        workspace: Workspace,
    ) -> RunnerOutputsT:
        """
        Execute the model run.

        Args:
            store: The model-specific input prepared by preprocessing.
            workspace (Workspace): The workspace.

        Returns:
            The model-specific outputs prepared by the runner.
        """
        self.state.set_sub_stage_progress("runner")
        return self._run(store, workspace)

    @abc.abstractmethod
    def _run(
        self,
        store: RunnerInputsT,
        workspace: Workspace,
    ) -> RunnerOutputsT:
        """
        Do the model run.

        Subclasses should return model-specific outputs without provenance side
        effects.

        Args:
            store: The model-specific input prepared by preprocessing.
            workspace (Workspace): The workspace.

        Returns:
            The model-specific outputs prepared by the runner.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def run_container(
        client,
        settings: PilatesConfig,
        image: str,
        volumes: dict,
        command: str,
        model_name: str,
        working_dir=None,
        environment=None,
        args=None,
        input_artifacts: List[Union[str, Any]] = None,
        output_paths: List[str] = None,
        lineage_mode: str = None,
    ) -> bool:
        """
        Execute container with the Consist container integration.
        """
        run_cfg = getattr(settings, "run", None)
        use_stubs = (
            getattr(run_cfg, "use_stubs", False) if run_cfg is not None else False
        )
        if use_stubs is True:
            logger.warning(
                "[%s] use_stubs=True; skipping container execution for image=%s",
                model_name,
                image,
            )
            return True

        mounts = GenericRunner._extract_mounts(volumes)
        consist_volumes = {host: container for host, container, _mode in mounts}
        runtime_tmp_base = GenericRunner._resolve_runtime_tmp_base(mounts)

        # Handle command + args: Split if string, combine with args
        full_command_list = GenericRunner._build_full_command(command, args)

        # Determine settings from config
        backend_type = get_setting(
            settings, "infrastructure.container_manager", "docker"
        )
        pull_latest = get_setting(
            settings, "infrastructure.docker_config.pull_latest", False
        )
        docker_stdout = get_setting(
            settings, "infrastructure.docker_config.stdout", None
        )
        if docker_stdout is None:
            docker_stdout = get_setting(settings, "docker_stdout", False)
        stream_container_logs = bool(docker_stdout) or logger.isEnabledFor(
            logging.DEBUG
        )

        tracker = cr.current_tracker()
        if not tracker:
            raise RuntimeError(
                "A Consist tracker must be active for container execution. "
                "Ensure the call occurs within a Consist scenario/run context."
            )
        if not (hasattr(tracker, "mounts") and hasattr(tracker, "start_run")):
            raise RuntimeError(
                f"Current tracker type {type(tracker).__name__} does not support "
                "Consist container execution."
            )

        strict_mounts = True
        local_root = os.environ.get("PILATES_LOCAL_RUN_DIR")
        archive_root = os.environ.get("PILATES_ARCHIVE_RUN_DIR")
        if output_paths and local_root and archive_root:
            local_root = os.path.abspath(local_root)
            archive_root = os.path.abspath(archive_root)
            if local_root != archive_root:
                for path in output_paths:
                    if not isinstance(path, str) or "://" in path:
                        continue
                    abs_path = os.path.abspath(path)
                    try:
                        in_local = (
                            os.path.commonpath([abs_path, local_root]) == local_root
                        )
                        in_archive = (
                            os.path.commonpath([abs_path, archive_root])
                            == archive_root
                        )
                    except ValueError:
                        continue
                    if in_local and not in_archive:
                        strict_mounts = False
                        break

        try:
            from consist.integrations.containers import (
                run_container as consist_run_container,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Consist container integration is unavailable. "
                "A Consist install with container support is required."
            ) from exc

        logger.info("[%s] Delegating container execution to Consist", model_name)
        with GenericRunner._temporary_container_debug_stream(
            enabled=stream_container_logs
        ), GenericRunner._temporary_container_runtime_env(
            runtime_tmp_base,
            backend=backend_type,
        ):
            return consist_run_container(
                tracker=tracker,
                run_id=f"{model_name}_container",
                image=image,
                command=full_command_list,
                volumes=consist_volumes,
                inputs=input_artifacts or [],
                outputs=output_paths or [],
                environment=environment or {},
                working_dir=working_dir,
                backend_type=backend_type,
                pull_latest=pull_latest,
                lineage_mode=lineage_mode,
                strict_mounts=strict_mounts,
            )

    @staticmethod
    @contextlib.contextmanager
    def _temporary_container_debug_stream(*, enabled: bool):
        previous = os.environ.get(CONSIST_CONTAINER_DEBUG_STREAM_ENV)
        if enabled:
            os.environ[CONSIST_CONTAINER_DEBUG_STREAM_ENV] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(CONSIST_CONTAINER_DEBUG_STREAM_ENV, None)
            else:
                os.environ[CONSIST_CONTAINER_DEBUG_STREAM_ENV] = previous

    @staticmethod
    def _resolve_runtime_tmp_base(mounts: List[tuple]) -> Optional[str]:
        for host_path, container_path, _mode in mounts:
            if container_path == "/tmp":
                return os.path.join(host_path, ".container_runtime")
        return None

    @staticmethod
    @contextlib.contextmanager
    def _temporary_container_runtime_env(
        runtime_tmp_base: Optional[str], *, backend: str
    ):
        if not runtime_tmp_base or backend != "singularity":
            yield
            return

        os.makedirs(runtime_tmp_base, exist_ok=True)
        overrides = {
            "TMPDIR": runtime_tmp_base,
            "APPTAINER_CACHEDIR": os.path.join(runtime_tmp_base, ".apptainer", "cache"),
            "APPTAINER_TMPDIR": os.path.join(runtime_tmp_base, ".apptainer", "tmp"),
            "SINGULARITY_CACHEDIR": os.path.join(
                runtime_tmp_base, ".apptainer", "cache"
            ),
            "SINGULARITY_TMPDIR": os.path.join(runtime_tmp_base, ".apptainer", "tmp"),
        }
        for path in overrides.values():
            if path != runtime_tmp_base:
                os.makedirs(path, exist_ok=True)

        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        logger.info(
            "Using per-run Singularity cache/tmp base: %s",
            runtime_tmp_base,
        )
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def _build_full_command(command: Union[str, List[str]], args=None) -> List[str]:
        if isinstance(command, list):
            full_command_list = list(command)
        else:
            full_command_list = shlex.split(str(command))

        if args:
            if isinstance(args, list):
                full_command_list.extend(args)
            else:
                full_command_list.extend(shlex.split(str(args)))

        return full_command_list

    @staticmethod
    def _extract_mounts(volumes: dict) -> List[tuple]:
        mounts = []
        for host_path, mount_info in volumes.items():
            if isinstance(mount_info, dict):
                container_path = mount_info.get("bind", mount_info)
                mode = mount_info.get("mode", "rw")
            else:
                container_path = mount_info
                mode = "rw"

            if not isinstance(container_path, str):
                raise ValueError(
                    f"Invalid container path for mount {host_path!r}: {container_path!r}"
                )

            abs_host_path = os.path.abspath(host_path)
            mounts.append((abs_host_path, container_path, mode))
        return mounts
