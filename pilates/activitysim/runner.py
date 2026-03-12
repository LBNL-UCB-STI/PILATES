import logging
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Mapping

from pilates.config import PilatesConfig
from pilates.generic.runner import GenericRunner
from pilates.workspace import Workspace
from workflow_state import WorkflowState
from pilates.utils.zone_utils import ensure_0_based_and_flag_zarr_skims
from pilates.activitysim.outputs import (
    ActivitySimCompileOutputs,
    ActivitySimPreprocessOutputs,
    ActivitySimRunOutputs,
    write_asim_run_marker,
    clear_asim_run_marker,
)
from pilates.workflows.artifact_keys import (
    ASIM_HOUSEHOLDS_IN,
    ASIM_LAND_USE_IN,
    ASIM_OMX_SKIMS,
    ASIM_PERSONS_IN,
    ASIM_SHARROW_CACHE_DIR,
)

logger = logging.getLogger(__name__)


def _asim_container_environment() -> Dict[str, str]:
    """
    Environment variables passed into ActivitySim containers.

    ``PYTHONNOUSERSITE=1`` prevents host user-site packages from leaking into
    the container Python environment, which can otherwise mix incompatible
    xarray/zarr versions with the image's pinned stack.
    """
    return {
        "NUMBA_CACHE_DIR": "/app/numba_cache/numba",
        "XDG_CACHE_HOME": "/app/numba_cache",
        "PYTHONNOUSERSITE": "1",
    }


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


def _dir_contains_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def _stage_runtime_input_path(
    *,
    key: str,
    input_path: str,
    workspace: Workspace,
) -> str:
    if key != "zarr_skims":
        return input_path

    runtime_path = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
    if os.path.abspath(input_path) == os.path.abspath(runtime_path):
        return runtime_path

    os.makedirs(os.path.dirname(runtime_path), exist_ok=True)
    if os.path.isdir(input_path):
        if os.path.exists(runtime_path):
            shutil.rmtree(runtime_path)
        shutil.copytree(input_path, runtime_path)
    else:
        shutil.copyfile(input_path, runtime_path)
    return runtime_path


class ActivitysimCompileRunner(GenericRunner):
    """
    Runner that performs the one-time ActivitySim compile step.
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
    def get_asim_docker_vols(settings: PilatesConfig, working_dir=None):
        return ActivitysimRunner.get_asim_docker_vols(settings, working_dir=working_dir)

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this runner expects from the workflow.
        """
        return {}

    @staticmethod
    def expected_outputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the output paths/artifacts this runner produces.

        Notes
        -----
        Output keys
            - ``zarr_skims``: Compiled skims in Zarr format written under the
              ActivitySim output cache directory.
            - ``asim_sharrow_cache_dir``: Persisted ActivitySim numba/sharrow
              compile cache directory when ``persist_sharrow_cache`` is enabled.
        Related docs
            - See `pilates/activitysim/inputs.py` for the corresponding input
              descriptions used by ActivitySim and downstream models.
        """
        outputs: Dict[str, Any] = {
            "zarr_skims": os.path.join(
                workspace.get_asim_output_dir(), "cache", "skims.zarr"
            )
        }
        if persist_sharrow_cache_enabled(settings):
            outputs[ASIM_SHARROW_CACHE_DIR] = asim_sharrow_cache_dir(workspace)
        return outputs

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
        workspace: Workspace,
    ) -> ActivitySimCompileOutputs:
        if not isinstance(inputs, ActivitySimPreprocessOutputs):
            raise TypeError(
                "ActivitysimCompileRunner.run expects ActivitySimPreprocessOutputs"
            )
        self.state.set_sub_stage_progress("runner")
        return self._run(inputs, workspace)

    def _run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        workspace: Workspace,
    ) -> ActivitySimCompileOutputs:
        del inputs
        settings = self.state.full_settings
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_workdir = os.path.join("activitysim", asim_subdir)

        shared_cache_dir = os.path.join(workspace.full_path, "shared_cache")
        shared_tmp_dir = os.path.join(workspace.full_path, "tmp")

        os.makedirs(os.path.join(shared_cache_dir, "numba"), exist_ok=True)
        os.makedirs(shared_tmp_dir, exist_ok=True)

        asim_docker_vols = self.get_asim_docker_vols(
            settings, working_dir=workspace.full_path
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

        asim_local_output_folder = os.path.abspath(
            os.path.join(workspace.full_path, settings.activitysim.local_output_folder)
        )
        os.makedirs(
            os.path.join(asim_local_output_folder, "cache", "numba"), exist_ok=True
        )

        all_skims_path = os.path.join(
            workspace.get_asim_output_dir(), "cache", "skims.zarr"
        )

        asim_cmd = self.get_base_asim_cmd(
            settings, household_sample_size=2500, num_processes=1
        )
        additional_args = self.get_asim_additional_args(
            settings, asim_docker_vols, True
        )

        success = self.run_container(
            client=None,
            settings=settings,
            image=activity_demand_image,
            volumes=asim_docker_vols,
            command=asim_cmd,
            model_name="activitysim_compile",
            working_dir=asim_workdir,
            args=additional_args,
            environment=_asim_container_environment(),
            output_paths=[all_skims_path],
            lineage_mode="none",
        )

        if not success:
            raise RuntimeError("ASim Compilation failed")

        zarr_skims_path = None
        if os.path.exists(all_skims_path):
            try:
                ensure_0_based_and_flag_zarr_skims(all_skims_path, settings, workspace)
            except Exception as e:
                logger.error(
                    f"Failed to correct and flag initial Zarr skims after compilation: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Failed to correct initial Zarr skims, cannot proceed."
                ) from e

            zarr_skims_path = all_skims_path
            logger.info(f"Using zarr skims from ASIM compilation: {all_skims_path}")
        else:
            logger.warning("ASIM compilation succeeded but skims.zarr was not found.")

        sharrow_cache_dir = None
        if persist_sharrow_cache_enabled(settings):
            cache_dir = asim_sharrow_cache_dir(workspace)
            if _dir_contains_files(cache_dir):
                sharrow_cache_dir = cache_dir
                logger.info(
                    "ActivitySim compile cache directory is available: %s", cache_dir
                )
            elif os.path.exists(cache_dir) and not os.path.isdir(cache_dir):
                logger.warning(
                    "ActivitySim compile cache path exists but is not a directory: %s",
                    cache_dir,
                )
            elif os.path.isdir(cache_dir):
                logger.info(
                    "ActivitySim compile cache persistence is enabled, but cache "
                    "directory is empty: %s",
                    cache_dir,
                )
            else:
                logger.warning(
                    "ActivitySim compile cache persistence is enabled, but cache "
                    "directory was not found: %s",
                    cache_dir,
                )

        self.state.compile_asim()
        return ActivitySimCompileOutputs(
            zarr_skims=Path(zarr_skims_path) if zarr_skims_path is not None else None,
            sharrow_cache_dir=(
                Path(sharrow_cache_dir) if sharrow_cache_dir is not None else None
            ),
        )


class ActivitysimRunner(GenericRunner):
    """
    Runner for ActivitySim model.
    """

    @staticmethod
    def expected_inputs(
        settings: PilatesConfig, state: "WorkflowState", workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Declare the input paths/artifacts this runner expects from the workflow.
        """
        zarr_path = os.path.join(workspace.get_asim_output_dir(), "cache", "skims.zarr")
        return {
            "zarr_skims": zarr_path if os.path.exists(zarr_path) else None,
        }

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
        Related docs
            - See `pilates/activitysim/inputs.py` for the corresponding input
              descriptions used by ActivitySim and downstream models.
        """
        return {"asim_output_dir": workspace.get_asim_output_dir()}

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

    def run(
        self,
        inputs: ActivitySimPreprocessOutputs,
        workspace: Workspace,
        *,
        extra_inputs: Optional[Mapping[str, Any]] = None,
    ) -> ActivitySimRunOutputs:
        if not isinstance(inputs, ActivitySimPreprocessOutputs):
            raise TypeError(
                "ActivitysimRunner.run expects ActivitySimPreprocessOutputs"
            )
        self.state.set_sub_stage_progress("runner")
        staged_extra_inputs: Dict[str, Any] = {}
        for key, value in (extra_inputs or {}).items():
            input_path = value if isinstance(value, str) else getattr(value, "path", value)
            if input_path is None:
                continue
            staged_extra_inputs[key] = _stage_runtime_input_path(
                key=key,
                input_path=str(input_path),
                workspace=workspace,
            )
        return self._run(
            inputs,
            workspace,
            extra_inputs=staged_extra_inputs,
        )

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
    def get_asim_docker_vols(settings: PilatesConfig, working_dir=None):
        region = settings.run.region
        asim_subdir = settings.activitysim.region_mappings["region_to_subdir"][region]
        asim_remote_workdir = os.path.join("/activitysim", asim_subdir)
        if working_dir is not None:
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
        asim_docker_vols = {
            asim_local_mutable_data_folder: {
                "bind": asim_remote_input_folder,
                "mode": "rw",
            },
            asim_local_output_folder: {"bind": asim_remote_output_folder, "mode": "rw"},
            asim_local_configs_mp_folder: {
                "bind": asim_remote_configs_mp_folder,
                "mode": "rw",
            },
            asim_local_configs_compile_folder: {
                "bind": asim_remote_configs_compile_folder,
                "mode": "rw",
            },
            asim_local_configs_folder: {
                "bind": asim_remote_configs_folder,
                "mode": "rw",
            },
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
        workspace: Workspace,
        *,
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
        # Create shared cache and tmp inside the run workspace
        shared_cache_dir = os.path.join(workspace.full_path, "shared_cache")
        shared_tmp_dir = os.path.join(workspace.full_path, "tmp")

        # Create them
        os.makedirs(os.path.join(shared_cache_dir, "numba"), exist_ok=True)
        os.makedirs(shared_tmp_dir, exist_ok=True)

        client = None  # Handled by Consist

        asim_docker_vols = self.get_asim_docker_vols(
            settings, working_dir=workspace.full_path
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

        all_skims_path = os.path.join(
            workspace.get_asim_output_dir(), "cache", "skims.zarr"
        )

        asim_local_output_folder = os.path.abspath(
            os.path.join(workspace.full_path, settings.activitysim.local_output_folder)
        )

        os.makedirs(
            os.path.join(asim_local_output_folder, "cache", "numba"), exist_ok=True
        )

        zarr_input_path = None
        for key, value in (extra_inputs or {}).items():
            input_path = value if isinstance(value, str) else getattr(value, "path", value)
            if input_path is None or key != "zarr_skims":
                continue
            zarr_input_path = str(input_path)
            break

        if zarr_input_path is None and os.path.exists(all_skims_path):
            zarr_input_path = all_skims_path

        if zarr_input_path is None:
            logger.warning(
                "No ASIM skims cache found at: {0}. OMX skims will be used.".format(
                    all_skims_path
                )
            )

        asim_cmd = self.get_base_asim_cmd(settings)

        additional_args = self.get_asim_additional_args(
            settings, asim_docker_vols, False
        )

        # Clear any stale success marker before running ActivitySim.
        clear_asim_run_marker(
            workspace.get_asim_output_dir(),
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
            environment=_asim_container_environment(),
            output_paths=[workspace.get_asim_output_dir()],
            lineage_mode="none",
        )

        if not success:
            message = "ASIM run failed for year {0} iteration {1}".format(
                self.state.current_year, self.state.current_inner_iter
            )
            logger.error(message)
            raise RuntimeError(message)

        # Assemble outputs from final_pipeline parquet files.
        output_dir = os.path.join(workspace.get_asim_output_dir(), "final_pipeline")
        raw_outputs: Dict[str, Path] = {}
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                fpath = os.path.join(output_dir, fname, "final.parquet")
                if os.path.isfile(fpath):
                    raw_outputs[fname + "_asim_out_temp"] = Path(fpath)

        if raw_outputs:
            write_asim_run_marker(
                workspace.get_asim_output_dir(),
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
            output_dir=Path(workspace.get_asim_output_dir()),
            raw_outputs=raw_outputs,
        )
