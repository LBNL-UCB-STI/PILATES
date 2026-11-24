from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from pilates.generic.runner import GenericRunner
from pilates.impacts.outputs import ImpactsPresimOutputs
from pilates.impacts.settings_bridge import (
    BEAM_INPUT_CONTAINER_DIR,
    INPUT_CACHE_CONTAINER_DIR,
    WORKSPACE_CONTAINER_DIR,
    _to_container_path,
    build_pilates_bridge_settings,
    resolve_shared_input_cache_folder,
)
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


class ImpactsPresimRunner(GenericRunner[None, ImpactsPresimOutputs, Workspace]):
    """Runs the IMPACTS presim stage (EMFAC activities + fleet) before BEAM."""

    def _run(
        self,
        store: Optional[Any],
        workspace: Workspace,
    ) -> ImpactsPresimOutputs:
        del store
        settings = self.state.full_settings
        cfg = settings.impacts
        if cfg is None:
            raise ValueError("Impacts config is missing")

        bridge = build_pilates_bridge_settings(
            settings, self.state, workspace, stage="presim"
        )

        bridge_path = Path(workspace.get_impacts_bridge_settings_path("presim"))
        settings_path = Path(workspace.get_impacts_settings_path("presim"))
        bridge_path.parent.mkdir(parents=True, exist_ok=True)
        bridge_path.write_text(
            yaml.safe_dump(bridge, sort_keys=False), encoding="utf-8"
        )

        bridge_container_path = _to_container_path(
            WORKSPACE_CONTAINER_DIR, str(bridge_path), workspace.full_path
        )
        settings_container_path = _to_container_path(
            WORKSPACE_CONTAINER_DIR, str(settings_path), workspace.full_path
        )

        _, image = self.get_model_and_image(settings, "impacts")
        volumes = {
            workspace.get_beam_mutable_data_dir(): {
                "bind": BEAM_INPUT_CONTAINER_DIR,
                "mode": "ro",
            },
            workspace.full_path: {"bind": WORKSPACE_CONTAINER_DIR, "mode": "rw"},
        }
        cache_folder = resolve_shared_input_cache_folder(settings)
        cache_folder.mkdir(parents=True, exist_ok=True)
        volumes[str(cache_folder)] = {
            "bind": INPUT_CACHE_CONTAINER_DIR,
            "mode": "rw",
        }

        derive_success = self.run_container(
            client=None,
            settings=settings,
            image=image,
            volumes=volumes,
            command=(
                f"derive_settings_from_pilates "
                f"--pilates-settings {bridge_container_path} "
                f"--output {settings_container_path}"
            ),
            model_name=self.model_name,
            working_dir=WORKSPACE_CONTAINER_DIR,
            input_artifacts=[str(bridge_path)],
            output_paths=[str(settings_path)],
            lineage_mode="none",
        )
        if not derive_success:
            raise RuntimeError("Impacts derive_settings_from_pilates failed")

        output_dir = self._presim_output_dir(bridge, workspace)

        run_success = self.run_container(
            client=None,
            settings=settings,
            image=image,
            volumes=volumes,
            command=f"presim --config {settings_container_path}",
            model_name=self.model_name,
            working_dir=WORKSPACE_CONTAINER_DIR,
            input_artifacts=[str(settings_path)],
            output_paths=[str(output_dir)],
            lineage_mode="none",
        )
        if not run_success:
            raise RuntimeError("Impacts presim execution failed")

        return ImpactsPresimOutputs(
            output_dir=output_dir,
            settings_path=settings_path,
            bridge_settings_path=bridge_path,
        )

    @staticmethod
    def _presim_output_dir(bridge: Dict[str, Any], workspace: Workspace) -> Path:
        impacts_output_dir = Path(workspace.get_impacts_output_dir())
        region = bridge["run"]["region"]
        output_run_name = bridge["run"]["output_run_name"]
        return impacts_output_dir / f"impacts_presim--{region}--{output_run_name}"
