from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from pilates.generic.runner import GenericRunner
from pilates.impacts.outputs import ImpactsPostsimOutputs
from pilates.impacts.settings_bridge import (
    BEAM_INPUT_CONTAINER_DIR,
    WORKSPACE_CONTAINER_DIR,
    _to_container_path,
    build_pilates_bridge_settings,
)
from pilates.workspace import Workspace

logger = logging.getLogger(__name__)


class ImpactsPostsimRunner(GenericRunner[None, ImpactsPostsimOutputs]):
    """Runs the IMPACTS postsim stage (emissions/inmap/aermod/exposure) after BEAM."""

    def _run(
        self,
        store: Optional[Any],
        workspace: Workspace,
    ) -> ImpactsPostsimOutputs:
        del store
        settings = self.state.full_settings
        cfg = settings.impacts
        if cfg is None:
            raise ValueError("Impacts config is missing")

        bridge = build_pilates_bridge_settings(settings, self.state, workspace, stage="postsim")

        bridge_path = Path(workspace.get_impacts_bridge_settings_path("postsim"))
        settings_path = Path(workspace.get_impacts_settings_path("postsim"))
        bridge_path.parent.mkdir(parents=True, exist_ok=True)
        bridge_path.write_text(yaml.safe_dump(bridge, sort_keys=False), encoding="utf-8")

        bridge_container_path = _to_container_path(
            WORKSPACE_CONTAINER_DIR, str(bridge_path), workspace.full_path
        )
        settings_container_path = _to_container_path(
            WORKSPACE_CONTAINER_DIR, str(settings_path), workspace.full_path
        )

        output_dir = self._postsim_output_dir(bridge, workspace)
        output_dir_container_path = _to_container_path(
            WORKSPACE_CONTAINER_DIR, str(output_dir), workspace.full_path
        )

        _, image = self.get_model_and_image(settings, "impacts")
        volumes = {
            workspace.get_beam_mutable_data_dir(): {
                "bind": BEAM_INPUT_CONTAINER_DIR,
                "mode": "ro",
            },
            workspace.full_path: {"bind": WORKSPACE_CONTAINER_DIR, "mode": "rw"},
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

        # Pin the postsim output directory via env var so it's deterministic
        # rather than relying on a timestamped directory name (see
        # impacts.__main__._resolve_postsim_output_root).
        run_success = self.run_container(
            client=None,
            settings=settings,
            image=image,
            volumes=volumes,
            command=f"postsim --config {settings_container_path}",
            model_name=self.model_name,
            working_dir=WORKSPACE_CONTAINER_DIR,
            environment={"IMPACTS_POSTSIM_OUTPUT_DIR": output_dir_container_path},
            input_artifacts=[str(settings_path)],
            output_paths=[str(output_dir)],
            lineage_mode="none",
        )
        if not run_success:
            raise RuntimeError("Impacts postsim execution failed")

        return ImpactsPostsimOutputs(
            output_dir=output_dir,
            settings_path=settings_path,
            bridge_settings_path=bridge_path,
            pipeline_manifest=output_dir / "pipeline_manifest.yaml",
            postprocess_manifest=output_dir / "postprocess_manifest.yaml",
            exposure_table=output_dir / "exposure" / "beam_concentration_distribution.parquet",
        )

    @staticmethod
    def _postsim_output_dir(bridge: Dict[str, Any], workspace: Workspace) -> Path:
        impacts_output_dir = Path(workspace.get_impacts_output_dir())
        region = bridge["run"]["region"]
        output_run_name = bridge["run"]["output_run_name"]
        return impacts_output_dir / f"impacts-postsim--{region}--{output_run_name}"
