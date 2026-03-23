from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import csv
import yaml

from pilates.generic.runner import GenericRunner
from pilates.impacts.outputs import ImpactsPreprocessOutputs, ImpactsRunOutputs
from pilates.workspace import Workspace


class ImpactsRunner(GenericRunner[ImpactsPreprocessOutputs, ImpactsRunOutputs]):
    """Docker-backed impacts runner scaffold."""

    def _run(
        self,
        store: ImpactsPreprocessOutputs,
        workspace: Workspace,
    ) -> ImpactsRunOutputs:
        settings = self.state.full_settings
        cfg = settings.impacts
        if cfg is None:
            raise ValueError("Impacts config is missing")
        if not isinstance(store, ImpactsPreprocessOutputs):
            raise TypeError("ImpactsRunner._run expects ImpactsPreprocessOutputs")

        output_dir = Path(workspace.get_impacts_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = output_dir / cfg.run_manifest_filename
        raw_exposure_table = output_dir / cfg.raw_exposure_output_filename

        _, image = self.get_model_and_image(settings, "impacts")
        container_input_dir = cfg.container_input_folder
        container_output_dir = cfg.container_output_folder
        container_input_manifest = str(Path(container_input_dir) / cfg.input_manifest_filename)
        container_output_manifest = str(Path(container_output_dir) / cfg.run_manifest_filename)
        container_exposure_output = str(
            Path(container_output_dir) / cfg.raw_exposure_output_filename
        )

        command = cfg.command_template.format(
            input_dir=store.input_dir,
            output_dir=output_dir,
            input_manifest=store.input_manifest,
            output_manifest=run_manifest,
            exposure_output=raw_exposure_table,
            container_input_dir=container_input_dir,
            container_output_dir=container_output_dir,
            container_input_manifest=container_input_manifest,
            container_output_manifest=container_output_manifest,
            container_exposure_output=container_exposure_output,
        )

        manifest_payload: Dict[str, Any] = {
            "model": "impacts",
            "status": "prepared",
            "image": image,
            "command": command,
            "input_manifest": str(store.input_manifest),
            "raw_exposure_table": str(raw_exposure_table),
        }

        if getattr(settings.run, "use_stubs", False):
            with raw_exposure_table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "cell_id",
                        "x",
                        "y",
                        "exposure_value",
                        "population_total",
                        "population_mix",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cell_id": "stub-cell-1",
                        "x": 0,
                        "y": 0,
                        "exposure_value": 0.0,
                        "population_total": 0,
                        "population_mix": "{}",
                    }
                )
            manifest_payload["status"] = "stubbed"
        else:
            volumes = {
                str(store.input_dir.resolve()): {
                    "bind": container_input_dir,
                    "mode": "rw",
                },
                str(output_dir.resolve()): {
                    "bind": container_output_dir,
                    "mode": "rw",
                },
            }
            success = self.run_container(
                client=None,
                settings=settings,
                image=image,
                volumes=volumes,
                command=command,
                model_name=self.model_name,
                input_artifacts=[str(store.input_manifest)],
                output_paths=[str(raw_exposure_table)],
            )
            if not success:
                raise RuntimeError("Impacts container execution failed")
            manifest_payload["status"] = "completed"

        run_manifest.write_text(
            yaml.safe_dump(manifest_payload, sort_keys=True),
            encoding="utf-8",
        )

        if not raw_exposure_table.exists():
            raise RuntimeError(
                f"Impacts raw exposure output was not created: {raw_exposure_table}"
            )

        return ImpactsRunOutputs(
            output_dir=output_dir,
            run_manifest=run_manifest,
            raw_exposure_table=raw_exposure_table,
            docker_command=command,
        )
