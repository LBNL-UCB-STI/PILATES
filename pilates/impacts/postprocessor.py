from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil
import yaml

from pilates.generic.postprocessor import GenericPostprocessor
from pilates.impacts.outputs import ImpactsPostprocessOutputs, ImpactsRunOutputs
from pilates.workspace import Workspace


class ImpactsPostprocessor(
    GenericPostprocessor[ImpactsRunOutputs, ImpactsPostprocessOutputs]
):
    """Finalize scaffolded impacts outputs into a single exposure table artifact."""

    def _postprocess(
        self,
        raw_outputs: ImpactsRunOutputs,
        workspace: Workspace,
        model_run_hash: Optional[str] = None,
    ) -> ImpactsPostprocessOutputs:
        del model_run_hash
        settings = self.state.full_settings
        cfg = settings.impacts
        if cfg is None:
            raise ValueError("Impacts config is missing")
        if not isinstance(raw_outputs, ImpactsRunOutputs):
            raise TypeError(
                "ImpactsPostprocessor._postprocess expects ImpactsRunOutputs"
            )

        output_dir = Path(workspace.get_impacts_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)

        exposure_table = output_dir / cfg.exposure_output_filename
        if raw_outputs.raw_exposure_table.resolve() != exposure_table.resolve():
            shutil.copy2(raw_outputs.raw_exposure_table, exposure_table)

        postprocess_manifest = output_dir / cfg.postprocess_manifest_filename
        postprocess_manifest.write_text(
            yaml.safe_dump(
                {
                    "model": "impacts",
                    "status": "finalized",
                    "raw_exposure_table": str(raw_outputs.raw_exposure_table),
                    "exposure_table": str(exposure_table),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return ImpactsPostprocessOutputs(
            output_dir=output_dir,
            exposure_table=exposure_table,
            postprocess_manifest=postprocess_manifest,
        )
