from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from pilates.workflows.outputs_base import StepOutputsBase
from pilates.workflows.artifact_keys import USIM_H5_UPDATED

if TYPE_CHECKING:
    pass


@dataclass
class AtlasPreprocessOutputs(StepOutputsBase):
    """
    Outputs from the ATLAS preprocess step.

    Attributes
    ----------
    atlas_mutable_input_dir : Path
        ATLAS mutable input directory prepared for the runner.
    prepared_inputs : dict
        Mapping of input short_name to prepared input path.
    """

    primary_output_attr: ClassVar[str] = "atlas_mutable_input_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("atlas_mutable_input_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)
    atlas_mutable_input_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)
    prepared_input_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield prepared ATLAS input records.
        """
        for key, path in self.prepared_inputs.items():
            yield key, path, f"ATLAS prepared input: {key}"


@dataclass
class AtlasRunOutputs(StepOutputsBase):
    """
    Outputs from the ATLAS run step.

    Attributes
    ----------
    atlas_output_dir : Path
        ATLAS output directory for the run.
    raw_outputs : dict
        Mapping of short_name to raw output path.
    """

    primary_output_attr: ClassVar[str] = "atlas_output_dir"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("atlas_output_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    atlas_output_dir: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield ATLAS raw output records.
        """
        for key, path in self.raw_outputs.items():
            yield key, path, f"ATLAS raw output: {key}"


@dataclass
class AtlasPostprocessOutputs(StepOutputsBase):
    """
    Outputs from the ATLAS postprocess step.

    Attributes
    ----------
    atlas_output_dir : Path
        ATLAS output directory after postprocessing.
    usim_datastore_h5 : Path, optional
        Updated UrbanSim datastore after ATLAS postprocessing.
    processed_outputs : dict
        Mapping of short_name to postprocessed output path.
    """

    primary_output_attr: ClassVar[str] = "usim_datastore_h5"
    required_path_fields: ClassVar[Tuple[str, ...]] = ("atlas_output_dir",)
    optional_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    atlas_output_dir: Path
    usim_datastore_h5: Optional[Path]
    processed_outputs: Dict[str, Path] = field(default_factory=dict)

    def _iter_record_items(self) -> Iterable[Tuple[str, Path, str]]:
        """
        Yield ATLAS postprocessed output records.
        """
        for key, path in self.processed_outputs.items():
            yield key, path, f"ATLAS postprocess output: {key}"
