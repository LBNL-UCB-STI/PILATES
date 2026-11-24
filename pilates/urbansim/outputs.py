from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Tuple, TYPE_CHECKING

from pilates.workflows.outputs_base import StepOutputsBase

if TYPE_CHECKING:
    pass


@dataclass
class UrbanSimPreprocessOutputs(StepOutputsBase):
    """
    Outputs from the UrbanSim preprocess step.

    Attributes
    ----------
    usim_mutable_data_dir : Path
        UrbanSim mutable data directory prepared for the runner.
    prepared_inputs : dict
        Mapping of input short_name to prepared input path.
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ("usim_mutable_data_dir",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("prepared_inputs",)
    usim_mutable_data_dir: Path
    prepared_inputs: Dict[str, Path] = field(default_factory=dict)


@dataclass
class UrbanSimRunOutputs(StepOutputsBase):
    """
    Outputs from the UrbanSim run step.

    Attributes
    ----------
    usim_datastore_h5 : Path, optional
        UrbanSim datastore output for the forecast year.
    raw_outputs : dict
        Mapping of short_name to raw output path.
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("raw_outputs",)
    usim_datastore_h5: Path
    raw_outputs: Dict[str, Path] = field(default_factory=dict)


@dataclass
class UrbanSimPostprocessOutputs(StepOutputsBase):
    """
    Outputs from the UrbanSim postprocess step.

    Attributes
    ----------
    usim_datastore_h5 : Path, optional
        UrbanSim datastore prepared for the next iteration.
    processed_outputs : dict
        Mapping of short_name to postprocessed output path.
    """

    required_path_fields: ClassVar[Tuple[str, ...]] = ("usim_datastore_h5",)
    dict_path_fields: ClassVar[Tuple[str, ...]] = ("processed_outputs",)
    usim_datastore_h5: Path
    processed_outputs: Dict[str, Path] = field(default_factory=dict)
