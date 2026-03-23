"""Impacts model scaffold."""

from .outputs import (
    ImpactsPostprocessOutputs,
    ImpactsPreprocessOutputs,
    ImpactsRunOutputs,
)
from .postprocessor import ImpactsPostprocessor
from .preprocessor import ImpactsPreprocessor
from .runner import ImpactsRunner

__all__ = [
    "ImpactsPreprocessOutputs",
    "ImpactsRunOutputs",
    "ImpactsPostprocessOutputs",
    "ImpactsPreprocessor",
    "ImpactsRunner",
    "ImpactsPostprocessor",
]
