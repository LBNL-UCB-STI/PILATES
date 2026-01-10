"""
Execution context protocol for provenance tracking.

This module defines the minimal interface required for tracking execution context
in provenance records. Any object satisfying this protocol can be used with
provenance tracking, including PILATES WorkflowState or custom user objects.
"""

from typing import Protocol, Optional, Union, runtime_checkable
from enum import Enum


@runtime_checkable
class ExecutionContext(Protocol):
    """
    Minimal execution context for provenance tracking.

    This protocol defines the interface for tracking execution metadata
    (year, stage, iteration) in provenance records. Any object implementing
    these three properties can be used as context.

    Attributes
    ----------
    current_year : int or None
        Year of execution (e.g., 2020, 2025, 2030).
        None if year is not applicable to this execution.

    current_major_stage : str, Enum, or None
        High-level execution stage or phase (e.g., "land_use", "preprocessing").
        Can be a string or Enum (will be converted to string).
        None if stage tracking is not applicable.

    current_inner_iter : int or None
        Iteration number within current stage (0-indexed).
        For non-iterative processes, should be 0.
        None if iteration tracking is not applicable.

    Examples
    --------
    PILATES WorkflowState automatically satisfies this protocol:

    >>> from workflow_state import WorkflowState
    >>> state = WorkflowState.from_settings(settings)
    >>> isinstance(state, ExecutionContext)
    True

    Create a simple custom context:

    >>> from types import SimpleNamespace
    >>> context = SimpleNamespace(
    ...     current_year=2025,
    ...     current_major_stage="preprocessing",
    ...     current_inner_iter=0
    ... )
    >>> isinstance(context, ExecutionContext)
    True

    Use with dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class BatchContext:
    ...     current_year: int = 2025
    ...     current_major_stage: str = "batch_processing"
    ...     current_inner_iter: int = 0
    >>> context = BatchContext()
    >>> isinstance(context, ExecutionContext)
    True

    Notes
    -----
    This is a Protocol (PEP 544), not a base class. Objects don't need to
    explicitly inherit from it - they just need to have the required attributes.
    This is called "structural subtyping" or "duck typing".

    The `@runtime_checkable` decorator allows isinstance() checks at runtime,
    which is useful for validation and debugging.

    See Also
    --------
    WorkflowState : PILATES workflow state that satisfies this protocol
    """

    @property
    def current_year(self) -> Optional[int]:
        """
        Year of execution.

        For time-series simulations, this represents the simulated year.
        For batch processing, this might represent the data vintage year.

        Returns
        -------
        int or None
            Year as an integer (e.g., 2020, 2025, 2030).
            None if year is not applicable to this execution.

        Examples
        --------
        >>> context.current_year
        2025
        """
        ...

    @property
    def current_major_stage(self) -> Optional[Union[str, Enum]]:
        """
        High-level execution stage or phase.

        This represents the major component or workflow step being executed.
        Can be a plain string or an Enum value (will be converted to string
        by provenance tracker).

        Returns
        -------
        str, Enum, or None
            Stage name (e.g., "land_use", "traffic_assignment", "preprocessing").
            None if stage tracking is not applicable.

        Examples
        --------
        >>> context.current_major_stage
        'land_use'

        >>> context.current_major_stage  # Can also be Enum
        <WorkflowStage.land_use: 'land_use'>
        """
        ...

    @property
    def current_inner_iter(self) -> Optional[int]:
        """
        Iteration number within current stage.

        For iterative processes (e.g., supply-demand equilibrium, convergence
        loops), this tracks which iteration is currently executing.

        By convention, iterations are 0-indexed. Non-iterative processes
        should return 0.

        Returns
        -------
        int or None
            Iteration number (0-indexed). Typically 0 for first/only iteration.
            None if iteration tracking is not applicable.

        Examples
        --------
        >>> context.current_inner_iter  # First iteration
        0

        >>> context.current_inner_iter  # Third iteration
        2
        """
        ...


def validate_context(context: ExecutionContext) -> None:
    """
    Validate that an object properly implements ExecutionContext.

    Parameters
    ----------
    context : ExecutionContext
        Object to validate

    Raises
    ------
    TypeError
        If context doesn't satisfy ExecutionContext protocol
    AttributeError
        If required attributes are missing or not accessible

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> context = SimpleNamespace(
    ...     current_year=2025,
    ...     current_major_stage="test",
    ...     current_inner_iter=0
    ... )
    >>> validate_context(context)  # No error

    >>> bad_context = SimpleNamespace(year=2025)  # Missing attributes
    >>> validate_context(bad_context)
    AttributeError: Context missing required attribute: current_year
    """
    if not isinstance(context, ExecutionContext):
        raise TypeError(
            f"Context must satisfy ExecutionContext protocol, got {type(context)}"
        )

    # Check required attributes are accessible
    required = ["current_year", "current_major_stage", "current_inner_iter"]
    for attr in required:
        if not hasattr(context, attr):
            raise AttributeError(f"Context missing required attribute: {attr}")
        # Try accessing to ensure it's not just defined but accessible
        try:
            getattr(context, attr)
        except Exception as e:
            raise AttributeError(
                f"Context attribute '{attr}' exists but is not accessible: {e}"
            )
