from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from workflow_state import WorkflowState


class AtlasSubState:
    """
    Workflow state for a single ATLAS sub-year within a larger forecast year.

    ATLAS runs multiple sub-years between the current year and forecast year.
    This state object represents a single sub-year's execution context while
    maintaining references to the parent forecast state.

    Attributes
    ----------
    year : int
        The current sub-year being simulated.
    current_year : int
        Alias for year (for consistency with WorkflowState).
    forecast_year : int
        The current sub-year (same as year, for model compatibility).
    main_forecast_year : int
        The overall forecast year from parent state.
    start_year : int
        The simulation start year (from parent state).
    full_settings : object
        Complete settings object (from parent state).
    is_start_year : callable
        Callable that checks if current year equals start year.
    """

    def __init__(self, parent_state: "WorkflowState", year: int) -> None:
        """
        Initialize AtlasSubState by copying parent state and overriding year fields.

        Parameters
        ----------
        parent_state : WorkflowState
            The parent workflow state for the forecast year.
        year : int
            The ATLAS sub-year to simulate.
        """
        self.__dict__ = parent_state.__dict__.copy()
        self.year = year
        self.current_year = year
        self.forecast_year = year
        self.main_forecast_year = parent_state.forecast_year
        self.start_year = parent_state.start_year
        self.full_settings = parent_state.full_settings
        self.is_start_year: Callable[[], bool] = (
            lambda: year == parent_state.start_year
        )
        self._parent_state = parent_state

    def set_sub_stage_progress(self, sub_stage_progress: str) -> None:
        """
        Update parent state's sub-stage progress tracking.

        Parameters
        ----------
        sub_stage_progress : str
            Progress indicator for current sub-stage.
        """
        self._parent_state.set_sub_stage_progress(sub_stage_progress)
