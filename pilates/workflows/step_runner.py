from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pilates.utils.consist_types import RunResultLike, ScenarioLike


@dataclass
class StepConfig:
    """
    Configuration bundle for a Consist scenario step.

    Parameters
    ----------
    fn : callable
        Callable executed by `scenario.run`.
    name : str
        Step name.
    model : str
        Model name for provenance.
    year : int, optional
        Simulation year.
    iteration : int, optional
        Iteration index within a year.
    inputs : dict, optional
        Input artifacts mapping.
    outputs : list of str, optional
        Output keys to declare.
    output_paths : dict, optional
        Output paths mapping keyed by output name.
    runtime_kwargs : dict, optional
        Runtime kwargs passed into `fn`.
    cache_mode : str, optional
        Consist cache mode.
    cache_hydration : str, optional
        Consist cache hydration mode.
    load_inputs : bool, optional
        Whether Consist should load inputs automatically.
    consist_kwargs : dict, optional
        Additional kwargs forwarded to `scenario.run`.
    """

    fn: Callable[..., Any]
    name: str
    model: str
    year: Optional[int] = None
    iteration: Optional[int] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[List[str]] = None
    output_paths: Optional[Dict[str, Any]] = None
    runtime_kwargs: Optional[Dict[str, Any]] = None
    cache_mode: Optional[str] = None
    cache_hydration: Optional[str] = None
    load_inputs: Optional[bool] = None
    consist_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Build a kwargs mapping for `scenario.run` while omitting unset values.

        Returns
        -------
        dict
            Keyword arguments ready to pass to `scenario.run`.
        """
        kwargs: Dict[str, Any] = {
            "fn": self.fn,
            "name": self.name,
            "model": self.model,
        }
        if self.year is not None:
            kwargs["year"] = self.year
        if self.iteration is not None:
            kwargs["iteration"] = self.iteration
        if self.inputs is not None:
            kwargs["inputs"] = self.inputs
        if self.outputs is not None:
            kwargs["outputs"] = self.outputs
        if self.output_paths is not None:
            kwargs["output_paths"] = self.output_paths
        if self.runtime_kwargs is not None:
            kwargs["runtime_kwargs"] = self.runtime_kwargs
        if self.cache_mode is not None:
            kwargs["cache_mode"] = self.cache_mode
        if self.cache_hydration is not None:
            kwargs["cache_hydration"] = self.cache_hydration
        if self.load_inputs is not None:
            kwargs["load_inputs"] = self.load_inputs
        kwargs.update(self.consist_kwargs)
        return kwargs


def run_step(scenario: ScenarioLike, config: StepConfig) -> RunResultLike:
    """
    Execute a step configuration using the provided scenario.

    Parameters
    ----------
    scenario : ScenarioLike
        Consist scenario or compatible interface.
    config : StepConfig
        Configuration for the step execution.

    Returns
    -------
    RunResultLike
        Result object from the scenario run.
    """
    return scenario.run(**config.to_kwargs())


def common_runtime_kwargs(
    *,
    settings: Any,
    state: Any,
    workspace: Any,
    **extras: Any,
) -> Dict[str, Any]:
    """
    Build runtime kwargs with the shared settings/state/workspace entries.

    Parameters
    ----------
    settings : object
        Simulation settings.
    state : object
        Workflow state.
    workspace : object
        Workspace instance.
    **extras : dict
        Additional runtime kwargs to include.

    Returns
    -------
    dict
        Runtime kwargs mapping ready for `scenario.run`.
    """
    return {
        "settings": settings,
        "state": state,
        "workspace": workspace,
        **extras,
    }


def build_step_config(
    *,
    fn: Callable[..., Any],
    name: str,
    model: str,
    state: Optional[Any] = None,
    year: Optional[int] = None,
    iteration: Optional[int] = None,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[List[str]] = None,
    output_paths: Optional[Dict[str, Any]] = None,
    runtime_kwargs: Optional[Dict[str, Any]] = None,
    cache_mode: Optional[str] = None,
    cache_hydration: Optional[str] = None,
    load_inputs: Optional[bool] = None,
    consist_kwargs: Optional[Dict[str, Any]] = None,
) -> StepConfig:
    """
    Create a StepConfig while defaulting year/iteration from state when omitted.

    Parameters
    ----------
    fn : callable
        Callable executed by `scenario.run`.
    name : str
        Step name.
    model : str
        Model name for provenance.
    state : object, optional
        Workflow state providing default year/iteration values.
    year : int, optional
        Simulation year override. Defaults to `state.year` when available.
    iteration : int, optional
        Iteration override. Defaults to `state.iteration` when available.
    inputs : dict, optional
        Input artifacts mapping.
    outputs : list of str, optional
        Output keys to declare.
    output_paths : dict, optional
        Output paths mapping keyed by output name.
    runtime_kwargs : dict, optional
        Runtime kwargs passed into `fn`.
    cache_mode : str, optional
        Consist cache mode.
    cache_hydration : str, optional
        Consist cache hydration mode.
    load_inputs : bool, optional
        Whether Consist should load inputs automatically.
    consist_kwargs : dict, optional
        Additional kwargs forwarded to `scenario.run`.

    Returns
    -------
    StepConfig
        Populated step configuration.
    """
    resolved_year = year
    resolved_iteration = iteration
    if state is not None:
        if resolved_year is None:
            resolved_year = getattr(state, "year", None)
        if resolved_iteration is None:
            resolved_iteration = getattr(state, "iteration", None)

    return StepConfig(
        fn=fn,
        name=name,
        model=model,
        year=resolved_year,
        iteration=resolved_iteration,
        inputs=inputs,
        outputs=outputs,
        output_paths=output_paths,
        runtime_kwargs=runtime_kwargs,
        cache_mode=cache_mode,
        cache_hydration=cache_hydration,
        load_inputs=load_inputs,
        consist_kwargs=consist_kwargs or {},
    )
