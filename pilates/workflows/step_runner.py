from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from consist.types import CacheOptions, ExecutionOptions, OutputPolicyOptions

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
    input_keys : list of str, optional
        Coupler keys to use as inputs.
    outputs : list of str, optional
        Output keys to declare.
    output_paths : dict, optional
        Output paths mapping keyed by output name.
    required_outputs : list of str, optional
        Deprecated alias for ``outputs`` (kept for compatibility).
    output_missing : str, optional
        Output policy value mapped into ``OutputPolicyOptions``.
    output_mismatch : str, optional
        Output policy value mapped into ``OutputPolicyOptions``.
    runtime_kwargs : dict, optional
        Runtime kwargs mapped into ``ExecutionOptions``.
    cache_mode : str, optional
        Cache policy value mapped into ``CacheOptions``.
    cache_hydration : str, optional
        Cache policy value mapped into ``CacheOptions``.
    load_inputs : bool, optional
        Execution policy value mapped into ``ExecutionOptions``.
    consist_kwargs : dict, optional
        Additional kwargs forwarded to `scenario.run`.
    """

    fn: Callable[..., Any]
    name: str
    model: str
    year: Optional[int] = None
    iteration: Optional[int] = None
    inputs: Optional[Mapping[str, Any]] = None
    input_keys: Optional[Sequence[str]] = None
    outputs: Optional[Sequence[str]] = None
    output_paths: Optional[Mapping[str, Any]] = None
    required_outputs: Optional[Sequence[str]] = None
    output_missing: Optional[str] = None
    output_mismatch: Optional[str] = None
    runtime_kwargs: Optional[Mapping[str, Any]] = None
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
        if self.input_keys is not None:
            kwargs["input_keys"] = self.input_keys
        resolved_outputs = self.outputs
        if self.required_outputs is not None:
            resolved_outputs = self.required_outputs
        if resolved_outputs is not None:
            kwargs["outputs"] = list(resolved_outputs)
        if self.output_paths is not None:
            kwargs["output_paths"] = self.output_paths
        if (
            "output_policy" not in self.consist_kwargs
            and (self.output_missing is not None or self.output_mismatch is not None)
        ):
            kwargs["output_policy"] = OutputPolicyOptions(
                output_missing=self.output_missing,
                output_mismatch=self.output_mismatch,
            )
        if (
            "execution_options" not in self.consist_kwargs
            and (self.runtime_kwargs is not None or self.load_inputs is not None)
        ):
            kwargs["execution_options"] = ExecutionOptions(
                runtime_kwargs=self.runtime_kwargs,
                load_inputs=self.load_inputs,
            )
        if (
            "cache_options" not in self.consist_kwargs
            and (self.cache_mode is not None or self.cache_hydration is not None)
        ):
            kwargs["cache_options"] = CacheOptions(
                cache_mode=self.cache_mode,
                cache_hydration=self.cache_hydration,
            )
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
    inputs: Optional[Mapping[str, Any]] = None,
    input_keys: Optional[Sequence[str]] = None,
    outputs: Optional[Sequence[str]] = None,
    output_paths: Optional[Mapping[str, Any]] = None,
    required_outputs: Optional[Sequence[str]] = None,
    output_missing: Optional[str] = None,
    output_mismatch: Optional[str] = None,
    runtime_kwargs: Optional[Mapping[str, Any]] = None,
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
    input_keys : list of str, optional
        Coupler keys to use as inputs.
    outputs : list of str, optional
        Output keys to declare.
    output_paths : dict, optional
        Output paths mapping keyed by output name.
    required_outputs : list of str, optional
        Deprecated alias for ``outputs``.
    output_missing : str, optional
        Output policy value mapped into ``OutputPolicyOptions``.
    output_mismatch : str, optional
        Output policy value mapped into ``OutputPolicyOptions``.
    runtime_kwargs : dict, optional
        Runtime kwargs mapped into ``ExecutionOptions``.
    cache_mode : str, optional
        Cache policy value mapped into ``CacheOptions``.
    cache_hydration : str, optional
        Cache policy value mapped into ``CacheOptions``.
    load_inputs : bool, optional
        Execution policy value mapped into ``ExecutionOptions``.
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
        input_keys=input_keys,
        outputs=outputs,
        output_paths=output_paths,
        required_outputs=required_outputs,
        output_missing=output_missing,
        output_mismatch=output_mismatch,
        runtime_kwargs=runtime_kwargs,
        cache_mode=cache_mode,
        cache_hydration=cache_hydration,
        load_inputs=load_inputs,
        consist_kwargs=consist_kwargs or {},
    )
