import logging
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

from pilates.generic.model_factory import ModelFactory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pilates.config.models import PilatesConfig
    from pilates.workspace import Workspace
    from workflow_state import WorkflowState


def expected_inputs_for(
    model_name: str,
    settings: "PilatesConfig",
    state: "WorkflowState",
    workspace: "Workspace",
) -> Dict[str, Any]:
    """
    Collect declared expected inputs for a model from its component classes.

    Parameters
    ----------
    model_name : str
        Model key registered in the factory.
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.

    Returns
    -------
    dict
        Mapping of expected input keys to values or paths.
    """
    expected = {}
    components = ModelFactory._registry.get(model_name, {})
    for component_name in ("preprocessor", "runner", "postprocessor"):
        component_cls = components.get(component_name)
        if component_cls is None:
            continue
        expected_fn = getattr(component_cls, "expected_inputs", None)
        if callable(expected_fn):
            merge_expected_inputs(
                expected, expected_fn(settings, state, workspace) or {}
            )
    return expected


def expected_outputs_for(
    model_name: str,
    settings: "PilatesConfig",
    state: "WorkflowState",
    workspace: "Workspace",
    *,
    components: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Collect declared expected outputs for a model from selected components.

    Parameters
    ----------
    model_name : str
        Model key registered in the factory.
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    components : iterable of str, optional
        Component names to query (defaults to runner and postprocessor).

    Returns
    -------
    dict
        Mapping of expected output keys to values or paths.
    """
    expected = {}
    registry_components = ModelFactory._registry.get(model_name, {})
    component_names = components or ("runner", "postprocessor")
    for component_name in component_names:
        component_cls = registry_components.get(component_name)
        if component_cls is None:
            continue
        expected_fn = getattr(component_cls, "expected_outputs", None)
        if callable(expected_fn):
            merge_expected_outputs(
                expected, expected_fn(settings, state, workspace) or {}
            )
    return expected


def merge_expected_inputs(
    target: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    prefer_expected: bool = False,
) -> Dict[str, Any]:
    """
    Merge expected inputs into a target mapping with optional override behavior.

    Parameters
    ----------
    target : dict
        Mapping to update in place.
    expected : dict
        Expected inputs to merge into target.
    prefer_expected : bool, optional
        When True, expected values override existing keys.

    Returns
    -------
    dict
        The updated target mapping.
    """
    for key, value in expected.items():
        if value is None:
            continue
        if key in target:
            if prefer_expected:
                target[key] = value
            else:
                logger.debug(
                    "Input '%s' already provided; keeping existing value.", key
                )
            continue
        target[key] = value
    return target


def merge_expected_outputs(
    target: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    prefer_expected: bool = False,
) -> Dict[str, Any]:
    """
    Merge expected outputs into a target mapping with optional override behavior.

    Parameters
    ----------
    target : dict
        Mapping to update in place.
    expected : dict
        Expected outputs to merge into target.
    prefer_expected : bool, optional
        When True, expected values override existing keys.

    Returns
    -------
    dict
        The updated target mapping.
    """
    for key, value in expected.items():
        if value is None:
            continue
        if key in target:
            if prefer_expected:
                target[key] = value
            else:
                logger.debug(
                    "Output '%s' already provided; keeping existing value.", key
                )
            continue
        target[key] = value
    return target


def merge_model_expected_inputs(
    model_name: str,
    base_inputs: Dict[str, Any],
    settings: "PilatesConfig",
    state: "WorkflowState",
    workspace: "Workspace",
    *,
    prefer_expected: bool = False,
) -> Dict[str, Any]:
    """
    Merge model-declared expected inputs into a base input mapping.

    Parameters
    ----------
    model_name : str
        Model key registered in the factory.
    base_inputs : dict
        Pre-populated inputs to augment.
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    prefer_expected : bool, optional
        When True, expected values override existing keys.

    Returns
    -------
    dict
        Updated input mapping with expected inputs merged in.
    """
    expected = expected_inputs_for(model_name, settings, state, workspace)
    return merge_expected_inputs(base_inputs, expected, prefer_expected=prefer_expected)


def merge_expected_model_inputs(
    model_names: Iterable[str],
    base_inputs: Dict[str, Any],
    settings: "PilatesConfig",
    state: "WorkflowState",
    workspace: "Workspace",
    *,
    prefer_expected: bool = False,
) -> Dict[str, Any]:
    """
    Merge expected inputs for multiple models in sequence.

    Parameters
    ----------
    model_names : iterable of str
        Model keys registered in the factory, merged in order.
    base_inputs : dict
        Pre-populated inputs to augment.
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    prefer_expected : bool, optional
        When True, expected values override existing keys.

    Returns
    -------
    dict
        Updated input mapping with expected inputs merged in.
    """
    merged = base_inputs
    for model_name in model_names:
        merged = merge_model_expected_inputs(
            model_name,
            merged,
            settings,
            state,
            workspace,
            prefer_expected=prefer_expected,
        )
    return merged


def build_outputs(
    model_name: str,
    settings: "PilatesConfig",
    state: "WorkflowState",
    workspace: "Workspace",
    *,
    components: Optional[Iterable[str]] = None,
    prefer_expected: bool = False,
) -> Dict[str, Any]:
    """
    Build a mapping of expected outputs for a model.

    Parameters
    ----------
    model_name : str
        Model key registered in the factory.
    settings : PilatesConfig
        Parsed simulation settings.
    state : WorkflowState
        Current workflow state.
    workspace : Workspace
        Workspace instance with paths.
    components : iterable of str, optional
        Component names to query (defaults to runner and postprocessor).
    prefer_expected : bool, optional
        When True, expected values override existing keys.

    Returns
    -------
    dict
        Mapping of expected output keys to values or paths.
    """
    expected = expected_outputs_for(
        model_name, settings, state, workspace, components=components
    )
    return merge_expected_outputs({}, expected, prefer_expected=prefer_expected)
