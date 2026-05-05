from typing import Any, Dict, Optional

from pilates.utils import consist_runtime as cr


def log_inputs(inputs: Dict[str, Any], descriptions: Dict[str, Optional[str]]) -> None:
    """
    Log provided input artifacts using per-key descriptions.

    Parameters
    ----------
    inputs : dict
        Mapping of artifact keys to paths or artifact-like objects.
    descriptions : dict
        Mapping of artifact keys to log descriptions. Use None to skip logging
        for a given key.
    """
    if not inputs:
        return
    if cr.current_run_id() is None:
        # Inputs are logged when steps run; avoid logging without an active run.
        return
    for key, value in inputs.items():
        if key in descriptions and descriptions[key] is None:
            continue
        description = descriptions.get(key, "")
        cr.log_input(value, key=key, description=description)
