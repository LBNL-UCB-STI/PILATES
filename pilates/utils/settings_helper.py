from typing import Union, Any, Dict
from pydantic import BaseModel


def get(settings: Union[BaseModel, Dict[str, Any]], path: str, default: Any = None) -> Any:
    """
    Access a nested attribute of a Pydantic model or a dictionary using a dot-separated path.
    """
    keys = path.split('.')
    value = settings
    for key in keys:
        if isinstance(value, BaseModel):
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return default
        elif isinstance(value, dict):
            if key in value:
                value = value[key]
            else:
                return default
        else:
            return default
    return value
