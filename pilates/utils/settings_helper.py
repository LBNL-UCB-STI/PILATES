"""
A helper utility for accessing settings from either a Pydantic model or a dict.
This provides a compatibility layer while migrating from the legacy flat dictionary
config to the new hierarchical Pydantic models.
"""
from pydantic import BaseModel

def get(settings, key, default=None):
    """
    Safely retrieve a value from a settings object (Pydantic model or dict).

    It can traverse a nested structure using dot notation for the key.
    e.g., get(settings, "run.models.land_use")

    Args:
        settings: The settings object (dict or Pydantic a BaseModel).
        key (str): The key to retrieve. Can be a.dot.separated.path.
        default: The default value to return if the key is not found.

    Returns:
        The value of the setting or the default.
    """
    if not isinstance(key, str):
        raise TypeError("Key must be a string.")

    # Treat the settings object itself as the root for traversal
    current_context = settings
    
    # Handle top-level keys that might exist in the flattened dict
    if isinstance(settings, dict) and key in settings:
        return settings.get(key, default)

    # Traverse nested structure
    try:
        for part in key.split('.'):
            if isinstance(current_context, dict):
                current_context = current_context[part]
            elif isinstance(current_context, BaseModel):
                current_context = getattr(current_context, part)
            else:
                # If we encounter a non-traversable type midway
                return default
        return current_context
    except (KeyError, AttributeError, TypeError):
        # If any part of the path doesn't exist
        return default