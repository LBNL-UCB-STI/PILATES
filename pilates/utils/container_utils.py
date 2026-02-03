"""
Utilities for converting Docker-style container configurations to Singularity format.
"""


def to_singularity_volumes(volumes):
    """
    Converts a dictionary of Docker-style volume mappings to a Singularity-compatible
    volume string.

    Args:
        volumes (dict): A dictionary where keys are local paths and values are
                        dictionaries containing 'bind' (container path) and 'mode'
                        (e.g., 'rw', 'ro').
                        Example: {'/local/path': {'bind': '/container/path', 'mode': 'rw'}}

    Returns:
        str: A comma-separated string of Singularity volume bindings.
             Example: '/local/path:/container/path:rw,/another/local:/another/container:ro'
    """
    bindings = [
        f"{local_folder}:{binding['bind']}:{binding['mode']}"
        for local_folder, binding in volumes.items()
    ]
    result_str = ",".join(bindings)
    return result_str


def to_singularity_env(env):
    """
    Converts a dictionary of environment variables to a Singularity-compatible
    environment string.

    Args:
        env (dict): A dictionary where keys are environment variable names and
                    values are their corresponding values.
                    Example: {'VAR1': 'value1', 'VAR2': 'value2'}

    Returns:
        str: A comma-separated string of environment variable assignments,
             enclosed in double quotes.
             Example: '"VAR1=value1,VAR2=value2"'
    """
    bindings = [f"{env_var}={value}" for env_var, value in env.items()]
    result_str = ",".join(bindings)
    return '"' + result_str + '"'
