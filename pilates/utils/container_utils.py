def to_singularity_volumes(volumes):
    bindings = [
        f"{local_folder}:{binding['bind']}:{binding['mode']}"
        for local_folder, binding in volumes.items()
    ]
    result_str = ",".join(bindings)
    return result_str


def to_singularity_env(env):
    bindings = [f"{env_var}={value}" for env_var, value in env.items()]
    result_str = ",".join(bindings)
    return '"' + result_str + '"'
