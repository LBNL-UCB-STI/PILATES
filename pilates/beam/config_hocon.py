from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from pilates.config.models import PilatesConfig


_OVERRIDE_BLOCK_BEGIN = "# BEGIN PILATES managed overrides"
_OVERRIDE_BLOCK_END = "# END PILATES managed overrides"
_OVERRIDE_BLOCK_RE = re.compile(
    rf"(?ms)^\s*{re.escape(_OVERRIDE_BLOCK_BEGIN)}\n(.*?)^\s*{re.escape(_OVERRIDE_BLOCK_END)}\s*$"
)


class BeamConfigHoconError(RuntimeError):
    """Raised when staged BEAM config HOCON handling cannot complete."""


class _RawHoconValue(str):
    """Literal HOCON scalar preserved from the managed override block."""


def beam_config_root(
    settings: PilatesConfig,
    *,
    workspace: Any = None,
    workspace_path: Optional[str | os.PathLike[str]] = None,
) -> Path:
    if workspace is not None and hasattr(workspace, "get_beam_mutable_data_dir"):
        return Path(workspace.get_beam_mutable_data_dir()) / settings.run.region
    if workspace_path is None:
        raise ValueError("workspace or workspace_path is required for BEAM config root resolution.")
    if settings.beam is None:
        raise ValueError("Beam settings are not configured.")
    return (
        Path(os.fspath(workspace_path))
        / settings.beam.local_mutable_data_folder
        / settings.run.region
    )


def beam_primary_config_path(
    settings: PilatesConfig,
    *,
    workspace: Any = None,
    workspace_path: Optional[str | os.PathLike[str]] = None,
) -> Path:
    if settings.beam is None:
        raise ValueError("Beam settings are not configured.")
    return beam_config_root(
        settings,
        workspace=workspace,
        workspace_path=workspace_path,
    ) / settings.beam.config


def beam_config_env_overrides(
    settings: PilatesConfig,
    *,
    workspace: Any = None,
    workspace_path: Optional[str | os.PathLike[str]] = None,
    config_root: Optional[Path] = None,
) -> dict[str, str]:
    resolved_config_root = (
        config_root.resolve()
        if config_root is not None
        else beam_config_root(
            settings,
            workspace=workspace,
            workspace_path=workspace_path,
        ).resolve()
    )
    pwd_candidates = [
        resolved_config_root.parent,
        resolved_config_root,
        resolved_config_root.parent.parent,
    ]
    expected_suffix = Path("input") / settings.run.region
    pwd_root = next(
        (root for root in pwd_candidates if (root / expected_suffix).exists()),
        resolved_config_root.parent,
    )
    return {
        "PWD": str(pwd_root),
        "inputDirectory": str(resolved_config_root),
    }


def load_resolved_beam_config_tree(
    config_path: Path,
    *,
    env_overrides: Mapping[str, str],
) -> dict[str, Any]:
    config_factory, hocon_converter, config_parser, non_existent_key = _require_pyhocon()

    try:
        config = config_factory.parse_file(str(config_path), resolve=False)
    except Exception as exc:
        raise BeamConfigHoconError(
            f"Failed to parse staged BEAM config {config_path}: {exc}"
        ) from exc

    injected_keys = _inject_resolution_overrides(config, env_overrides)
    try:
        config_parser.resolve_substitutions(config, accept_unresolved=False)
    except Exception as exc:
        raise BeamConfigHoconError(
            f"Failed to resolve staged BEAM config {config_path}: {exc}"
        ) from exc
    for key in injected_keys:
        _remove_dotted_key(config, key)
    return json.loads(hocon_converter.to_json(config))


def resolve_beam_config_value(
    config_path: Path,
    *,
    key: str,
    env_overrides: Mapping[str, str],
) -> Any:
    config_tree = load_resolved_beam_config_tree(
        config_path,
        env_overrides=env_overrides,
    )
    return _lookup_dotted_key(config_tree, key)


def update_staged_beam_config_value(
    config_path: Path,
    *,
    key: str,
    value: Any,
    env_overrides: Mapping[str, str],
) -> bool:
    _require_pyhocon()

    current_value = resolve_beam_config_value(
        config_path,
        key=key,
        env_overrides=env_overrides,
    )
    target_value = _resolve_value_semantics(value, env_overrides=env_overrides)
    if current_value == target_value:
        return False

    original_text = config_path.read_text(encoding="utf-8")
    managed_overrides = _parse_managed_overrides(original_text)
    managed_overrides[str(key)] = value
    override_block = _render_override_block(managed_overrides)
    rewritten_text = _replace_override_block(original_text, override_block)
    if rewritten_text == original_text:
        return False
    config_path.write_text(rewritten_text, encoding="utf-8")
    return True


def _require_pyhocon():
    try:
        from pyhocon import ConfigFactory, HOCONConverter
        from pyhocon.config_parser import ConfigParser
        from pyhocon.config_tree import NonExistentKey
    except Exception as exc:  # pragma: no cover
        raise BeamConfigHoconError(
            "pyhocon is required for staged BEAM config mutation and resolution."
        ) from exc
    return ConfigFactory, HOCONConverter, ConfigParser, NonExistentKey


def _lookup_dotted_key(mapping: Mapping[str, Any], key: str) -> Any:
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _remove_dotted_key(config: Any, key: str) -> None:
    parts = [part for part in str(key).split(".") if part]
    if not parts:
        return
    cursor = config
    parents: list[tuple[Any, str]] = []
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            return
        child = cursor.get(part)
        if not isinstance(child, dict):
            return
        parents.append((cursor, part))
        cursor = child
    if not isinstance(cursor, dict):
        return
    cursor.pop(parts[-1], None)
    for parent, part in reversed(parents):
        child = parent.get(part)
        if isinstance(child, dict) and not child:
            parent.pop(part, None)
        else:
            break


def _resolve_value_semantics(value: Any, *, env_overrides: Mapping[str, str]) -> Any:
    if not isinstance(value, str) or "${" not in value:
        return value
    config_factory, hocon_converter, config_parser, non_existent_key = _require_pyhocon()
    try:
        expression_config = config_factory.parse_string(
            f"pilates_override = {value}\n",
            resolve=False,
        )
    except Exception as exc:
        raise BeamConfigHoconError(
            f"Failed to parse staged BEAM override value {value!r}: {exc}"
        ) from exc
    injected_keys = _inject_resolution_overrides(expression_config, env_overrides)
    try:
        config_parser.resolve_substitutions(expression_config, accept_unresolved=False)
    except Exception as exc:
        raise BeamConfigHoconError(
            f"Failed to resolve staged BEAM override value {value!r}: {exc}"
        ) from exc
    for key in injected_keys:
        _remove_dotted_key(expression_config, key)
    resolved = json.loads(hocon_converter.to_json(expression_config))
    return resolved.get("pilates_override")


def _parse_managed_overrides(config_text: str) -> dict[str, Any]:
    match = _OVERRIDE_BLOCK_RE.search(config_text)
    if not match:
        return {}
    body = match.group(1).strip()
    if not body:
        return {}
    overrides: dict[str, Any] = {}
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise BeamConfigHoconError(
                f"Failed to parse managed BEAM override line: {line!r}"
            )
        key, raw_value = stripped.split("=", 1)
        overrides[key.strip()] = _RawHoconValue(raw_value.strip())
    return overrides


def _inject_resolution_overrides(
    config: Any,
    env_overrides: Mapping[str, str],
) -> list[str]:
    injected: dict[str, str] = {
        key: value for key, value in env_overrides.items() if key and value is not None
    }
    input_directory = injected.get("inputDirectory")
    if input_directory:
        injected["beam.inputDirectory"] = input_directory
    for key, value in injected.items():
        config.put(key, value)
    return list(injected)


def _flatten_mapping(
    mapping: Mapping[str, Any],
    flattened: dict[str, Any],
    prefix: Optional[str] = None,
) -> None:
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_mapping(value, flattened, prefix=path)
        else:
            flattened[path] = value


def _render_override_block(overrides: Mapping[str, Any]) -> str:
    lines = [_OVERRIDE_BLOCK_BEGIN]
    for key in sorted(overrides):
        lines.append(f"{key} = {_render_hocon_scalar(overrides[key])}")
    lines.append(_OVERRIDE_BLOCK_END)
    return "\n".join(lines) + "\n"


def _render_hocon_scalar(value: Any) -> str:
    if isinstance(value, _RawHoconValue):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and "${" in value:
        return value
    return json.dumps(value)


def _replace_override_block(config_text: str, override_block: str) -> str:
    if _OVERRIDE_BLOCK_RE.search(config_text):
        rewritten = _OVERRIDE_BLOCK_RE.sub(override_block.rstrip("\n"), config_text)
        if not rewritten.endswith("\n"):
            rewritten += "\n"
        return rewritten
    base = config_text.rstrip()
    if base:
        return f"{base}\n\n{override_block}"
    return override_block
