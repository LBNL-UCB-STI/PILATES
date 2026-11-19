"""Validated logical paths and sharrow helpers for ActivitySim configuration roots."""

from __future__ import annotations

from pathlib import Path
import yaml

from pilates.config.activitysim import (
    ActivitySimConfig,
    ActivitySimConfigRoot,
    required_activitysim_config_roots,
)


def _extract_activitysim_config(settings: object) -> object | None:
    if settings is None:
        return None
    if isinstance(settings, ActivitySimConfig):
        return settings
    if isinstance(settings, dict):
        return settings["activitysim"] if "activitysim" in settings else settings
    try:
        settings_vars = vars(settings)
    except TypeError:
        return None
    if "activitysim" in settings_vars:
        return settings_vars.get("activitysim")
    return settings


def configured_activitysim_sharrow(settings: object) -> bool | None:
    """Return an explicit ``activitysim.sharrow`` boolean when configured."""
    cfg = _extract_activitysim_config(settings)
    if cfg is None:
        return None
    if isinstance(cfg, ActivitySimConfig):
        return cfg.sharrow
    if isinstance(cfg, dict):
        value = cfg.get("sharrow")
    else:
        try:
            value = vars(cfg).get("sharrow")
        except TypeError:
            return None
    return value if isinstance(value, bool) else None


def configured_activitysim_main_configs_dir(settings: object) -> str:
    """Return ``activitysim.main_configs_dir`` or ``'configs_extended'`` by default."""
    cfg = _extract_activitysim_config(settings)
    if cfg is None:
        return "configs_extended"
    if isinstance(cfg, ActivitySimConfig):
        return cfg.main_configs_dir
    if isinstance(cfg, dict):
        value = cfg.get("main_configs_dir")
    else:
        try:
            value = vars(cfg).get("main_configs_dir")
        except TypeError:
            value = None
    if isinstance(value, str) and value.strip():
        return ActivitySimConfigRoot.parse(value).as_posix()
    return "configs_extended"


def sync_activitysim_sharrow_configs(
    configs_root: str | Path,
    *,
    main_configs_dir: str = "configs_extended",
    sharrow: bool | None,
) -> list[Path]:
    """Synchronize an explicit ``sharrow`` boolean into staged ``settings.yaml`` files."""
    if not isinstance(sharrow, bool):
        return []
    root_path = Path(configs_root)
    if not root_path.is_dir():
        return []
    normalized_main = ActivitySimConfigRoot.parse(main_configs_dir).as_posix()
    updated_files: list[Path] = []
    for config_root in required_activitysim_config_roots(normalized_main):
        settings_path = config_root.path_under(root_path) / "settings.yaml"
        if not settings_path.is_file():
            continue
        loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        root_name = config_root.as_posix()
        if "sharrow" not in loaded and root_name not in {normalized_main, "configs"}:
            continue
        if loaded.get("sharrow") is sharrow:
            continue
        loaded["sharrow"] = sharrow
        settings_path.write_text(
            yaml.safe_dump(loaded, sort_keys=False),
            encoding="utf-8",
        )
        updated_files.append(settings_path)
    return updated_files


def resolve_activitysim_sharrow_enabled(
    settings: object,
    *,
    configs_root: str | Path | None = None,
    main_configs_dir: str | None = None,
) -> bool:
    """Resolve whether ActivitySim sharrow mode is active for ``settings``."""
    explicit = configured_activitysim_sharrow(settings)
    if isinstance(explicit, bool):
        return explicit

    if configs_root is not None:
        root_path = Path(configs_root)
        if root_path.is_dir():
            resolved_main = ActivitySimConfigRoot.parse(
                main_configs_dir or configured_activitysim_main_configs_dir(settings)
            ).as_posix()
            seen: set[str] = set()
            for candidate in (
                "configs_mp",
                resolved_main,
                "configs_extended",
                "configs",
            ):
                if candidate in seen:
                    continue
                seen.add(candidate)
                settings_path = (
                    ActivitySimConfigRoot.parse(candidate).path_under(root_path)
                    / "settings.yaml"
                )
                if not settings_path.is_file():
                    continue
                loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    continue
                sharrow_val = loaded.get("sharrow")
                if isinstance(sharrow_val, bool):
                    return sharrow_val
    return True


__all__ = [
    "ActivitySimConfigRoot",
    "configured_activitysim_main_configs_dir",
    "configured_activitysim_sharrow",
    "required_activitysim_config_roots",
    "resolve_activitysim_sharrow_enabled",
    "sync_activitysim_sharrow_configs",
]
