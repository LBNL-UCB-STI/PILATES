"""Validated logical paths for ActivitySim configuration roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True, slots=True)
class ActivitySimConfigRoot:
    """One normalized path relative to the ActivitySim configuration tree."""

    relative_path: PurePosixPath

    @classmethod
    def parse(cls, value: str) -> "ActivitySimConfigRoot":
        if not isinstance(value, str) or not value:
            raise ValueError(
                "ActivitySim config root must be a non-empty logical relative path"
            )
        windows_path = PureWindowsPath(value)
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or "\\" in value
            or path == PurePosixPath(".")
            or ".." in path.parts
        ):
            raise ValueError(
                "ActivitySim config root must be a logical relative path without "
                f"absolute or parent traversal components: {value!r}"
            )
        return cls(relative_path=path)

    def as_posix(self) -> str:
        return self.relative_path.as_posix()

    def path_under(self, root: Path) -> Path:
        return root.joinpath(*self.relative_path.parts)


def required_activitysim_config_roots(
    main_configs_dir: str,
) -> tuple[ActivitySimConfigRoot, ...]:
    """Return the validated config roots in adapter and launch order."""

    candidates = (
        main_configs_dir,
        "configs",
        "configs_extended",
        "configs_mp",
        "configs_sh_compile",
    )
    ordered: list[ActivitySimConfigRoot] = []
    seen: set[str] = set()
    for candidate in candidates:
        root = ActivitySimConfigRoot.parse(candidate)
        logical_path = root.as_posix()
        if logical_path in seen:
            continue
        seen.add(logical_path)
        ordered.append(root)
    return tuple(ordered)
