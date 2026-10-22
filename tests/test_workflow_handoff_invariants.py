from __future__ import annotations

from pathlib import Path


def test_workflow_code_does_not_use_removed_step_output_mapping_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow_root = repo_root / "pilates" / "workflows"

    offending_uses = []
    for path in workflow_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if (
                "step_output_mapping(" not in line
                and "step_output_handoff_mapping(" not in line
            ):
                continue
            offending_uses.append(f"{path}:{line_no}:{line.strip()}")

    assert offending_uses == [], (
        "Legacy step_output_mapping / step_output_handoff_mapping helpers are retired; "
        "publish and resolve artifacts via the coupler instead.\n"
        + "\n".join(offending_uses)
    )
