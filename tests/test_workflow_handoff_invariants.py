from __future__ import annotations

from pathlib import Path


def test_workflow_code_uses_lossy_step_output_mapping_only_in_replay_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow_root = repo_root / "pilates" / "workflows"
    allowed_files = {
        workflow_root / "outputs_base.py",
        workflow_root / "orchestration.py",
    }

    offending_uses = []
    for path in workflow_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "step_output_mapping(" not in line:
                continue
            if path in allowed_files:
                continue
            offending_uses.append(f"{path}:{line_no}:{line.strip()}")

    assert offending_uses == [], (
        "Lossy step_output_mapping(...) should not be used in workflow runtime "
        "handoffs; use step_output_handoff_mapping(...) instead.\n"
        + "\n".join(offending_uses)
    )
