from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DatasetManifest:
    dataset_name: str
    archive_run_dir: str
    db_path: str
    query: Dict[str, Any]
    files: Dict[str, str]
    row_counts: Dict[str, int]
    key_columns: List[str]
    notes: List[str] = field(default_factory=list)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path
