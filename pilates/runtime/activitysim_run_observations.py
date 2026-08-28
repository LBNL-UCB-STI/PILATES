"""Opt-in native execution observations for ActivitySim acceptance evidence.

Normal PILATES runs do not set the observation-path environment variable, so
the hooks below are no-ops outside the isolated acceptance driver.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal


OBSERVATION_PATH_ENV = "PILATES_ACTIVITYSIM_RUN_ACCEPTANCE_OBSERVATIONS"
ObservationEvent = Literal["activitysim_run_body", "activitysim_runner_preparation"]


def record_activitysim_observation(event: ObservationEvent) -> None:
    """Append one native seam entry when the acceptance driver opted in."""

    raw_path = os.environ.get(OBSERVATION_PATH_ENV)
    if raw_path is None or raw_path == "":
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid()}) + "\n")
