import logging

logger = logging.getLogger(__name__)


def persist_state_on_error(state, context: str) -> None:
    """Persist workflow state after a step failure."""
    logger.exception("%s failed.", context)
    if state is None:
        return
    write_state = getattr(state, "write_state", None)
    if callable(write_state):
        write_state()
