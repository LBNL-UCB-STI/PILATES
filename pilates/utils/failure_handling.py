import logging

logger = logging.getLogger(__name__)


def persist_state_on_error(state, context: str) -> None:
    """Log step failure context."""
    del state
    logger.exception("%s failed.", context)
