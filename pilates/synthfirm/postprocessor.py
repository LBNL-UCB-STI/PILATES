import logging

logger = logging.getLogger(__name__)


def copy_to_commerce_demand(settings):
    logger.info("Copying synthfirm output to frism input")
