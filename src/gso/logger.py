import logging

log_to_console = False
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers = []

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# create file handler and set level to debug
fh = logging.FileHandler("gso.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# create console handler and set level to debug if log_to_console is True
if log_to_console:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Example log messages
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
