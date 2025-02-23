import docker
import logging
from pathlib import Path


def setup_logger(instance_id: str, log_file: Path, mode="w"):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger


def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def natural_sort_key(s):
    import re

    return [int(c) if c.isdigit() else c for c in re.split("(\d+)", s)]
