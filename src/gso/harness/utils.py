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

    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


def retag_remote_to_local_image(instances, client):
    """
    Retag remote instance images to local instance images.
    """
    remote_to_local = {
        x.remote_instance_image_key: x.instance_image_key for x in instances
    }
    for remote_image in remote_to_local:
        try:
            client.images.get(remote_image).tag(remote_to_local[remote_image])
            # print(f"Retagged {remote_image} -> {remote_to_local[remote_image]}")
        except docker.errors.ImageNotFound:
            pass
