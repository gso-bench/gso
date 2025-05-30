import re
import time
import subprocess
import docker
import docker.errors
import logging
import traceback
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path

from gso.data.dataset import GSOInstance
from gso.constants import INSTANCE_IMAGE_BUILD_DIR, HIGH_RESOURCE_REPOS
from gso.utils.multiprocess import run_tasks_in_parallel_iter
from gso.harness.utils import setup_logger, close_logger
from gso.harness.environment.dockerfile import get_dockerfile_instance
from gso.harness.grading.evalscript import get_eval_script
from gso.harness.environment.patches import apply_patches
from gso.harness.environment.docker_utils import (
    image_exists_on_dockerhub,
    push_to_dockerhub,
    remove_image,
    cleanup_container,
)


@dataclass
class BuildPushConfig:
    image_name: str
    setup_scripts: dict
    dockerfile: str
    platform: str
    build_dir: Path
    instance_id: str
    dockerhub_id: str
    push_to_registry: bool
    force_rebuild: bool
    resource_level: str = "low"


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    build_dir: Path,
    nocache: bool = False,
):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """

    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    start_time = time.time()
    client = docker.from_env()
    logger = setup_logger(image_name, build_dir / "build_image.log")

    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    try:
        # Write the setup scripts to the build directory
        for script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)

            if script_name not in dockerfile and "gso_test" not in script_name:
                logger.warning(
                    f"Setup script {script_name} may not be used in Dockerfile"
                )

        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        print(f"Building {image_name} in {build_dir} with platform {platform}")
        logger.info(f"Building {image_name} in {build_dir} with platform {platform}")

        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
        )

        buildlog = ""
        for chunk in response:
            if "stream" in chunk:
                chunk_stream = ansi_escape.sub("", chunk["stream"])
                logger.info(chunk_stream.strip())
                buildlog += chunk_stream
            elif "errorDetail" in chunk:
                logger.error(
                    f"Error: {ansi_escape.sub('', chunk['errorDetail']['message'])}"
                )
                raise docker.errors.BuildError(
                    chunk["errorDetail"]["message"], buildlog
                )
        logger.info("Image built successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess error during {image_name} build: {e}")
        raise e
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise e
    finally:
        end_time = time.time()
        build_time = end_time - start_time
        print(f"Build time for {image_name}: {timedelta(seconds=build_time)}")
        logger.info(f"Build time for {image_name}: {timedelta(seconds=build_time)}")
        close_logger(logger)


def build_and_push_mp_helper(config: BuildPushConfig) -> str:
    client = docker.from_env(timeout=600)

    # check if already on dockerhub if not rebuilding
    if (
        (not config.force_rebuild)
        and config.push_to_registry
        and image_exists_on_dockerhub(config.image_name)
    ):
        print(f"Image {config.image_name} already exists on DockerHub, skipping ...")
        return config.image_name

    # check if inst image exists locally
    image_exists = False
    try:
        instance_image = client.images.get(config.image_name)
        image_exists = True
    except docker.errors.ImageNotFound:
        pass

    # build instance image
    if config.force_rebuild or (not image_exists):
        build_image(
            image_name=config.image_name,
            setup_scripts=config.setup_scripts,
            dockerfile=config.dockerfile,
            platform=config.platform,
            build_dir=config.build_dir,
        )
    else:
        print(f"Instance image {config.image_name} exists, skipping build.")

    # push to dockerhub
    if config.push_to_registry:
        push_to_dockerhub(client, config.image_name, force_push=config.force_rebuild)
        remove_image(client, config.image_name, None)  # delete local image

    return config.image_name


def build_instance_images(
    dataset: list[GSOInstance],
    max_workers: int = 4,
    force_rebuild: bool = False,
    push_to_registry: bool = False,
    dockerhub_id: str = "",
) -> tuple:
    """
    Builds the instance images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test insts or dataset to build images for
        max_workers (int): Maximum number of workers to use for building images
        force_rebuild (bool): Whether to force rebuild the images
        push_to_registry (bool): Whether to push images to DockerHub registry
        dockerhub_id (str): ID for DockerHub image names
    """
    print(f"Total instance images to build: {len(dataset)}")

    build_push_tasks = []
    for inst in dataset:
        # TODO: add ":inst.instance_tag_name" to the image name
        instance_image_name = (
            f"{dockerhub_id}:gso.eval.{inst.arch}.{inst.instance_id.lower()}"
            if dockerhub_id
            else inst.instance_image_key
        )

        instance_build_dir = INSTANCE_IMAGE_BUILD_DIR / inst.instance_image_key.replace(
            ":", "__"
        )

        inst.tests = apply_patches(inst.instance_id, inst.tests)

        is_high_resource = inst.repo in HIGH_RESOURCE_REPOS

        build_push_tasks.append(
            BuildPushConfig(
                image_name=instance_image_name,
                instance_id=inst.instance_id,
                setup_scripts={
                    "setup_repo.sh": inst.install_repo_script,
                    "eval.sh": get_eval_script(inst),
                    **{f"gso_test_{i}.py": ts for i, ts in enumerate(inst.tests)},
                },
                dockerfile=get_dockerfile_instance(inst.platform, inst.arch),
                platform=inst.platform,
                build_dir=instance_build_dir,
                dockerhub_id=dockerhub_id,
                push_to_registry=push_to_registry,
                force_rebuild=force_rebuild,
                resource_level="high" if is_high_resource else "low",
            )
        )

    # split tasks by resource level
    lr_tasks = [t for t in build_push_tasks if t.resource_level == "low"]
    hr_tasks = [t for t in build_push_tasks if t.resource_level == "high"]
    successful, failed = [], []

    if lr_tasks:
        results = run_tasks_in_parallel_iter(
            build_and_push_mp_helper,
            tasks=lr_tasks,
            num_workers=max_workers,
            timeout_per_task=3600,  # 1hr per image
            use_progress_bar=True,
            progress_bar_desc="Building images [type: lr]",
        )

        for task, config in zip(results, lr_tasks):
            if task.is_success():
                successful.append(config.image_name)
            else:
                failed.append(config.image_name)
                if task.is_timeout():
                    print(f"Build timed out for {config.image_name}")
                elif task.is_process_expired():
                    print(f"Process expired while building {config.image_name}")
                elif task.is_exception():
                    print(f"Error building {config.image_name}:")
                    print(task.exception_tb)

    if hr_tasks:
        results = run_tasks_in_parallel_iter(
            build_and_push_mp_helper,
            tasks=hr_tasks,
            num_workers=1,  # force sequential processing
            timeout_per_task=5400,  # increase timeout
            use_progress_bar=True,
            progress_bar_desc="Building images [type: hr]",
        )

        for task, config in zip(results, hr_tasks):
            if task.is_success():
                successful.append(config.image_name)
            else:
                failed.append(config.image_name)
                if task.is_timeout():
                    print(f"Build timed out for {config.image_name}")
                elif task.is_process_expired():
                    print(f"Process expired while building {config.image_name}")
                elif task.is_exception():
                    print(f"Error building {config.image_name}:")
                    print(task.exception_tb)

    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    return successful, failed


def create_container(
    instance: GSOInstance,
    client: docker.DockerClient,
    run_id: str,
    logger: logging.Logger,
):
    """
    Create a container for an instance using it's image.
    NOTE: expects the image to already be built and available locally.

    Args:
        instance (GSOInstance): GSOInstance to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
    """
    # create the container
    container = None
    try:
        # Create the container
        logger.info(f"Creating container for {instance.instance_id}...")

        # Define arguments for running the container
        run_args = {}
        cap_add = run_args.get("cap_add", [])

        container = client.containers.create(
            image=instance.instance_image_key,
            name=instance.get_instance_container_name(run_id),
            user="root",
            detach=True,
            command="tail -f /dev/null",
            platform=instance.platform,
            cap_add=cap_add,
        )
        logger.info(f"Container for {instance.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {instance.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise Exception(f"Error creating container for {instance.instance_id}: {e}")
