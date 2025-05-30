import docker
import resource
from argparse import ArgumentParser

from gso.utils.io import load_gso_dataset, str2bool
from gso.harness.environment.docker_build import build_instance_images


def main(
    dataset_name,
    split,
    instance_ids,
    max_workers,
    force_rebuild,
    push_to_registry,
    open_file_limit,
    dockerhub_username,
    dockerhub_repo,
):
    """Build docker images for the specified instances."""
    # Set open file limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    dockerhub_id = (
        f"{dockerhub_username}/{dockerhub_repo}"
        if dockerhub_username and dockerhub_repo
        else ""
    )

    dataset = load_gso_dataset(dataset_name, split, instance_ids)

    # Build images for instances
    successful, failed = build_instance_images(
        dataset=dataset,
        max_workers=max_workers,
        force_rebuild=force_rebuild,
        push_to_registry=push_to_registry,
        dockerhub_id=dockerhub_id,
    )
    print(f"Successfully built {len(successful)} images")
    print(f"Failed to build {len(failed)} images")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gso-bench/gso",
        help="Name of HF dataset to use or local json/jsonl file",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max workers for parallel processing"
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild all images",
    )
    parser.add_argument(
        "--push_to_registry",
        type=str2bool,
        default=False,
        help="Push images to DockerHub registry",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=8192, help="Open file limit"
    )
    parser.add_argument(
        "--dockerhub_username",
        type=str,
        default="",
        help="DockerHub username",
    )
    parser.add_argument(
        "--dockerhub_repo",
        type=str,
        default="",
        help="DockerHub repository name",
    )
    args = parser.parse_args()
    main(**vars(args))
