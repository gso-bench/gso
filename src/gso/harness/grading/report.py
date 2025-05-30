import docker
import json
from pathlib import Path

from gso.constants import RUN_EVALUATION_LOG_DIR, EVALUATION_REPORTS_DIR
from gso.harness.environment.docker_utils import list_images


def make_run_report(
    predictions: dict,
    full_dataset: list,
    run_id: str,
    client: docker.DockerClient | None = None,
) -> Path:
    """
    Generate a comprehensive evaluation report for a GSO run.

    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        run_id (str): Run ID
        client (docker.DockerClient, optional): Docker client for container/image status

    Returns:
        Path: Path to the generated report file
    """
    # Track different categories of instances
    incomplete_ids = set()  # Instances without predictions
    empty_patch_ids = set()  # Instances with predictions but with empty patches
    error_ids = set()  # Instances that could not be evaluated due to errors
    completed_ids = set()  # Instances that were run evaluated
    passed_ids = set()  # Instances where tests passed
    base_failed_ids = set()  # Instances where base failed to run
    patch_failed_ids = set()  # Instances where patch failed to apply
    test_failed_ids = set()  # Instances where tests failed to pass
    opt_base = set()  # Instances that improved over base
    opt_commit = set()  # Instances that improved over commit
    opt_main = set()  # Instances that improved over main
    opt_stats = {}

    # Process each instance
    for instance in full_dataset:
        instance_id = instance.instance_id
        if instance_id not in predictions:
            incomplete_ids.add(instance_id)
            continue

        # Check for empty patches
        prediction = predictions[instance_id]
        if prediction.get("model_patch") in [None, ""]:
            empty_patch_ids.add(instance_id)
            continue

        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction.get("model_name_or_path", "None").replace("/", "__")
            / instance_id
            / "report.json"
        )

        # instance was not run successfully
        if not report_file.exists():
            error_ids.add(instance_id)
            continue

        completed_ids.add(instance_id)

        # Load and process individual report
        try:
            with open(report_file) as f:
                report = json.load(f)
                instance_report = report[instance_id]

            # Check if patch was None or didn't exist
            if instance_report.get("patch_is_None", False) or not instance_report.get(
                "patch_exists", True
            ):
                empty_patch_ids.add(instance_id)
                continue

            if not instance_report.get("base_successfully_run"):
                base_failed_ids.add(instance_id)
                continue

            # Check patch application success
            if not instance_report["patch_successfully_applied"]:
                patch_failed_ids.add(instance_id)
                continue

            # Check test success
            if instance_report["test_passed"]:
                passed_ids.add(instance_id)
                opt_stats[instance_id] = (
                    instance_report["opt_stats"] | instance_report["time_stats"]
                )
            else:
                test_failed_ids.add(instance_id)

            # Track performance improvements
            if instance_report["opt_base"]:
                opt_base.add(instance_id)
                # opt_stats[instance_id] = (
                #     instance_report["opt_stats"] | instance_report["time_stats"]
                # )
            if instance_report["opt_commit"]:
                opt_commit.add(instance_id)
            if instance_report["opt_main"]:
                opt_main.add(instance_id)

        except Exception as e:
            error_ids.add(instance_id)
            print(f"Error processing report for {instance_id}: {e}")

    # Generate final report
    report = {
        "summary": {
            "total_instances": len(full_dataset),
            "total_predictions": len(predictions),
            "completed_instances": len(completed_ids),
            "incomplete_instances": len(incomplete_ids),
            "passed_instances": len(passed_ids),
            "base_failed_instances": len(base_failed_ids),
            "patch_failed_instances": len(patch_failed_ids),
            "test_failed_instances": len(test_failed_ids),
            "empty_patch_instances": len(empty_patch_ids),
            "error_instances": len(error_ids),
            "opt_base": len(opt_base),
            "opt_commit": len(opt_commit),
            "opt_main": len(opt_main),
        },
        "instance_sets": {
            "completed_ids": sorted(completed_ids),
            "passed_ids": sorted(passed_ids),
            "base_failed_ids": sorted(base_failed_ids),
            "patch_failed_ids": sorted(patch_failed_ids),
            "test_failed_ids": sorted(test_failed_ids),
            "incomplete_ids": sorted(incomplete_ids),
            "empty_patch_ids": sorted(empty_patch_ids),
            "error_ids": sorted(error_ids),
            "opt_base_ids": sorted(opt_base),
            "opt_commit_ids": sorted(opt_commit),
            "opt_main_ids": sorted(opt_main),
        },
        "opt_stats": opt_stats,
        "schema_version": 1,
    }

    # Add Docker-related info if client provided
    if client:
        containers = client.containers.list(all=True)
        images = list_images(client)

        docker_info = {
            "active_containers": [
                container.name for container in containers if run_id in container.name
            ],
            "instance_images": sorted(images),
        }
        report["docker_status"] = docker_info

    # Write report to file
    model_name = (
        list(predictions.values())[0]
        .get("model_name_or_path", "None")
        .replace("/", "__")
    )
    EVALUATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = EVALUATION_REPORTS_DIR / Path(f"{model_name}.{run_id}.report.json")

    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    # Print summary
    summary = report["summary"]
    opt_base = summary["opt_base"] / summary["total_instances"]
    opt_commit = summary["opt_commit"] / summary["total_instances"]
    opt_main = summary["opt_main"] / summary["total_instances"]
    print("\n=== Evaluation Summary ===")
    print(f"Total instances: {summary['total_instances']}")
    print(f"Instances submitted: {summary['total_predictions']}")
    print(f"Instances completed: {summary['completed_instances']}")
    print(f"Incomplete incomplete: {summary['incomplete_instances']}")
    print("-" * 10)
    print(f"Instances that passed: {summary['passed_instances']}")
    print(f"Instances with failed tests: {summary['test_failed_instances']}")
    print(f"Instances with failed patch: {summary['patch_failed_instances']}")
    print(f"Instances with empty patches: {summary['empty_patch_instances']}")
    print(f"Instances with errors: {summary['error_instances']}")
    print("-" * 10)
    print(f"opt(base): {summary['opt_base']} ({opt_base*100:.2f}%) ")
    print(f"opt(commit): {summary['opt_commit']} ({opt_commit*100:.2f}%)")
    print(f"opt(main): {summary['opt_main']} ({opt_main*100:.2f}%) ")

    print(f"\nReport written to: {report_file}")
    return report_file
