import json
import docker
import traceback
from pathlib import Path, PurePosixPath

from gso.data.dataset import GSOInstance
from gso.constants import RUN_EVALUATION_LOG_DIR, INSTANCE_IMAGE_BUILD_DIR
from gso.harness.utils import setup_logger, close_logger
from gso.harness.grading.metrics import get_eval_report
from gso.harness.grading.evalscript import get_eval_script
from gso.harness.environment.docker_build import create_container
from gso.harness.environment.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)
from gso.harness.environment.patches import apply_patches


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        log_msg = traceback.format_exc()
        self.logger.info(log_msg)
        return (
            f"{self.instance_id}: {super().__str__()}\n"
            f"Check ({self.log_file}) for more information."
        )


def setup_eval_dir(instance, model_name_or_path, run_id):
    log_dir = (
        RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance.instance_id
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    report_path = log_dir / "report.json"
    log_file = log_dir / "run_instance.log"

    # link image build dir in log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / instance.instance_image_key.replace(
        ":", "__"
    )
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except Exception as e:
            print(f"Error linking image build dir: {str(e)}")
            pass

    return log_dir, report_path, log_file


def grade_instance(
    instance: GSOInstance,
    pred: dict,
    rm_image: bool,
    run_id: str,
    timeout: int | None = None,
    reformat_reports: bool = False,
    retry_count: int = 0,
    max_retries: int = 5,
):
    """
    Run a single instance with the given prediction.

    Args:
        instance (GSOInstance): GSOInstance instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        reformat_reports (bool): True if eval run is just to reformat existing report
    """
    client = docker.from_env()

    # Set up logging directory
    instance_id = instance.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir, report_path, log_file = setup_eval_dir(
        instance, model_name_or_path, run_id
    )
    logger = setup_logger(instance_id, log_file)

    if reformat_reports:
        test_output_path = log_dir / "test_output.txt"
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            instance=instance, prediction=pred, test_log_path=test_output_path
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report, test_output_path

    # Run the instance
    container = None
    try:
        # Start instance container (IMP: instance image should already be built)
        container = create_container(instance, client, run_id, logger)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        if retry_count > 0:
            logger.info(f"Retrying ({retry_count}/{max_retries})")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred["model_patch"] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, copying to container..."
        )
        copy_to_container(container, patch_file, PurePosixPath("/tmp/patch.diff"))

        # copy latest eval script to container
        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(get_eval_script(instance))
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        # copy test scripts to container (with patches applied)
        patched_tests = apply_patches(instance.instance_id, instance.tests)
        for i, test_content in enumerate(patched_tests):
            test_file = Path(log_dir / f"gso_test_{i}.py")
            test_file.write_text(test_content)
            copy_to_container(container, test_file, PurePosixPath(f"/gso_test_{i}.py"))
        logger.info(
            f"Copied {len(patched_tests)} test scripts to container for {instance_id}"
        )

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        test_output_path = log_dir / "test_output.txt"
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            instance=instance, prediction=pred, test_log_path=test_output_path
        )
        logger.info(f"Report: {report}")

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report, test_output_path
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, instance.instance_image_key, logger)
        close_logger(logger)
    return
