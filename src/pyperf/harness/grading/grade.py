import json
import docker
import traceback
from pathlib import Path, PurePosixPath

from pyperf.data.dataset import PyPerfInstance
from pyperf.constants import RUN_EVALUATION_LOG_DIR, INSTANCE_IMAGE_BUILD_DIR
from pyperf.harness.utils import setup_logger, close_logger
from pyperf.harness.grading.metrics import get_eval_report
from pyperf.harness.environment.docker_build import create_container
from pyperf.harness.environment.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)


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


def grade_instance(
    instance: PyPerfInstance,
    pred: dict,
    rm_image: bool,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
):
    """
    Run a single instance with the given prediction.

    Args:
        instance (PyPerfInstance): PyPerfInstance instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    client = docker.from_env()

    # Set up logging directory
    instance_id = instance.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up report file
    report_path = log_dir / "report.json"

    if rewrite_reports:
        test_output_path = log_dir / "test_output.txt"
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            instance=instance,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report

    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / instance.instance_image_key.replace(
        ":", "__"
    )
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            # link the image build dir in the log dir
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except Exception as e:
            # some error, idk why
            print(f"Error linking image build dir: {str(e)}")
            pass

    # Set up logger
    log_file = log_dir / "run_instance.log"
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Start instance container (IMP: instance image should already be built)
        container = create_container(instance, client, run_id, logger)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred["model_patch"] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, PurePosixPath("/tmp/patch.diff"))

        # HACK: add a source ./venv/bin/activate to the eval.sh script
        # TODO: move this to eval.sh during build!
        hack_cmd = "sed -i '/echo \"Running performance test before patch...\"/i source .venv/bin/activate' /eval.sh"
        hack_output, _, _ = exec_run_with_timeout(container, hack_cmd, timeout)
        logger.info(f"Hack command output: {hack_output}")

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
            instance=instance,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
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
