import re
import numpy as np
from enum import Enum

from pyperf.constants import OPTIM_THRESH
from pyperf.data.dataset import PyPerfInstance
from pyperf.harness.grading.evalscript import (
    APPLY_PATCH_FAIL,
    INSTALL_FAIL,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    START_BASE_OUTPUT,
    END_BASE_OUTPUT,
    START_PATCH_OUTPUT,
    END_PATCH_OUTPUT,
    START_COMMIT_OUTPUT,
    END_COMMIT_OUTPUT,
    START_MAIN_OUTPUT,
    END_MAIN_OUTPUT,
)


class TestStatus(Enum):
    PATCH_FAILED = "PATCH_FAILED"
    TESTS_ERRORED = "TESTS_ERRORED"
    TESTS_PASSED = "TESTS_PASSED"


def speedup(before_mean, after_mean) -> float:
    opt_perc = ((before_mean - after_mean) / before_mean) * 100
    speedup_factor = before_mean / after_mean
    return opt_perc, speedup_factor


def compute_stats(times):
    mean = np.mean(times)
    std_dev = np.std(times, ddof=1)
    if np.isnan(mean):
        return None, None
    return mean, std_dev


def parse_times(content, start_marker, end_marker):
    time_str = content.split(start_marker)[1].split(end_marker)[0]

    pattern = r"Execution time:\s+([\d\.]+)s"
    times = []
    for line in time_str.strip().split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            seconds = float(match.group(1))
            times.append(seconds)
    return times


def parse_logs(content):
    time_map = {
        "base_times": None,
        "patch_times": None,
        "commit_times": None,
        "main_times": None,
    }

    time_map["base_times"] = parse_times(content, START_BASE_OUTPUT, END_BASE_OUTPUT)
    time_map["patch_times"] = parse_times(content, START_PATCH_OUTPUT, END_PATCH_OUTPUT)
    time_map["commit_times"] = parse_times(
        content, START_COMMIT_OUTPUT, END_COMMIT_OUTPUT
    )
    time_map["main_times"] = parse_times(content, START_MAIN_OUTPUT, END_MAIN_OUTPUT)

    return time_map


def get_opt_status(time_map) -> dict:
    """
    Compute the optimization status of the patch based on the execution times of the tests.

    Args:
        time_map (dict): a dictionary containing the execution times of the tests
    Returns:
        opt_status (dict): a dictionary containing the optimization status of the patch
    """
    opt_status = {
        "improved_base": False,
        "improved_commit": False,
        "improved_main": False,
        "time_stats": {},
        "opt_stats": {},
    }

    time_stats = {
        k: None
        for k in [
            "base_mean",
            "base_std",
            "patch_mean",
            "patch_std",
            "commit_mean",
            "commit_std",
            "main_mean",
            "main_std",
        ]
    }

    opt_stats = {
        k: None
        for k in [
            "opt_perc_base",
            "opt_perc_commit",
            "opt_perc_main",
            "speedup_base",
            "speedup_commit",
            "speedup_main",
        ]
    }

    base_mean, base_std_dev = compute_stats(time_map["base_times"])
    patch_mean, patch_std_dev = compute_stats(time_map["patch_times"])
    commit_mean, commit_std_dev = compute_stats(time_map["commit_times"])
    main_mean, main_std_dev = compute_stats(time_map["main_times"])

    time_stats.update(
        {
            "base_mean": base_mean,
            "base_std": base_std_dev,
            "patch_mean": patch_mean,
            "patch_std": patch_std_dev,
            "commit_mean": commit_mean,
            "commit_std": commit_std_dev,
            "main_mean": main_mean,
            "main_std": main_std_dev,
        }
    )

    patch_base_opt, patch_base_speedup = speedup(base_mean, patch_mean)
    patch_com_opt, patch_com_speedup = speedup(commit_mean, patch_mean)
    patch_main_opt, patch_main_speedup = speedup(main_mean, patch_mean)

    if base_mean > patch_mean and patch_base_opt >= OPTIM_THRESH:
        opt_status.update({"improved_base": True})
        opt_stats.update(
            {"opt_perc_base": patch_base_opt, "speedup_base": patch_base_speedup}
        )

        # if patch improves over base, then check how it compares to commit/main

        if commit_mean > patch_mean and patch_com_opt >= OPTIM_THRESH:
            opt_status.update({"improved_commit": True})
            opt_stats.update(
                {"opt_perc_commit": patch_com_opt, "speedup_commit": patch_com_speedup}
            )

        if main_mean > patch_mean and patch_main_opt >= OPTIM_THRESH:
            opt_status.update({"improved_main": True})
            opt_stats.update(
                {"opt_perc_main": patch_main_opt, "speedup_main": patch_main_speedup}
            )

    return {**opt_status, "time_stats": time_stats, "opt_stats": opt_stats}


def get_logs_eval(instance: PyPerfInstance, log_fp: str) -> tuple[dict, TestStatus]:
    """
    Parse the evaluation logs to extract the results of the evaluation.

    Args:
        instance (PyPerfInstance): the instance being evaluated
        test_log_path (str): path to the evaluation log
    Returns:
        time_map (dict): a dictionary containing the execution times of the tests
        status (TestStatus): the status of the evaluation
    """

    with open(log_fp) as f:
        content = f.read()

        if APPLY_PATCH_FAIL in content or (INSTALL_FAIL in content):
            return {}, TestStatus.PATCH_FAILED
        elif (TESTS_ERROR in content) or (TESTS_TIMEOUT in content):
            return {}, TestStatus.TESTS_ERRORED
        elif not (START_PATCH_OUTPUT in content and END_PATCH_OUTPUT in content):
            return {}, TestStatus.TESTS_ERRORED

        # get the time map
        return parse_logs(content), TestStatus.TESTS_PASSED


def get_eval_report(
    instance: PyPerfInstance,
    prediction: dict[str, str],
    test_log_path: str,
):
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation log.

    Args:
        test_spec (dict): test spec containing keys "instance_id", "FAIL_TO_PASS", and "PASS_TO_PASS"
        prediction (dict): prediction containing keys "instance_id", "model_name_or_path", and "model_patch"
        log_path (str): path to evaluation log
    Returns:
        report (dict): report of metrics
    """
    report_map = {}

    instance_id = prediction["instance_id"]
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "test_passed": False,
        "base_times": None,
        "patch_times": None,
        "commit_times": None,
        "main_times": None,
        "improved_base": False,
        "improved_commit": False,
        "improved_main": False,
        "time_stats": {},
        "opt_stats": {},
    }

    # Check if the model patch exists
    if prediction["model_patch"] is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    # parse the evaluation logs
    time_map, test_status = get_logs_eval(instance, test_log_path)

    if test_status == TestStatus.PATCH_FAILED:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    if test_status == TestStatus.TESTS_ERRORED:
        return report_map
    report_map[instance_id]["test_passed"] = True

    # add the execution times from time_map
    report_map[instance_id].update(time_map)

    # compute the optimization status
    opt_status = get_opt_status(time_map)
    report_map[instance_id].update(opt_status)

    return report_map
