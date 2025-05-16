import re
import numpy as np
from enum import Enum

from gso.constants import MIN_PROB_SPEEDUP, BEAT_GM_THRESH, BEAT_GSD_THRESH
from gso.data.dataset import PyPerfInstance
from gso.harness.grading.evalscript import (
    START_EVAL_PATCH,
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
    TEST_DELIMITER,
    MAIN_REGRESS_WARNING,
)


class TestStatus(Enum):
    BASE_FAILED = "BASE_FAILED"
    PATCH_FAILED = "PATCH_FAILED"
    TESTS_ERRORED = "TESTS_ERRORED"
    TESTS_PASSED = "TESTS_PASSED"


def beat(gm_speedup, gsd_speedup, perc_slowdowns):
    """
    Check if the patch beats the reference based on the speedup and generality.

    Args:
        gm_speedup (float): geometric mean of the speedups
        gsd_speedup (float): geometric standard deviation of the speedups
        perc_slowdowns (float): percentage of tests that regressed
    Returns:
        bool: True if the patch `beats` the reference, False otherwise
    """
    has_slowdowns = perc_slowdowns > 0.0
    if has_slowdowns:
        return gm_speedup > BEAT_GM_THRESH and gsd_speedup < BEAT_GSD_THRESH
    else:
        return gm_speedup > BEAT_GM_THRESH


def geometric_stats(nums):
    if not nums:
        return None, None
    logs = np.log(nums)
    gm = np.exp(np.mean(logs))
    gsd = np.exp(np.std(logs, ddof=1))
    return gm, gsd


def speedup(before_mean, after_mean, before_test_means, after_test_means):
    if before_mean is None or after_mean is None:
        return None, None, None, None, None

    mean_opt_perc = ((before_mean - after_mean) / before_mean) * 100
    mean_speedup = before_mean / after_mean
    ratios = [b / a for b, a in zip(before_test_means, after_test_means)]
    slowdowns = [r for r in ratios if r < 1.0]
    gm_speedup, gsd_speedup = geometric_stats(ratios) if ratios else (None, None)
    perc_slowdowns = len(slowdowns) / len(ratios) * 100 if ratios else None

    return mean_opt_perc, mean_speedup, gm_speedup, gsd_speedup, perc_slowdowns


def compute_time_stats(test_groups):
    if not test_groups:
        return None, None, []

    per_test_means = [np.mean(t) for t in test_groups if t]
    if not per_test_means:
        return None, None, []
    overall_mean = np.mean(per_test_means)
    overall_std = np.std(per_test_means, ddof=1) if len(per_test_means) > 1 else 0.0
    return overall_mean, overall_std, per_test_means


def parse_times(content, start_marker, end_marker):
    segment = content.split(start_marker)[1].split(end_marker)[0]
    delim_base = TEST_DELIMITER.split("{")[0]
    test_blocks = re.split(
        r"^" + re.escape(delim_base) + r"\d+\s*$", segment, flags=re.MULTILINE
    )

    tests = []
    pattern = r"Execution time:\s+([\d\.]+)s"
    for block in test_blocks:
        block = block.strip()
        if not block:
            continue
        times = []
        for line in block.splitlines():
            m = re.match(pattern, line.strip())
            if m:
                times.append(float(m.group(1)))
        if times:
            tests.append(times)
    return tests


def parse_logs(content):
    tm = {
        "base_times": None,
        "patch_times": None,
        "commit_times": None,
        "main_times": None,
    }

    tm["base_times"] = parse_times(content, START_BASE_OUTPUT, END_BASE_OUTPUT)
    tm["patch_times"] = parse_times(content, START_PATCH_OUTPUT, END_PATCH_OUTPUT)
    tm["commit_times"] = parse_times(content, START_COMMIT_OUTPUT, END_COMMIT_OUTPUT)

    if not MAIN_REGRESS_WARNING in content:
        try:
            tm["main_times"] = parse_times(content, START_MAIN_OUTPUT, END_MAIN_OUTPUT)
        except:
            pass

    return tm


def get_opt_status(time_map) -> dict:
    """
    Compute the optimization status of the patch based on the execution times of the tests.

    Args:
        time_map (dict): a dictionary containing the execution times of the tests
    Returns:
        opt_status (dict): a dictionary containing the optimization status of the patch
    """
    opt_status = {
        "beat_base": False,
        "beat_commit": False,
        "beat_main": False,
        "time_stats": {},
        "opt_stats": {},
    }

    base_mean, base_std, base_means = compute_time_stats(time_map["base_times"])
    patch_mean, patch_std, patch_means = compute_time_stats(time_map["patch_times"])
    commit_mean, commit_std, commit_means = compute_time_stats(time_map["commit_times"])
    main_mean, main_std, main_means = compute_time_stats(time_map["main_times"])

    time_stats = {
        "base_mean": base_mean,
        "base_std": base_std,
        "patch_mean": patch_mean,
        "patch_std": patch_std,
        "commit_mean": commit_mean,
        "commit_std": commit_std,
        "main_mean": main_mean,
        "main_std": main_std,
        "per_test_means": {
            "base": base_means,
            "patch": patch_means,
            "commit": commit_means,
            "main": main_means,
        },
    }

    opt_stats = {
        "opt_perc_patch_base": None,
        "opt_perc_patch_commit": None,
        "opt_perc_patch_main": None,
        "speedup_patch_base": None,
        "speedup_patch_commit": None,
        "speedup_patch_main": None,
        "gm_speedup_patch_base": None,
        "gm_speedup_patch_commit": None,
        "gm_speedup_patch_main": None,
    }

    pb_opt, pb_speedup, pb_speedup_gm, pb_speedup_gsd, pb_slowdowns = speedup(
        base_mean, patch_mean, base_means, patch_means
    )
    pc_opt, pc_speedup, pc_speedup_gm, pc_speedup_gsd, pc_slowdowns = speedup(
        commit_mean, patch_mean, commit_means, patch_means
    )
    pm_opt, pm_speedup, pm_speedup_gm, pm_speedup_gsd, pm_slowdowns = speedup(
        main_mean, patch_mean, main_means, patch_means
    )

    if base_mean > patch_mean and pb_speedup_gm >= MIN_PROB_SPEEDUP:
        opt_status["beat_base"] = True
        opt_stats.update(
            {
                "opt_perc_patch_base": pb_opt,
                "speedup_patch_base": pb_speedup,
                "gm_speedup_patch_base": pb_speedup_gm,
                "gsd_speedup_patch_base": pb_speedup_gsd,
                "slowdown_perc_patch_base": pb_slowdowns,
            }
        )

        if beat(pc_speedup_gm, pc_speedup_gsd, pc_slowdowns):
            opt_status["beat_commit"] = True
            opt_stats.update(
                {
                    "opt_perc_patch_commit": pc_opt,
                    "speedup_patch_commit": pc_speedup,
                    "gm_speedup_patch_commit": pc_speedup_gm,
                    "gsd_speedup_patch_commit": pc_speedup_gsd,
                    "slowdown_perc_patch_commit": pc_slowdowns,
                }
            )

        if main_mean is None:
            pass

        elif beat(pm_speedup_gm, pm_speedup_gsd, pm_slowdowns):
            opt_status["beat_main"] = True
            opt_stats.update(
                {
                    "opt_perc_patch_main": pm_opt,
                    "speedup_patch_main": pm_speedup,
                    "gm_speedup_patch_main": pm_speedup_gm,
                    "gsd_speedup_patch_main": pm_speedup_gsd,
                    "slowdown_perc_patch_main": pm_slowdowns,
                }
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
        if START_EVAL_PATCH not in content:
            return {}, TestStatus.BASE_FAILED
        elif APPLY_PATCH_FAIL in content or (INSTALL_FAIL in content):
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
        "base_successfully_run": False,
        "patch_successfully_applied": False,
        "test_passed": False,
        "base_times": None,
        "patch_times": None,
        "commit_times": None,
        "main_times": None,
        "beat_base": False,
        "beat_commit": False,
        "beat_main": False,
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

    if test_status == TestStatus.BASE_FAILED:
        return report_map
    report_map[instance_id]["base_successfully_run"] = True

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
