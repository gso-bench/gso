from r2e.multiprocess import run_tasks_in_parallel_iter

from pyperf.data import Repo, Problem, Tests
from pyperf.generate.prompt import *
from pyperf.generate.helpers import *
from pyperf.logger import logger


def prepare_mp_helper(args) -> Tests:
    """Helper function to prepare test objects for a commit."""
    repo, prob, commit = args
    context_msg = CONTEXT_MSG.format(
        api=prob.api,
        repo_name=repo.repo_name,
        commit_message=strip_empty_lines(commit.message),
        commit_diff=commit.diff_text,
    )

    if commit.linked_pr is not None:
        pr_messages = get_github_convo(repo, commit.linked_pr)
        context_msg += PR_INFO.format(pr_messages=pr_messages)
    else:
        context_msg += "No associated pull request for this commit.\n"

    task_msg = (
        f"Write a test for the {prob.api} API in the {repo.repo_name} repository. "
        "Remember to NOT time the setup code."
    )

    commit_tests = Tests.from_commit(commit)
    commit_tests.init_chat(SYSTEM_MSG, context_msg, task_msg)

    return commit_tests


def prepare(args) -> Problem:
    """Prepare the context and add it to a problem."""
    repo: Repo = args[0]
    prob: Problem = args[1]

    tasks = [(repo, prob, commit) for commit in prob.commits]
    prepare_iter = run_tasks_in_parallel_iter(
        prepare_mp_helper, tasks, num_workers=8, use_progress_bar=False, use_spawn=False
    )

    all_commit_tests = []
    for task_result in prepare_iter:
        if task_result.is_success():
            all_commit_tests.append(task_result.result)
        else:
            logger.error(f"Failed to prepare: {task_result.exception_tb}")

    prob.set_tests(all_commit_tests)
    return prob
