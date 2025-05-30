from r2e.multiprocess import run_tasks_in_parallel_iter

from gso.data import Repo, Problem, Tests, PerformanceCommit
from gso.collect.generate.prompt import *
from gso.collect.generate.helpers import *
from gso.logger import logger

MAX_COMMIT_TOKENS = 50000
MAX_PR_TOKENS = 20000


def prepare_mp_helper(args) -> Tests:
    """Helper function to prepare test objects for a commit."""
    repo, prob, commit, is_oversample = args

    if count_tokens(commit.diff_text) > MAX_COMMIT_TOKENS:
        diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
    else:
        diff_text = commit.diff_text

    context_msg = CONTEXT_MSG.format(
        api=prob.api,
        repo_name=repo.repo_name,
        commit_message=strip_empty_lines(commit.message),
        commit_diff=diff_text,
    )

    if commit.linked_pr is not None:
        pr_messages = get_github_convo(repo, commit.linked_pr)
        if count_tokens(pr_messages) > MAX_PR_TOKENS:
            pr_messages = pr_messages[:MAX_PR_TOKENS] + "...(truncated)..."

        context_msg += PR_INFO.format(pr_messages=pr_messages)
    else:
        context_msg += "No associated pull request for this commit.\n"

    task_msg = (
        f"Write a test for the {prob.api} API in the {repo.repo_name} repository. "
        "Remember to NOT time the setup code."
    )

    if repo.repo_instr:
        task_msg += f"\n\nRepo-specific Instructions:\n{repo.repo_instr}\n"

    commit_tests = Tests.from_commit(commit)
    commit_tests.init_chat(SYSTEM_MSG, context_msg, task_msg)

    if is_oversample:
        prev_good_tests = get_latest_good_tests(prob, commit.quick_hash())
        commit_tests.add_samples(prev_good_tests)  # copy previous good tests
        commit_tests.add_message(
            "user", OVERSAMPLE_MSG.format(prev_test=prev_good_tests[0]), idx=2
        )

    return commit_tests


def prepare(args) -> Problem:
    """Prepare the context and add it to a problem."""
    repo: Repo = args[0]
    prob: Problem = args[1]

    tasks = [(repo, prob, commit, False) for commit in prob.commits]
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


def prepare_oversample(args) -> Problem:
    """Prepare the context for oversampling and add it to a problem."""
    repo: Repo = args[0]
    prob: Problem = args[1]
    commit_hashes: list[str] = args[2]

    tasks = [
        (repo, prob, commit, True)
        for commit in prob.commits
        if commit.quick_hash() in commit_hashes
    ]
    commits_in_task = [c for _, _, c, _ in tasks]

    prepare_iter = run_tasks_in_parallel_iter(
        prepare_mp_helper, tasks, num_workers=8, use_progress_bar=False, use_spawn=False
    )

    all_commit_tests = []
    for task, task_result in zip(tasks, prepare_iter):
        if task_result.is_success():
            all_commit_tests.append(task_result.result)
        else:
            logger.error(f"Failed to prepare: {task_result.exception_tb}")

    prob.set_tests(all_commit_tests)
    return prob
