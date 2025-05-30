import os
import re
import tiktoken
from ghapi.core import GhApi

from gso.data import Repo

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")
tokenizer = tiktoken.encoding_for_model("gpt-4")


def count_tokens(context: str):
    return len(tokenizer.encode(context, disallowed_special=()))


def get_github_convo(repo: Repo, pr_num: str, max_count=5) -> str:
    """Get the conversation for a pull request.

    Goal: capture any information that might be related to testing the PR's performance.

    Note: this is not PR reviews, but regular comments on the PR.
    PR reviews usually don't have interesting testing information and also
    contain older code edits that are not relevant to the state after PR merge.
    """

    def format_comments(comments, max_count=5, min_lines=2):
        """Formats the comments for a pull request."""
        formatted_comments = []
        for comment in comments:
            if comment.user.type != "User":
                continue

            if len(comment.body.split("\n")) < min_lines:
                continue

            body = re.sub("![.*](.*)", "", comment.body)
            body = re.sub("<img.*>", "", body).strip()
            formatted_comment = f"{comment.user.login}: {body}"
            formatted_comment = strip_empty_lines(formatted_comment)
            formatted_comments.append(f"\n{formatted_comment.strip()}")

        return "".join(formatted_comments[:max_count])

    repo_owner = repo.repo_owner
    repo_name = repo.repo_name

    # use ghapi to get the PR discussion messages
    api = GhApi(token=GHAPI_TOKEN)
    try:
        pr = api.pulls.get(repo_owner, repo_name, int(pr_num))
        comments = api.issues.list_comments(repo_owner, repo_name, int(pr_num))
        comments_str = format_comments(comments)
    except Exception as e:
        return ""

    resp = ""
    if pr and pr.body and pr.body != "":
        resp += f"Description: {pr.body.strip()}"
    if comments_str != "":
        resp += f"\n\nComments:\n{comments_str.strip()}"

    return resp


def strip_empty_lines(text: str):
    return "\n".join([line for line in text.splitlines() if line.strip()])


def extract_codeblock(output) -> str:
    outputlines = output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def get_generated_tests(outputs) -> list[list[str]]:
    results = []
    for output in outputs:
        code_blocks = []
        for sample in output:
            code = extract_codeblock(sample)
            code_blocks.append(code)
        results.append(code_blocks)
    return results


def get_latest_good_tests(prob, commit_hash) -> list[str]:
    latest_run_key, latest_results = list(prob.results.items())[-1]
    good_test_ids = [r["test_id"] for r in latest_results if r["commit"] == commit_hash]
    good_tests = [prob.get_test(commit_hash, tid) for tid in good_test_ids]
    return good_tests
