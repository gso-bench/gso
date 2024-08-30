import os
import argparse
import requests
from github import Github

from pyperf.templates import *
from pyperf.analyze_diff import get_modified_constructs
from pyperf.context import get_context_by_name


class PRManager:
    def __init__(self, repo_full_name, pr_number, mode="diff", function_name=None):
        self.g = Github()

        self.repo_org = repo_full_name.split("/")[0]
        self.repo_name = repo_full_name.split("/")[1]
        self.repo_id = self.repo_org + "___" + self.repo_name
        self.repo_full_name = repo_full_name

        self.repo = self.g.get_repo(repo_full_name)
        self.pr_number = pr_number
        self.pr = self.repo.get_pull(pr_number)

        self.logs_dir = f"./logs/{self.repo_id}_{pr_number}"
        os.makedirs(self.logs_dir, exist_ok=True)

        self.mode = mode
        self.function_name = function_name

    def get_pr_summary(self, mode="diff"):
        """Get the summary of a PR and save the diff to a file."""
        title = self.pr.title
        description = self.pr.body
        files_changed = [f.filename for f in self.pr.get_files()]
        code_diff = requests.get(self.pr.diff_url).text

        with open(os.path.join(self.logs_dir, "diff.patch"), "w") as diff_file:
            diff_file.write(code_diff)
            self.diff_file = os.path.abspath(diff_file.name)

        if self.mode == "diff":
            return PR_SUMMARY_DIFF.format(title, description, files_changed, code_diff)

        elif self.mode == "slice":
            constructs = get_modified_constructs(self.repo_name, self.diff_file)
            return PR_SUMMARY_SLICE.format(
                function_name=args.function_name,
                sliced_context=self._get_context(constructs),
            )

    def create_scripts(self):
        base_commit = self.pr.base.sha
        setup_script, apply_patch_script = self._fill_script_templates(base_commit)

        with open(os.path.join(self.logs_dir, "setup_repo.sh"), "w") as setup_file:
            setup_file.write(setup_script)

        with open(os.path.join(self.logs_dir, "apply_patch.sh"), "w") as patch_file:
            patch_file.write(apply_patch_script)

    def _fill_script_templates(self, base_commit):
        setup_script = SETUP_SCRIPT.format(
            repo_full_name=self.repo_full_name,
            repo_name=self.repo_name,
            base_commit=base_commit,
        )

        patch_script = APPLY_PATCH_SCRIPT.format(
            repo_name=self.repo_name, diff_file=self.diff_file
        )

        return setup_script, patch_script

    def _get_context(self, modified_constructs: list) -> str:
        context_str = ""
        for f in modified_constructs:
            fp, fn, parents = f
            for p in parents:
                context = get_context_by_name(self.repo_id, fp, p)
                context_str += context.context
        return context_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--repo_full_name",
        type=str,
        help="Full name of the repository (e.g., google/jax)",
        default="google/jax",
    )
    parser.add_argument(
        "--pr_number",
        type=int,
        help="Pull request number",
        default=22114,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode of operation (diff or slice)",
        default="diff",
        choices=["diff", "slice"],
    )
    parser.add_argument(
        "--function_name",
        type=str,
        help="Name of the function to slice",
        default=None,
    )

    args = parser.parse_args()

    if args.mode == "slice" and args.function_name is None:
        parser.error("--function_name is required in slice mode")

    pr_manager = PRManager(
        args.repo_full_name, args.pr_number, args.mode, args.function_name
    )
    pr_summary = pr_manager.get_pr_summary()
    pr_manager.create_scripts()

    print(pr_summary)
