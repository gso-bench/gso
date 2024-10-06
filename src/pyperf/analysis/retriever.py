import os
from typing import List
from pathlib import Path

from pyperf.analysis.data.models import PerformanceCommit
from r2e.llms.llm_args import LLMArgs
from r2e.llms.completions import LLMCompletions
from pyperf.analysis.utils import *

MAX_COMMIT_TOKENS = 90000


class Retriever:
    def __init__(self, repo_path: Path, n_files: int = 5):
        self.repo_path = repo_path
        self.n_files = n_files

    def get_file_structure(self, commit_hash: str) -> str:
        def build_structure(path: Path, prefix="") -> List[str]:
            result = []
            entries = sorted(path.iterdir(), key=lambda e: e.name.lower())
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                if entry.is_dir():
                    if entry.name.startswith(".") or entry.name.startswith("doc"):
                        continue
                    if entry.name.lower() in ["test", "tests"]:
                        continue
                    result.append(
                        f"{prefix}{'└── ' if is_last else '├── '}{entry.name}/"
                    )
                    result.extend(
                        build_structure(entry, prefix + ("    " if is_last else "│   "))
                    )
                elif entry.suffix == ".py":
                    if entry.name.startswith("__"):
                        continue
                    result.append(
                        f"{prefix}{'└── ' if is_last else '├── '}{entry.name}"
                    )
            return result

        # Checkout the specific commit
        os.system(f"git -C {self.repo_path} checkout {commit_hash} --quiet")

        structure = build_structure(self.repo_path)

        # Return to the original branch
        os.system(f"git -C {self.repo_path} checkout - --quiet")

        return "\n".join(structure)

    def build_prompt(
        self, commit: PerformanceCommit, file_structure: str
    ) -> List[dict]:
        system_prompt = (
            f"You are an expert software engineer analyzing a performance-related commit "
            f"in a Python repository. Your task is to identify the most likely files "
            f"that contain high-level APIs (functions or methods) affected by this performance optimization."
        )

        if count_tokens(commit.diff_text) > MAX_COMMIT_TOKENS:
            diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
        else:
            diff_text = commit.diff_text

        user_prompt = (
            f"Commit message: {commit.message}\n\n"
            f"Commit diff:\n: {diff_text}\n\n"
            f"File structure of the repository at this commit:\n"
            f"{file_structure}\n\n"
            f"Please list the {self.n_files} most likely files that contain high-level APIs "
            f"affected by this performance optimization. Provide your answer as a numbered list "
            f"in a markdown code block, like this:\n"
            f"```\n"
            f"1. path/to/file1.py\n"
            f"2. path/to/file2.py\n"
            f"```\n"
            f"Think step-by-step about which files are most likely to contain the affected APIs "
            f"based on the commit and file structure."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def extract_file_paths(self, llm_response: str) -> List[str]:
        file_paths = []
        in_code_block = False
        for line in llm_response.split("\n"):
            if line.strip() == "```":
                in_code_block = not in_code_block
                continue
            if in_code_block:
                parts = line.split(".", 1)
                if len(parts) > 1:
                    file_path = parts[1].strip()
                    file_paths.append(file_path)
        return file_paths

    def retrieve_affected_files(
        self, commits: List[PerformanceCommit], llm_args: LLMArgs
    ) -> None:
        prompts = []
        for commit in commits:
            file_structure = self.get_file_structure(commit.commit_hash)
            prompts.append(self.build_prompt(commit, file_structure))

        responses = LLMCompletions.get_llm_completions(llm_args, prompts)

        for commit, response in zip(commits, responses):
            affected_files = self.extract_file_paths(response[0])
            commit.add_affected_paths(affected_files)
