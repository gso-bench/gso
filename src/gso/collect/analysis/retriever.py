import os
from pathlib import Path
from collections import deque


from gso.data import PerformanceCommit
from r2e.llms.llm_args import LLMArgs
from r2e.llms.completions import LLMCompletions
from gso.collect.analysis.utils import *

MAX_COMMIT_TOKENS = 30000
IGNORE_DIRECTORY_SIZE = 500
MAX_FILE_TOKENS = 30000


class Retriever:
    def __init__(self, repo_path: Path, n_files: int = 3):
        self.repo_path = repo_path
        self.n_files = n_files
        self.file_structure, self.file_content_map = self.collect_files(self.repo_path)
        self.indented_file_structure = self.indent_file_structure(self.file_structure)

    def collect_files(self, clone_dir: Path) -> tuple[dict, dict[str, str]]:
        def add_to_structure(structure, path_parts, is_file):
            current = structure
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            if is_file:
                current[path_parts[-1]] = None
            elif path_parts[-1] not in current:
                current[path_parts[-1]] = {}

        file_structure = {}
        file_content_map = {}
        dir_queue = deque([clone_dir])

        while dir_queue:
            current_dir = dir_queue.popleft()
            try:
                with os.scandir(current_dir) as entries:
                    entries = list(entries)
                    if len(entries) > IGNORE_DIRECTORY_SIZE:
                        continue
                    for entry in entries:
                        relative_path = os.path.relpath(entry.path, clone_dir)
                        path_parts = relative_path.split(os.sep)

                        if entry.is_file():
                            if entry.name.endswith(
                                (".py", ".pyi", ".c", ".cpp", ".rs", ".pyx", ".pxd")
                            ):
                                if entry.name.endswith("__init__.py"):
                                    continue

                                content = self.read_file(entry.path)
                                if (
                                    len(content) > 10
                                    and count_tokens(content) < MAX_FILE_TOKENS
                                ):
                                    add_to_structure(file_structure, path_parts, True)
                                    file_content_map[entry.path] = content

                        elif entry.is_dir():

                            if entry.name.startswith(".") or entry.name.startswith(
                                "doc"
                            ):
                                continue

                            if entry.name.lower() in ["tests", "test"]:
                                continue

                            add_to_structure(file_structure, path_parts, False)
                            dir_queue.append(entry.path)  # type: ignore

            except PermissionError:
                print(f"Permission denied: {current_dir}")

        return file_structure, file_content_map

    def indent_file_structure(self, structure, indent=""):
        result = ""
        for key, value in sorted(structure.items()):
            if value is None:
                result += f"{indent}{key}\n"
            else:
                result += f"{indent}{key}/\n"
                result += self.indent_file_structure(value, indent + "  ")
        return result

    def read_file(self, file_path: str) -> str:
        try:
            with open(file_path, "r") as file:
                content = file.read().strip()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="ISO-8859-1") as file:
                    content = file.read().strip()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="utf-16") as file:
                        content = file.read().strip()
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError: {file_path}")
                    content = ""

        return content

    def extract_match_file_names(self, model_response: str) -> list[str]:
        file_names = []
        lines = model_response.split("\n")
        markdown_indices = [i for i, line in enumerate(lines) if line.startswith("```")]
        if len(markdown_indices) < 2:
            return file_names

        start_index = markdown_indices[0]
        end_index = markdown_indices[1]
        for line in lines[start_index + 1 : end_index]:
            if line.strip():
                if "." in line:
                    line = ".".join(line.split(".")[1:])
                file_name = os.path.basename(line)
                file_names.append(file_name.strip())

        matched_files = []
        for file_name in file_names:
            for file_path in self.file_content_map.keys():
                if file_path.endswith(f"/{file_name}"):
                    if file_path not in matched_files:
                        matched_files.append(file_path)

        return matched_files

    def build_prompt(self, commit: PerformanceCommit) -> list[dict]:
        system_prompt = (
            f"You are an expert software engineer analyzing a performance-related commit "
            f"in a Python repository. Your task is to identify the most likely files "
            f"that contain high-level APIs (functions or methods) affected by this performance optimization. "
            f"By high/top-level, we mean APIs that are not internal helper functions. "
            f"E.g., pd.read_csv (pandas), requests.get (requests), model.generate (transformers), etc."
        )

        system_prompt += "\nNOTE: If the repo uses python bindings to C/C++/Rust (e.g., huggingface/tokenizers), make sure to identify ALL (1) Python files, (2) Interface files (e.g., .pyi), (3) C/C++/Rust files affected by the commit."

        if count_tokens(commit.diff_text) > MAX_COMMIT_TOKENS:
            diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
        else:
            diff_text = commit.diff_text

        user_prompt = (
            f"Commit message: {commit.message}\n\n"
            f"Commit diff:\n: {diff_text}\n\n"
            f"File structure of the repository at this commit:\n"
            f"{self.indented_file_structure}\n\n"
            f"Please list the top {self.n_files} most likely files that contain high-level APIs affected by this optimization. "
            f"Enclose the file names in a list in a markdown code block as shown below:\n"
            f"```\n"
            f"1. file1.py\n"
            f"2. file2.py\n"
            f"```\n"
            f"Think step-by-step about which files are most likely to contain the affected APIs "
            f"based on the commit and file structure."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def retrieve_affected_files(self, commits, llm_args) -> None:
        prompts = [self.build_prompt(commit) for commit in commits]
        responses = LLMCompletions.get_llm_completions(llm_args, prompts)

        for commit, response in zip(commits, responses):
            affected_files = self.extract_match_file_names(response[0])
            commit.add_affected_paths(affected_files)
