import os
import ast
import re
import argparse

from constants import HOME_DIR, REPO_DIR


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_parents = {}
        self.current_function = None
        self.current_class = None

    def visit_ClassDef(self, node):
        previous_class = self.current_class
        self.current_class = node.name
        for item in node.body:
            self.visit(item)
        self.current_class = previous_class

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def _process_function(self, node):
        function_name = node.name
        if self.current_class:
            parent = self.current_class
        else:
            parent = self.current_function

        if function_name in self.function_parents:
            self.function_parents[function_name].append(parent)
        else:
            self.function_parents[function_name] = [parent]

        previous_function = self.current_function
        self.current_function = function_name
        for item in node.body:
            self.visit(item)
        self.current_function = previous_function

    def get_function_parents(self):
        for key, value in self.function_parents.items():
            self.function_parents[key] = tuple(value)

        return self.function_parents


def parse_python_file_for_function_hierarchy(file_path):
    with open(file_path, "r") as source_file:
        source_code = source_file.read()
    tree = ast.parse(source_code)
    visitor = FunctionVisitor()
    visitor.visit(tree)
    return visitor.get_function_parents()


def split_diff_into_files(diff_text):
    """Split the diff text into sections per file."""
    file_sections = []
    current_section = []
    lines = diff_text.split("\n")
    for line in lines:
        if line.startswith("diff --git"):
            if current_section:
                file_sections.append("\n".join(current_section))
                current_section = []
        current_section.append(line)
    if current_section:
        file_sections.append("\n".join(current_section))
    return file_sections


def get_modified_functions(repo_name, diff_file_path):
    """Identify modified functions in the PR diff and their top-level functions."""
    modified_functions = []

    with open(diff_file_path, "r") as diff_file:
        diff_text = diff_file.read()

    func_def_regex = re.compile(
        r"^(?:@@.*@@)?\s*(async\s+def|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    )
    file_change_regex = re.compile(r"^diff --git a/(.+) b/\1")
    files_in_diff = split_diff_into_files(diff_text)

    for file in files_in_diff:
        lines = file.split("\n")
        file_change_line = lines[0]
        file_name = file_change_regex.match(file_change_line).group(1)
        file_path = os.path.join(REPO_DIR, repo_name, file_name)
        function_parents_map = parse_python_file_for_function_hierarchy(file_path)
        latest_function = None

        if file_name.endswith("test.py"):
            continue

        for line in lines[1:]:
            is_function_def = func_def_regex.match(line)
            if is_function_def:
                latest_function = is_function_def.group(2)
                continue

            if line.startswith("+") or line.startswith("-"):
                if latest_function and file_name:
                    parents = function_parents_map.get(latest_function, latest_function)
                    modified_functions.append((file_name, latest_function, parents))
                    latest_function = None

    return set(modified_functions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--repo_name",
        type=str,
        help="Short name of the repository (e.g., jax)",
        default="jax",
    )
    parser.add_argument(
        "--pr_number",
        type=int,
        help="Pull request number",
        default=22114,
    )
    args = parser.parse_args()

    diff_file_path = f"./logs/google___jax_{args.pr_number}/diff.patch"

    modified_functions = get_modified_functions(args.repo_name, diff_file_path)
    for file, function, parents in modified_functions:
        print(f"File: {file}, Function: {function}, Parents: {parents}")
