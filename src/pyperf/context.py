import ast

from r2e.models import Function, Method, Repo, File, Context
from r2e.generators.context.manager import ContextManager

from pyperf.constants import REPOS_DIR


def get_context_by_name(
    repo_id: str, file_path: str, function_name: str, context_type="sliced"
) -> Context:

    repo_org, repo_name = repo_id.split("___")
    repo_path = str(REPOS_DIR / repo_name)
    file_path = str(REPOS_DIR / repo_name / file_path)

    repo = Repo(
        repo_org=repo_org,
        repo_name=repo_name,
        repo_id=repo_id,
        local_repo_path=repo_path,
    )

    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    func_names = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]

    if function_name not in func_names:
        raise ValueError(f"Function {function_name} not found in {file_path}")

    func_obj = Function.from_name_file_repo(function_name, file_path, repo)

    return ContextManager.get_context(context_type, func_obj, max_context_size)
