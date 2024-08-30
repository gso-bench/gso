import ast

from r2e.models import Function, Method, Class, Repo, File, Context
from r2e.generators.context.manager import ContextManager

from pyperf.constants import REPOS_DIR


def is_func_meth(name, tree):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name == name:
                return True
    return False


def is_class(name, tree):
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            if n.name == name:
                return True
    return False


def get_context_by_name(
    repo_id: str, file_path: str, construct_name: str, context_type="sliced"
) -> Context:

    repo_org, repo_name = repo_id.split("___")
    repo_path = str(REPOS_DIR / repo_name)
    file_path = str(REPOS_DIR / repo_name / file_path)

    repo = Repo(
        repo_org=repo_org,
        repo_name=repo_name,
        repo_id=repo_name,
        local_repo_path=repo_path,
    )

    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    if is_func_meth(construct_name, tree):
        obj = Function.from_name_file_repo(construct_name, file_path, repo)
    elif is_class(construct_name, tree):
        obj = Class.from_name_file_repo(construct_name, file_path, repo)
    else:
        raise ValueError(f"{construct_name} not found in {file_path}")

    return ContextManager.get_context(context_type, obj, 6000)


if __name__ == "__main__":
    modified_functions = [
        (
            "jax/experimental/array_serialization/serialization.py",
            "release_bytes",
            ("_LimitInFlightBytes",),
        ),
        (
            "jax/experimental/array_serialization/serialization.py",
            "_write_array",
            ("async_serialize",),
        ),
    ]

    repo_id = "google___jax"
    construct_name = "async_serialize"

    for f in modified_functions:
        fp, fn, parents = f
        for p in parents:
            context = get_context_by_name(repo_id, fp, p)
            print(context.context)
