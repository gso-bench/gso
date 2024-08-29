from r2e.models import Function, Method, Repo, File, Context
from r2e.generators.context.manager import ContextManager


def get_context_by_name(
    repo_id: str,
    repo_path: str,
    file_path: str,
    function_name: str,
    context_type="sliced",
) -> Context:

    repo_org, repo_name = repo_path.split("___")
    repo = Repo(
        repo_org=repo_org,
        repo_name=repo_name,
        repo_id=repo_id,
        local_repo_path=repo_path,
    )

    # read the file
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    func_names = [f.name for f in ]

    return ContextManager.get_context(context_type, func_meth, max_context_size)
