from flask import Flask, render_template, request, jsonify
import os
import json
from math import ceil
from collections import defaultdict

from gso.constants import EXPS_DIR
from gso.utils.io import load_problems
from gso.collect.execute.evaluate import speedup_summary

app = Flask(__name__)

APIS_PER_PAGE = 5  # Number of APIs to show per page


def parse_range(range_str, type=float):
    if "+" in range_str:
        if type == int:
            return int(range_str[:-1]), 9999999
        return float(range_str[:-1]), float("inf")
    if "-" in range_str:
        if type == int:
            return map(int, range_str.split("-"))
        return map(float, range_str.split("-"))
    raise ValueError(f"Invalid format: {range_str}")


def get_repo_list():
    repo_dirs = [d for d in os.listdir(EXPS_DIR) if os.path.isdir(EXPS_DIR / d)]
    repo_list = []
    for repo in repo_dirs:
        file_path = os.path.join(EXPS_DIR, f"{repo}", f"{repo}_results.json")
        if os.path.exists(file_path):
            repo_list.append(repo)
    return repo_list


def load_repo_data(repo_name, page=1, per_page=APIS_PER_PAGE, search_query=None):
    file_path = os.path.join(EXPS_DIR, f"{repo_name}", f"{repo_name}_results.json")
    all_problems = load_problems(file_path)
    api_groups = defaultdict(list)

    # Get speedup mode from request args, default to "target"
    speedup_mode = request.args.get("speedup_mode", "commit")

    for prob in all_problems:
        if prob.is_valid():
            stats, _, _ = speedup_summary(prob, speedup_mode=speedup_mode)
            if stats:
                for s in stats:
                    test = prob.get_test(stats[s]["commit"], stats[s]["test_id"])
                    commit = next(
                        (
                            c
                            for c in prob.commits
                            if c.quick_hash() == stats[s]["commit"]
                        ),
                        None,
                    )
                    if commit:
                        # lambda to get file type from path
                        get_file_type = lambda x: x.split(".")[-1]

                        result = stats[s].copy()
                        result["full_commit_hash"] = commit.commit_hash
                        result["test"] = test
                        result["date"] = commit.date.isoformat()
                        result["repo_url"] = prob.repo.repo_url
                        result["stats"] = commit.stats
                        result["stats"]["ftypes"] = list(
                            set(map(get_file_type, commit.files_changed))
                        )
                        api_groups[result["api"]].append(result)

    # Apply search filter after generating api_groups but before other filters
    if search_query and search_query.strip():
        search_query = search_query.lower()
        api_groups = {
            api: problems
            for api, problems in api_groups.items()
            if search_query in api.lower()
        }

    # Apply filters
    filters = {
        "non_python_only": request.args.get("non_python_only") == "true",
        "file_count_range": request.args.get("file_count_range"),
        "loc_range": request.args.get("loc_range"),
        # "commit_count_range": request.args.get("commit_count_range"),
        "speedup_range": request.args.get("speedup_range"),
    }
    api_groups = filter_problems(api_groups, filters)

    # Compute mean speedup for each API
    api_mean_speedups = {}
    for api in api_groups:
        speedups = [prob["speedup_factor"] for prob in api_groups[api]]
        if speedups:
            mean_speedup = sum(speedups) / len(speedups)
            api_mean_speedups[api] = mean_speedup
        else:
            api_mean_speedups[api] = 0

    # Sort API names based on mean speedup
    api_names = sorted(
        api_groups.keys(), key=lambda api: api_mean_speedups[api], reverse=True
    )
    total_apis = len(api_names)
    total_pages = ceil(total_apis / per_page)

    # Slice the APIs for the current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    current_page_apis = api_names[start_idx:end_idx]

    # Create the response data structure
    paginated_data = {
        "apis": {api: api_groups[api] for api in current_page_apis},
        "total_pages": total_pages,
        "current_page": page,
        "total_apis": total_apis,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }

    return paginated_data


def filter_problems(api_groups, filters):
    filtered_groups = defaultdict(list)

    for api, problems in api_groups.items():
        filtered_problems = problems

        if filters.get("non_python_only"):
            ignore_list = ["py", "rst", "md", "txt", "yml", "toml", "gitignore"]
            filtered_problems = [
                p
                for p in filtered_problems
                if any(ftype not in ignore_list for ftype in p["stats"]["ftypes"])
            ]

        file_count_range = filters.get("file_count_range")
        if file_count_range:
            min_files, max_files = parse_range(file_count_range, type=int)
            filtered_problems = [
                p
                for p in filtered_problems
                if min_files <= p["stats"]["num_non_test_files"] <= max_files
            ]

        loc_range = filters.get("loc_range")
        if loc_range:
            min_loc, max_loc = parse_range(loc_range, type=int)
            filtered_problems = [
                p
                for p in filtered_problems
                if min_loc <= p["stats"]["num_edited_lines"] <= max_loc
            ]

        speedup_range = filters.get("speedup_range")
        if speedup_range:
            min_speedup, max_speedup = parse_range(speedup_range, type=float)
            filtered_problems = [
                p
                for p in filtered_problems
                if min_speedup <= p["speedup_factor"] <= max_speedup
            ]

        commit_count_range = filters.get("commit_count_range")
        if commit_count_range:
            num_unique_commits = len(set(p["commit"] for p in filtered_problems))
            min_commits, max_commits = parse_range(commit_count_range, type=int)
            filtered_problems = (
                []
                if num_unique_commits < min_commits or num_unique_commits > max_commits
                else filtered_problems
            )

        if filtered_problems:  # Only add groups that have problems after filtering
            filtered_groups[api] = filtered_problems

    return filtered_groups


@app.route("/")
def home():
    repos = get_repo_list()
    default_repo = repos[0] if repos else None
    return render_template("base.html", repos=repos, default_repo=default_repo)


@app.route("/get_repo_data/<repo_name>")
def get_repo_data(repo_name):
    page = int(request.args.get("page", 1))
    search_query = request.args.get("search", "")
    data = load_repo_data(repo_name, page=page, search_query=search_query)
    if data:
        return jsonify(data)
    return jsonify({"error": "Repo not found"}), 404


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5600
    app.run(debug=False, host=host, port=port)
