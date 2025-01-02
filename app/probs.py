from flask import Flask, render_template, request, jsonify
import os
import json
from math import ceil
from collections import defaultdict

from pyperf.constants import EXPS_DIR
from pyperf.utils.io import load_problems
from pyperf.execute.evaluate import speedup_summary

app = Flask(__name__)

APIS_PER_PAGE = 5  # Number of APIs to show per page

def get_repo_list():
    return [d for d in os.listdir(EXPS_DIR) if os.path.isdir(EXPS_DIR / d)]

def load_repo_data(repo_name, page=1, per_page=APIS_PER_PAGE):
    file_path = os.path.join(EXPS_DIR, f"{repo_name}", f"{repo_name}_results.json")
    all_problems = load_problems(file_path)
    api_groups = defaultdict(list)
    
    for prob in all_problems:
        if prob.is_valid():
            stats,_,_ = speedup_summary(prob)
            if stats:
                for s in stats:
                    test = prob.get_test(stats[s]["commit"], stats[s]["test_id"])
                    commit = next(
                        (c for c in prob.commits if c.quick_hash() == stats[s]["commit"]),
                        None,
                    )
                    if commit:
                        # lambda to get file type from path
                        get_file_type = lambda x: x.split(".")[-1]
                        
                        result = stats[s].copy()
                        result["test"] = test
                        result["date"] = commit.date.isoformat()
                        result["repo_url"] = prob.repo.repo_url
                        result["stats"] = commit.stats
                        result["stats"]["ftypes"] = list(set(map(get_file_type, commit.files_changed)))
                        api_groups[result["api"]].append(result)

    # Sort problems within each API group
    sort_by = request.args.get('sort', 'date')
    for api in api_groups:
        if sort_by == 'speedup':
            api_groups[api].sort(key=lambda x: x['speedup_factor'], reverse=True)
        else:
            api_groups[api].sort(key=lambda x: x['date'], reverse=True)

    # Get sorted list of API names
    api_names = sorted(api_groups.keys())
    total_apis = len(api_names)
    total_pages = ceil(total_apis / per_page)
    
    # Slice the APIs for the current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    current_page_apis = api_names[start_idx:end_idx]
    
    # Create the response data structure
    paginated_data = {
        'apis': {
            api: api_groups[api] 
            for api in current_page_apis
        },
        'total_pages': total_pages,
        'current_page': page,
        'total_apis': total_apis,
        'has_next': page < total_pages,
        'has_prev': page > 1
    }
    
    return paginated_data

@app.route("/")
def home():
    repos = get_repo_list()
    default_repo = repos[0] if repos else None
    return render_template("base.html", repos=repos, default_repo=default_repo)

@app.route("/get_repo_data/<repo_name>")
def get_repo_data(repo_name):
    page = int(request.args.get('page', 1))
    data = load_repo_data(repo_name, page=page)
    if data:
        return jsonify(data)
    return jsonify({"error": "Repo not found"}), 404

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5600
    app.run(debug=False, host=host, port=port)