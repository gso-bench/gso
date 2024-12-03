import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from datetime import datetime

from pyperf.constants import EXPS_DIR
from pyperf.utils.io import load_problems
from pyperf.execute.evaluate import speedup_summary

app = Flask(__name__)


def get_repo_list():
    return [
        d for d in os.listdir(EXPS_DIR) if os.path.isdir(EXPS_DIR / d)
    ]


def load_repo_data(repo_name):
    file_path = os.path.join(EXPS_DIR, f"{repo_name}", f"{repo_name}_results.json")
    all_problems = load_problems(file_path)
    opt_problems = []
    
    for prob in all_problems:
        # print("///", prob.pid, "///")
        if prob.is_valid():
            stats = speedup_summary(prob)
            if stats:
                for s in stats:
                    test = prob.get_test(stats[s]['commit'], stats[s]['test_id'])
                    if test:
                        stats[s]['test'] = test
                    opt_problems.append(stats[s])
    return opt_problems


def save_repo_data(repo_name, data):
    file_path = os.path.join(EXPS_DIR, f"{repo_name}", f"{repo_name}_results.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def home():
    repos = get_repo_list()
    default_repo = repos[0] if repos else None
    return render_template("base.html", repos=repos, default_repo=default_repo)


@app.route("/get_repo_data/<repo_name>")
def get_repo_data(repo_name):
    data = load_repo_data(repo_name)
    if data:
        return jsonify(data)
    return jsonify({"error": "Repo not found"}), 404

@app.route("/add_comment", methods=["POST"])
def add_comment():
    return jsonify(success=False, message="Not implemented"), 501
    # repo_name = request.form["repo_name"]
    # commit_hash = request.form["commit_hash"]
    # comment = request.form["comment"]

    # data = load_repo_data(repo_name)
    # if data:
    #     for commit in data["performance_commits"]:
    #         if commit["commit_hash"] == commit_hash:
    #             if "comments" not in commit:
    #                 commit["comments"] = []
    #             commit["comments"].append(
    #                 {"text": comment, "timestamp": datetime.now().isoformat()}
    #             )
    #             save_repo_data(repo_name, data)
    #             return jsonify(success=True)

    # return jsonify(success=False, message="Commit not found"), 404


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5600
    app.run(debug=False, host=host, port=port)
    # app.run(debug=True, port=port)
