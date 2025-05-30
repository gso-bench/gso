import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from datetime import datetime

from gso.constants import ANALYSIS_COMMITS_DIR
from gso.collect.analysis.apis import APIAnalyzer

app = Flask(__name__)


def get_repo_list():
    return [
        f.replace("_commits.json", "")
        for f in os.listdir(ANALYSIS_COMMITS_DIR)
        if f.endswith("_commits.json")
    ]


def load_repo_data(repo_name):
    file_path = os.path.join(ANALYSIS_COMMITS_DIR, f"{repo_name}_commits.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None


def save_repo_data(repo_name, data):
    file_path = os.path.join(ANALYSIS_COMMITS_DIR, f"{repo_name}_commits.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def home():
    repos = get_repo_list()
    default_repo = repos[0] if repos else None
    return render_template("commits.html", repos=repos, default_repo=default_repo)


@app.route("/get_repo_data/<repo_name>")
def get_repo_data(repo_name):
    data = load_repo_data(repo_name)

    # drop the diff_text field to avoid sending too much data
    if data:
        for commit in data["performance_commits"]:
            commit.pop("diff_text", None)

    if data:
        return jsonify(data)
    return jsonify({"error": "Repo not found"}), 404


@app.route("/api_commit_map/<repo_name>")
def api_commit_map(repo_name):
    analyzer = APIAnalyzer()
    commit_analysis = analyzer.load_analysis(
        ANALYSIS_COMMITS_DIR / f"{repo_name}_commits.json"
    )
    repo_url = commit_analysis.repo_url
    analyzer.create_api_to_commits_map(commit_analysis)
    api_summary = analyzer.api_commit_map()
    sorted_apis = sorted(
        api_summary.items(), key=lambda item: len(item[1]), reverse=True
    )

    html = "<h1>API Summary</h1>"
    html += f"<p>Number of Commits: {len(commit_analysis.performance_commits)}</p>"
    html += f"<p>Number of APIs: {len(sorted_apis)}</p>"
    for api, commits in sorted_apis:
        html += f"<h2>API: {api}</h2>"
        html += f"<p>Number of affecting commits: {len(commits)}</p>"
        html += "<p>Affecting commits:</p>"
        for commit in commits:
            html += f"<p>{commit['date']} <a href='{repo_url}/commit/{commit['commit_hash']}'>{commit['commit_hash'][:8]}: {commit['subject']}</a></p>"
        html += "<br>"
    return html


@app.route("/add_comment", methods=["POST"])
def add_comment():
    repo_name = request.form["repo_name"]
    commit_hash = request.form["commit_hash"]
    comment = request.form["comment"]

    data = load_repo_data(repo_name)
    if data:
        for commit in data["performance_commits"]:
            if commit["commit_hash"] == commit_hash:
                if "comments" not in commit:
                    commit["comments"] = []
                commit["comments"].append(
                    {"text": comment, "timestamp": datetime.now().isoformat()}
                )
                save_repo_data(repo_name, data)
                return jsonify(success=True)

    return jsonify(success=False, message="Commit not found"), 404


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5700
    app.run(debug=False, host=host, port=port)
    # app.run(debug=True, port=port)
