import json
import os
import pandas as pd
from pathlib import Path

from flask import Flask, jsonify, render_template, url_for

from pyperf.constants import SUBMISSIONS_DIR, EVALUATION_REPORTS_DIR

app = Flask(__name__)


def get_available_logs():
    """Recursively find all .jsonl files in the logs directory"""
    log_files = []
    for root, dirs, files in os.walk(SUBMISSIONS_DIR):
        for file in files:
            # jsonl file with name output.jsonl
            if file.endswith(".jsonl") and file == "output.jsonl":
                # ignore if archive is in the file path
                if "archives" in root:
                    continue

                # ignore if plans is in the file path
                if "plans" in root:
                    continue

                # Get relative path from SUBMISSIONS_DIR
                rel_path = os.path.relpath(os.path.join(root, file), SUBMISSIONS_DIR)
                log_files.append(rel_path)
    return sorted(log_files)


# Global variables to store the conversations and current log
conversations = {}
current_indices = {}
instance_id_maps = {}  # Maps log_path -> {instance_id: index}
current_log = None


def load_jsonl(file_path):
    """Load a JSONL file and store its conversations"""
    # Make sure we're loading the actual jsonl file
    if not file_path.endswith("output.jsonl"):
        file_path = os.path.join(file_path, "output.jsonl")

    # open the analsis csv file into a df
    analysis_df = pd.read_csv(
        "~/pyperf/experiments/qualitative/trajectory_analysis.csv"
    )

    full_path = os.path.join(SUBMISSIONS_DIR, file_path)
    with open(full_path, "r") as f:
        conversations[file_path] = []
        instance_id_maps[file_path] = {}

        for idx, line in enumerate(f):
            try:
                conv = json.loads(line)
                run_id = Path(file_path).parent.name
                conv["run_id"] = run_id
                conv["analysis"] = ""
                instance_id = conv.get("instance_id")
                if instance_id:
                    analysis = analysis_df[
                        (analysis_df["run_id"] == run_id)
                        & (analysis_df["instance_id"] == conv["instance_id"])
                    ]
                    if not analysis.empty:
                        conv["analysis"] = analysis.iloc[0].get("analysis", "")

                    try:
                        # Look for report in the reports directory
                        possible_reports = list(
                            EVALUATION_REPORTS_DIR.glob(f"*{run_id}.test.report.json")
                        )

                        if possible_reports:
                            report_path = possible_reports[0]
                            with open(report_path, "r") as report_file:
                                report_data = json.load(report_file)

                                # Check if instance is in improved_commit_ids
                                improved_commit = instance_id in report_data.get(
                                    "instance_sets", {}
                                ).get("improved_commit_ids", [])
                                improved_main = instance_id in report_data.get(
                                    "instance_sets", {}
                                ).get("improved_main_ids", [])

                                # Get optimization stats if available
                                opt_stats = report_data.get("opt_stats", {}).get(
                                    instance_id, {}
                                )

                                # Add to the conversation data
                                if "test_result" not in conv:
                                    conv["test_result"] = {}

                                conv["test_result"]["improved_commit"] = improved_commit
                                conv["test_result"]["improved_main"] = improved_main
                                conv["test_result"]["opt_stats"] = opt_stats
                    except Exception as e:
                        print(f"Error loading report for {instance_id}: {e}")

                conversations[file_path].append(conv)

                # Map instance_id to index
                if instance_id:
                    instance_id_maps[file_path] = instance_id_maps.get(file_path, {})
                    instance_id_maps[file_path][instance_id] = idx
            except json.JSONDecodeError:
                continue

    current_indices[file_path] = 0
    return file_path


@app.route("/")
def index():
    log_files = get_available_logs()
    return render_template("index.html", logs=log_files)


@app.route("/matrix")
def instance_matrix():
    """Generate a matrix showing instance performance across different runs"""
    # Get all available log files
    log_files = get_available_logs()

    # Dictionary to hold the matrix data
    # Format: {instance_id: {run_id: {"success": bool, "log_url": url}}}
    matrix = {}

    # Lists to track all unique instances and runs
    all_instances = set()
    all_runs = set()

    # Process each log file
    for log_path in log_files:
        # Load log if not already loaded
        if log_path not in conversations:
            load_jsonl(log_path)

        # Extract run_id from path (folder name containing output.jsonl)
        run_id = Path(log_path).parent.name
        all_runs.add(run_id)

        # Process each conversation in this log file
        for idx, conv in enumerate(conversations[log_path]):
            instance_id = conv.get("instance_id")
            if not instance_id:
                continue

            # Add instance to our tracking set
            all_instances.add(instance_id)

            # Check if there's test result data
            test_result = conv.get("test_result", {})
            success = test_result.get("improved_commit", False)

            # Create URL for this specific conversation
            log_url = url_for(
                "view_by_instance_id",
                log_path=Path(log_path).parent,
                instance_id=instance_id,
            )

            # Initialize matrix entry for this instance if needed
            if instance_id not in matrix:
                matrix[instance_id] = {}

            # Add result for this run
            matrix[instance_id][run_id] = {"success": success, "log_url": log_url}

    # Sort runs and instances for consistent display
    sorted_runs = sorted(all_runs)
    sorted_instances = sorted(all_instances)

    return render_template(
        "matrix.html", matrix=matrix, instances=sorted_instances, runs=sorted_runs
    )


@app.route("/view/<path:log_path>")
def view_log(log_path):
    global current_log

    # Normalize path to always include output.jsonl
    if not log_path.endswith("output.jsonl"):
        log_path = os.path.join(log_path, "output.jsonl")

    current_log = log_path

    # Load the log file if it hasn't been loaded yet
    if log_path not in conversations:
        load_jsonl(log_path)

    return render_template("conversation.html", log_path=log_path)


@app.route("/view/<path:log_path>/<string:instance_id>")
def view_by_instance_id(log_path, instance_id):
    global current_log

    # Normalize path to ensure it includes output.jsonl
    if not log_path.endswith("output.jsonl"):
        file_path = os.path.join(log_path, "output.jsonl")
    else:
        file_path = log_path
        # Remove output.jsonl from path for consistent handling
        log_path = os.path.dirname(log_path)

    current_log = file_path

    # Load file if needed
    if file_path not in conversations:
        load_jsonl(file_path)

    # Try to find the instance ID in our mapping
    if file_path in instance_id_maps and instance_id in instance_id_maps[file_path]:
        current_indices[file_path] = instance_id_maps[file_path][instance_id]

    return render_template("conversation.html", log_path=file_path)


@app.route("/conversation/current/<path:log_path>")
def get_current_conversation(log_path):
    if log_path not in conversations:
        return jsonify({"error": "Log file not loaded"})

    current_conversation = conversations[log_path][current_indices[log_path]]
    instance_id = current_conversation.get("instance_id", "")

    return jsonify(
        {
            "conversation": current_conversation,
            "current_index": current_indices[log_path],
            "total": len(conversations[log_path]),
            "instance_id": instance_id,
        }
    )


@app.route("/conversation/next/<path:log_path>")
def next_conversation(log_path):
    if current_indices[log_path] < len(conversations[log_path]) - 1:
        current_indices[log_path] += 1
    return get_current_conversation(log_path)


@app.route("/conversation/previous/<path:log_path>")
def previous_conversation(log_path):
    if current_indices[log_path] > 0:
        current_indices[log_path] -= 1
    return get_current_conversation(log_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5760)
