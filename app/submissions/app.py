import json
import os
import pandas as pd
from pathlib import Path

from flask import Flask, jsonify, render_template, url_for, redirect

from gso.constants import (
    SUBMISSIONS_DIR,
    EVALUATION_REPORTS_DIR,
    RUN_EVALUATION_LOG_DIR,
)

app = Flask(__name__)

# Configuration
EXP_TYPE = "pass"
EXCLUDED_MODELS = [
    "claude-3-5",
    "o4-mini",
    "o3-mini",
    "gpt-4o",
    "v0.25.0",
    "archives",
    "plans",
    "steps",
]

# Global variables
conversations = {}
current_indices = {}
instance_id_maps = {}
current_log = None


def get_available_logs():
    """Find all .jsonl files in the logs directory"""
    log_files = []
    for root, dirs, files in os.walk(SUBMISSIONS_DIR):
        for file in files:
            if file.endswith(".jsonl") and file == "output.jsonl":
                if EXP_TYPE not in root:
                    continue
                if any(excluded_model in root for excluded_model in EXCLUDED_MODELS):
                    continue
                rel_path = os.path.relpath(os.path.join(root, file), SUBMISSIONS_DIR)
                log_files.append(rel_path)
    return sorted(log_files)


def load_jsonl(file_path):
    """Load a JSONL file and store its conversations"""
    model = "claude" if "claude" in file_path else "o4-mini"

    if not file_path.endswith("output.jsonl"):
        file_path = os.path.join(file_path, "output.jsonl")

    # Try to load analysis CSV file, but make it optional
    analysis_df = None
    try:
        analysis_csv_path = f"/home/gcpuser/gso-internal/experiments/qualitative/analyses/trajectory_analysis_{model}.csv"
        if os.path.exists(analysis_csv_path):
            analysis_df = pd.read_csv(analysis_csv_path)
    except Exception as e:
        print(f"Warning: Could not load analysis CSV: {e}")

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
                    # Only try to get analysis if CSV was loaded successfully
                    if analysis_df is not None:
                        analysis = analysis_df[
                            (analysis_df["run_id"] == run_id)
                            & (analysis_df["instance_id"] == conv["instance_id"])
                        ]
                        if not analysis.empty:
                            conv["analysis"] = analysis.iloc[0].get("analysis", "")

                    test_output_path = os.path.join(
                        RUN_EVALUATION_LOG_DIR, EXP_TYPE, run_id, instance_id
                    )
                    test_output_file = os.path.join(test_output_path, "test_output.txt")
                    if os.path.exists(test_output_file):
                        with open(test_output_file, "r") as test_output:
                            test_output_data = test_output.read()
                            conv["test_output"] = test_output_data

                    try:
                        # Look for reports in the run's log directory
                        run_log_dir = RUN_EVALUATION_LOG_DIR / EXP_TYPE / run_id
                        possible_reports = list(
                            run_log_dir.glob(f"*.{EXP_TYPE}.report.json")
                        )

                        if possible_reports:
                            report_path = possible_reports[0]
                            with open(report_path, "r") as report_file:
                                report_data = json.load(report_file)

                                # Check if instance is in opt_commit_ids
                                opt_commit = instance_id in report_data.get(
                                    "instance_sets", {}
                                ).get("opt_commit_ids", [])
                                opt_main = instance_id in report_data.get(
                                    "instance_sets", {}
                                ).get("opt_main_ids", [])

                                # Get optimization stats if available
                                opt_stats = report_data.get("opt_stats", {}).get(
                                    instance_id, {}
                                )

                                # Add to the conversation data
                                if "test_result" not in conv:
                                    conv["test_result"] = {}

                                conv["test_result"]["opt_commit"] = opt_commit
                                conv["test_result"]["opt_main"] = opt_main
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
    print("Loading matrix from evaluation reports...")

    matrix = {}
    all_instances = set()
    all_runs = set()
    report_files = []
    for run_dir in RUN_EVALUATION_LOG_DIR.glob(f"{EXP_TYPE}/*"):
        if run_dir.is_dir():
            report_files.extend(list(run_dir.glob(f"*.{EXP_TYPE}.report.json")))

    print(f"Found {len(report_files)} report files")

    for report_path in report_files:
        try:
            with open(report_path, "r") as f:
                report_data = json.load(f)

            run_id = report_path.stem.replace(f".{EXP_TYPE}.report", "")

            # Skip excluded models
            if any(excluded_model in run_id for excluded_model in EXCLUDED_MODELS):
                continue

            all_runs.add(run_id)

            # Get completed and successful instances
            completed_ids = report_data.get("instance_sets", {}).get(
                "completed_ids", []
            )
            opt_commit_ids = set(
                report_data.get("instance_sets", {}).get("opt_commit_ids", [])
            )

            # Find corresponding log path for URL generation
            log_files = get_available_logs()
            log_path = next(
                (log_file for log_file in log_files if run_id in log_file), None
            )

            # Process each completed instance
            for instance_id in completed_ids:
                all_instances.add(instance_id)

                if instance_id not in matrix:
                    matrix[instance_id] = {}

                # Create URL
                log_url = None
                if log_path:
                    log_url = url_for(
                        "view_by_instance_id",
                        log_path=Path(log_path).parent,
                        instance_id=instance_id,
                    )

                # Determine success
                success = instance_id in opt_commit_ids

                matrix[instance_id][run_id] = {"success": success, "log_url": log_url}

        except Exception as e:
            print(f"Warning: Could not load report {report_path}: {e}")

    print(
        f"Matrix generated with {len(all_instances)} instances across {len(all_runs)} runs"
    )

    # Sort runs by success count (descending)
    def success_count(run_id):
        return sum(
            1
            for instance_id in matrix
            if run_id in matrix[instance_id] and matrix[instance_id][run_id]["success"]
        )

    sorted_runs = sorted(all_runs, key=lambda run_id: (-success_count(run_id), run_id))
    sorted_instances = sorted(all_instances)

    return render_template(
        "matrix.html", matrix=matrix, instances=sorted_instances, runs=sorted_runs
    )


@app.route("/view/<path:log_path>")
def view_log(log_path):
    global current_log

    if not log_path.endswith("output.jsonl"):
        log_path = os.path.join(log_path, "output.jsonl")

    current_log = log_path

    if log_path not in conversations:
        load_jsonl(log_path)

    return render_template("conversation.html", log_path=log_path)


@app.route("/view/<path:log_path>/<string:instance_id>")
def view_by_instance_id(log_path, instance_id):
    global current_log

    if not log_path.endswith("output.jsonl"):
        file_path = os.path.join(log_path, "output.jsonl")
    else:
        file_path = log_path
        log_path = os.path.dirname(log_path)

    current_log = file_path

    if file_path not in conversations:
        load_jsonl(file_path)

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


@app.route("/patches")
def patches_index():
    """Redirect to the first available instance"""
    try:
        patches_csv_path = "/home/gcpuser/gso-internal/experiments/qualitative/analyses/gt_vs_model_patches.csv"
        if os.path.exists(patches_csv_path):
            patches_df = pd.read_csv(patches_csv_path)
        else:
            return redirect(url_for("index"))
    except Exception as e:
        print(f"Warning: Could not load patches CSV: {e}")
        return redirect(url_for("index"))

    if patches_df.empty:
        return redirect(url_for("index"))

    first_instance = patches_df.iloc[0].to_dict()
    run_id = first_instance["run_id"]
    instance_id = first_instance["instance_id"]

    return redirect(url_for("view_patches", run_id=run_id, instance_id=instance_id))


@app.route("/patches/<string:run_id>/<string:instance_id>")
def view_patches(run_id, instance_id):
    """View the patches for a specific instance"""
    try:
        patches_csv_path = "/home/gcpuser/gso-internal/experiments/qualitative/analyses/gt_vs_model_patches.csv"
        if os.path.exists(patches_csv_path):
            patches_df = pd.read_csv(patches_csv_path)
        else:
            return "Patches CSV file not found", 404
    except Exception as e:
        print(f"Warning: Could not load patches CSV: {e}")
        return "Error loading patches CSV", 500

    current_idx = patches_df[
        (patches_df["run_id"] == run_id) & (patches_df["instance_id"] == instance_id)
    ].index

    if len(current_idx) == 0:
        return "Instance not found", 404

    current_idx = current_idx[0]
    instance = patches_df.iloc[current_idx].to_dict()

    prev_instance = None
    next_instance = None

    if current_idx > 0:
        prev_instance = patches_df.iloc[current_idx - 1].to_dict()

    if current_idx < len(patches_df) - 1:
        next_instance = patches_df.iloc[current_idx + 1].to_dict()

    return render_template(
        "patch_view.html",
        instance=instance,
        prev_instance=prev_instance,
        next_instance=next_instance,
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
    app.run(debug=False, host="0.0.0.0", port=5760)
