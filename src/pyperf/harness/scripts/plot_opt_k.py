import subprocess
import re
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Store results for each k
k_values = []
base_values = []
commit_values = []
main_values = []

dataset_name = "pyperf_pandas_dataset.jsonl"
# prediction_paths = "~/OpenHands/evaluation/evaluation_outputs/outputs/manishs__pyperf_pandas-test/CodeActAgent/claude-3-5-sonnet-v2-20241022_maxiter_50_N_v0.25.0-no-hint-run_*/output.pyperf.jsonl"
prediction_paths = "~/OpenHands/evaluation/evaluation_outputs/outputs/manishs__pyperf_pandas-test/CodeActAgent/o3-mini_maxiter_50_N_v0.25.0-no-hint-run_*/output.pyperf.jsonl"
# Expand the glob pattern into a list of paths
path_list = glob.glob(os.path.expanduser(prediction_paths))
timeout = 3600
run_id = "test"
model = "claude"
reformat_reports = True
K = 10

for k in tqdm(range(1, K + 1), desc="Processing K predictions"):
    # Construct the command
    cmd = [
        "uv",
        "run",
        "src/pyperf/harness/opt@k.py",  # Changed to opt@k.py
        "--dataset_name",
        dataset_name,
        "--prediction_paths",
    ]

    # Add paths as separate arguments (not as a single string)
    cmd.extend(path_list)

    # Add remaining arguments
    cmd.extend(
        [
            "--timeout",
            str(timeout),
            "--run_id",
            run_id,
            "--model_name",  # Changed to model_name to match opt@k.py
            model,
            "--k",
            str(k),
        ]
    )

    if reformat_reports:
        cmd.append("--reformat_reports")

    try:
        # Run the command and capture output
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

        # Extract metrics using regex
        base_match = re.search(rf"Opt\(base\)@{k}: \d+ \((\d+\.\d+)%\)", output)
        commit_match = re.search(rf"Opt\(commit\)@{k}: \d+ \((\d+\.\d+)%\)", output)
        main_match = re.search(rf"Opt\(main\)@{k}: \d+ \((\d+\.\d+)%\)", output)

        if base_match and commit_match and main_match:
            k_values.append(k)
            base_values.append(float(base_match.group(1)))
            commit_values.append(float(commit_match.group(1)))
            main_values.append(float(main_match.group(1)))

            print(f"Completed run for k={k}")
            print(f"Base: {base_values[-1]}%")
            print(f"Commit: {commit_values[-1]}%")
            print(f"Main: {main_values[-1]}%")
            print("-" * 40)

    except subprocess.CalledProcessError as e:
        print(f"Error running command for k={k}: {e}")
        print(f"Output: {e.output}")
        continue

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, base_values, "o-", label="Opt(base)@k", linewidth=2)
plt.plot(k_values, commit_values, "s-", label="Opt(commit)@k", linewidth=2)
plt.plot(k_values, main_values, "^-", label="Opt(main)@k", linewidth=2)

plt.xlabel("# Agent Rollouts (K)")
plt.ylabel("Optimized (%)")
plt.title("Inference time scaling for Opt@K")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("plots/opt_at_k.png", dpi=300, bbox_inches="tight")
print("Plot saved as plots/opt_at_k.png")
