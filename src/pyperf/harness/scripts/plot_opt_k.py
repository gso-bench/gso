import subprocess
import re
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description="Run and plot Opt@K evaluations")
parser.add_argument(
    "--dataset_name", type=str, default="manishs/pyperf", help="Dataset name"
)
parser.add_argument(
    "--prediction_paths",
    type=str,
    nargs="+",
    help="Glob pattern for prediction paths",
    required=True,
)
parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
parser.add_argument("--run_id", type=str, default="test", help="Run identifier")
parser.add_argument("--model_name", type=str, help="Model name", required=True)
parser.add_argument(
    "--reformat_reports", action="store_true", help="Whether to just reformat reports"
)
parser.add_argument("--k", type=int, default=10, help="Maximum K value")
parser.add_argument(
    "--output_dir", type=str, default="plots", help="Directory to save plots"
)
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Store results for each k
k_values = []
base_values = []
commit_values = []
main_values = []

for k in tqdm(range(1, args.k + 1), desc="Processing K predictions"):
    # Construct the command
    cmd = [
        "uv",
        "run",
        "src/pyperf/harness/opt@k.py",  # Changed to opt@k.py
        "--dataset_name",
        args.dataset_name,
        "--prediction_paths",
    ]

    # Add paths as separate arguments (not as a single string)
    cmd.extend(args.prediction_paths)

    # Add remaining arguments
    cmd.extend(
        [
            "--timeout",
            str(args.timeout),
            "--run_id",
            args.run_id,
            "--model_name",  # Changed to model_name to match opt@k.py
            args.model_name,
            "--k",
            str(k),
        ]
    )

    if args.reformat_reports:
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
output_path = os.path.join(args.output_dir, "opt_at_k.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
