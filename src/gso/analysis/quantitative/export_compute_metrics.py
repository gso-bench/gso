"""
Export compute metrics (time, turns, score) for all models to JSON for the GSO website.

Scores are read automatically from eval report files (*.pass.report.json).

Usage:
    uv run src/gso/analysis/quantitative/export_compute_metrics.py \
        --models "Label::run_dir::eval_dir" [...] \
        --output_file ~/gso-bench.github.io/assets/compute_metrics.json
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

from gso.constants import EVALUATION_REPORTS_DIR


def get_instance_stats(run_dir: str) -> list[dict]:
    """Get per-instance wall-clock duration (min) and turn count."""
    instances = []

    # Try trajectories dir first
    traj_dir = os.path.join(run_dir, "trajectories")
    if os.path.isdir(traj_dir):
        for f in os.listdir(traj_dir):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(traj_dir, f)) as fh:
                data = json.load(fh)
            history = data if isinstance(data, list) else data.get("history", [])
            if not history or len(history) < 2:
                continue
            t0 = datetime.fromisoformat(history[0]["timestamp"])
            t1 = datetime.fromisoformat(history[-1]["timestamp"])
            agent_turns = sum(1 for e in history if e.get("action") is not None)
            instances.append({
                "duration_min": (t1 - t0).total_seconds() / 60,
                "turns": agent_turns,
            })
        return instances

    # Fall back to output.jsonl
    output = os.path.join(run_dir, "output.jsonl")
    if os.path.isfile(output):
        with open(output) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                history = rec.get("history") or []
                if len(history) < 2:
                    continue
                try:
                    t0 = datetime.fromisoformat(history[0]["timestamp"])
                    t1 = datetime.fromisoformat(history[-1]["timestamp"])
                    agent_turns = sum(1 for e in history if e.get("action") is not None)
                    instances.append({
                        "duration_min": (t1 - t0).total_seconds() / 60,
                        "turns": agent_turns,
                    })
                except (KeyError, ValueError):
                    pass
    return instances


def get_score_from_eval(eval_dir: str) -> float | None:
    """Read Opt@1 score from the eval report."""
    # eval_dir may be a glob pattern
    dirs = sorted(glob.glob(eval_dir))
    for d in dirs:
        # Find *.pass.report.json in the directory
        reports = glob.glob(os.path.join(d, "*.pass.report.json"))
        for report_path in reports:
            with open(report_path) as f:
                report = json.load(f)
            summary = report.get("summary", {})
            total = summary.get("total_instances", 0)
            opt_commit = summary.get("opt_commit", 0)
            if total > 0:
                return round(opt_commit / total * 100, 2)
    return None


def bootstrap_ci(vals, stat_fn=np.median, n_boot=2000, ci=95):
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    arr = np.array(vals)
    boots = [float(stat_fn(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return float(stat_fn(arr)), lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True,
        help='Each entry: "DisplayName::run_dir::eval_dir"',
    )
    default_output = str((EVALUATION_REPORTS_DIR / "analysis" / "compute_metrics.json").resolve())
    parser.add_argument(
        "--output_file", default=default_output,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    model_configs = []
    for entry in args.models:
        parts = entry.split("::")
        if len(parts) != 3:
            parser.error(f"Invalid format: {entry}. Expected 'Label::run_dir::eval_dir'")
        name, run_dir, eval_dir = parts
        model_configs.append((name, run_dir, eval_dir))

    # Collect stats in parallel
    run_dirs = [m[1] for m in model_configs]
    with ProcessPoolExecutor() as pool:
        all_stats = list(pool.map(get_instance_stats, run_dirs))

    results = {}
    for (name, run_dir, eval_dir), stats in zip(model_configs, all_stats):
        score = get_score_from_eval(eval_dir)
        if score is None:
            print(f"WARN {name}: no eval report found in {eval_dir}, skipping")
            continue
        if not stats:
            print(f"SKIP {name}: no trajectory data in {run_dir}")
            continue

        durations = [s["duration_min"] for s in stats]
        turns = [s["turns"] for s in stats]

        time_med, time_lo, time_hi = bootstrap_ci(durations)
        turns_med, turns_lo, turns_hi = bootstrap_ci(turns)

        results[name] = {
            "score": score,
            "median_time_min": round(time_med, 1),
            "time_ci_lo": round(time_lo, 1),
            "time_ci_hi": round(time_hi, 1),
            "median_turns": round(turns_med, 1),
            "turns_ci_lo": round(turns_lo, 1),
            "turns_ci_hi": round(turns_hi, 1),
            "num_instances": len(stats),
        }
        print(
            f"{name}: score={score}, "
            f"time={time_med:.1f} [{time_lo:.1f}-{time_hi:.1f}]min, "
            f"turns={turns_med:.0f} [{turns_lo:.0f}-{turns_hi:.0f}], "
            f"n={len(stats)}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved {args.output_file} with {len(results)} models")


if __name__ == "__main__":
    main()
