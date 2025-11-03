"""
Helper script to systematically compare patches between different models
and extract insights about why certain models succeed where others fail on specific problems.
"""

import json
from typing import Dict
import openai
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from gso.utils.io import load_gso_dataset, load_report
from gso.constants import EVALUATION_REPORTS_DIR, RUN_EVALUATION_LOG_DIR

MODEL_CONFIG = {
    "claude-sonnet-4.5": {
        "report": "claude-sonnet-4-5-20250929_maxiter_100_N_v0.51.1-no-hint-run_1.pass.report.json",
        "run_dir": "claude-sonnet-4-5-20250929_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
    "gpt-5": {
        "report": "gpt-5-2025-08-07_maxiter_100_N_v0.51.1-no-hint-run_1.pass.report.json",
        "run_dir": "gpt-5-2025-08-07_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
    "o3": {
        "report": "o3_maxiter_100_N_v0.35.0-no-hint-run_1.pass.report.json",
        "run_dir": "o3_maxiter_100_N_v0.35.0-no-hint-run_1",
    },
    "claude-sonnet-4": {
        "report": "claude-sonnet-4-20250514_maxiter_100_N_v0.35.0-no-hint-run_1.pass.report.json",
        "run_dir": "claude-sonnet-4-20250514_maxiter_100_N_v0.35.0-no-hint-run_1",
    },
    "claude-opus-4": {
        "report": "claude-opus-4-20250514_maxiter_100_N_v0.51.1-no-hint-run_1.pass.report.json",
        "run_dir": "claude-opus-4-20250514_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
}


def load_model_patch(run_dir: str, instance_id: str) -> str | None:
    """Load the patch submitted by a model for a specific instance."""
    patch_path = RUN_EVALUATION_LOG_DIR / "pass" / run_dir / instance_id / "patch.diff"

    if patch_path.exists():
        try:
            with open(patch_path, "r") as f:
                return f.read()
        except Exception as e:
            return None
    return None


def get_unique_solves(reports: Dict[str, dict], target_model: str) -> set:
    """Identify problems that ONLY the target model solved."""
    target_solves = set(reports[target_model]["instance_sets"]["opt_commit_ids"])

    other_solves = set()
    for model_name, report in reports.items():
        if model_name != target_model:
            other_solves.update(set(report["instance_sets"]["opt_commit_ids"]))

    unique_solves = target_solves - other_solves
    return unique_solves


def analyze_patch_with_llm(
    instance_id: str,
    repo: str,
    api: str,
    human_diff: str,
    target_diff: str,
    other_diffs: Dict[str, str],
    target_model: str,
) -> Dict:
    """
    Use GPT-5 to analyze and compare the patches for a specific problem.
    """

    # Prepare the prompt with better context
    other_models_text = ""
    for model_name, diff in other_diffs.items():
        if diff:
            other_models_text += f"\n\n{model_name.upper()} PATCH (failed to achieve performance target):\n```\n{diff[:1500]}{'...' if len(diff) > 1500 else ''}\n```"
        else:
            other_models_text += f"\n\n{model_name.upper()}: No patch available"

    prompt = f"""You are analyzing code optimization patches for a software performance benchmark. 

CONTEXT: This is a performance optimization benchmark where models must generate code patches that improve performance. The "opt_commit" threshold means the patch must achieve a measurable performance improvement to be considered successful.

PROBLEM: {instance_id}
Repository: {repo}  
API: {api}

HUMAN ORACLE PATCH (expert solution, achieves performance target):
```
{human_diff[:1500]}{'...' if len(human_diff) > 1500 else ''}
```

{target_model.upper()} PATCH (succeeded, achieved performance target):
```
{target_diff[:1500]}{'...' if len(target_diff) > 1500 else ''}
```
{other_models_text}

Analyze these patches and provide insights in JSON format:

{{
    "human_approach": "Concrete description of what the human expert did",
    "{target_model}_approach": "Concrete description of what {target_model} did", 
    "why_{target_model}_succeeded": [
        "Specific technical reason 1",
        "Specific technical reason 2"
    ],
    "why_others_failed": {{
        "model_name": [
            "Specific technical reason for failure"
        ]
    }},
    "key_insights": [
        "Concrete insight about optimization differences"
    ],
    "optimization_techniques": {{
        "human": ["specific_technique1", "specific_technique2"],
        "{target_model}": ["specific_technique1", "specific_technique2"],
        "others": {{
            "model_name": ["specific_technique1"]
        }}
    }}
}}

Focus on CONCRETE technical details:
- What specific code changes were made
- What optimization techniques were used (vectorization, caching, algorithmic changes, etc.)
- Why one approach worked and others didn't
- Code quality and precision differences
- Performance bottleneck identification

Be specific about the actual code changes, not high-level concepts."""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-5",
            reasoning_effort="high",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response
        response_text = response.choices[0].message.content

        # Try to find JSON in the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON decode error: {e}",
                    "raw_response": response_text,
                }
        else:
            return {
                "error": "Could not find JSON in response",
                "raw_response": response_text,
            }

    except Exception as e:
        return {"error": str(e)}


def analyze_single_problem(args):
    """Analyze a single problem - designed for parallel execution."""
    instance_id, instance, run_dirs, other_diffs, target_model = args

    print(f"  Analyzing {instance_id}...")

    # Load patches
    human_diff = instance.gt_diff
    target_diff = load_model_patch(run_dirs[target_model], instance_id)

    if not target_diff:
        return {"instance_id": instance_id, "error": f"No {target_model} patch found"}

    # Analyze with LLM
    analysis = analyze_patch_with_llm(
        instance_id,
        instance.repo,
        instance.api,
        human_diff,
        target_diff,
        other_diffs,
        target_model,
    )

    if "error" in analysis:
        return {"instance_id": instance_id, "error": analysis["error"]}

    # Store all the data
    analysis["instance_id"] = instance_id
    analysis["repo"] = instance.repo
    analysis["api"] = instance.api
    analysis["target_model"] = target_model
    analysis["human_diff"] = human_diff
    analysis["target_diff"] = target_diff
    analysis["other_diffs"] = other_diffs

    return analysis


def main():
    print("=" * 80)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 80)

    # Configuration - can be modified to analyze any model
    TARGET_MODEL = "gpt-5"

    # Load all reports
    print("\nLoading evaluation reports...")
    reports = {}
    run_dirs = {}
    for model_name, config in MODEL_CONFIG.items():
        try:
            reports[model_name] = load_report(config["report"])
            run_dirs[model_name] = config["run_dir"]
            print(f"  {model_name}")
        except Exception as e:
            print(f"  ERROR {model_name}: {e}")
            return

    # Load GSO dataset
    print("\nLoading GSO dataset...")
    try:
        dataset = load_gso_dataset("gso-bench/gso", "test")
        instance_lookup = {inst.instance_id: inst for inst in dataset}
        print(f"  Loaded {len(dataset)} instances")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # Get unique solves
    unique_solves = get_unique_solves(reports, TARGET_MODEL)
    print(f"\nFound {len(unique_solves)} unique solves by {TARGET_MODEL}")

    # Prepare data for parallel analysis
    print(f"\nPreparing data for parallel analysis...")
    analysis_args = []

    for instance_id in sorted(unique_solves):
        if instance_id not in instance_lookup:
            continue

        instance = instance_lookup[instance_id]

        # Load other model patches
        other_diffs = {}
        for model_name in MODEL_CONFIG.keys():
            if model_name == TARGET_MODEL:
                continue
            patch = load_model_patch(run_dirs[model_name], instance_id)
            if patch:
                other_diffs[model_name] = patch

        analysis_args.append(
            (instance_id, instance, run_dirs, other_diffs, TARGET_MODEL)
        )

    print(f"  Prepared {len(analysis_args)} problems for analysis")

    # Run parallel analysis
    print(f"\nRunning parallel analysis with {min(4, len(analysis_args))} workers...")
    all_results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_args = {
            executor.submit(analyze_single_problem, args): args
            for args in analysis_args
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_args), 1):
            args = future_to_args[future]
            instance_id = args[0]

            try:
                result = future.result()
                all_results.append(result)
                if "error" not in result:
                    print(f"  COMPLETE {i}/{len(analysis_args)}: {instance_id}")
                else:
                    print(
                        f"  ERROR {i}/{len(analysis_args)}: {instance_id} - {result['error']}"
                    )
            except Exception as e:
                error_result = {"instance_id": instance_id, "error": f"Exception: {e}"}
                all_results.append(error_result)
                print(
                    f"  ERROR {i}/{len(analysis_args)}: {instance_id} - Exception: {e}"
                )

    # Separate successful and failed analyses
    analyses = [r for r in all_results if "error" not in r]
    failed_analyses = [r for r in all_results if "error" in r]

    print(f"\n📊 Analysis Results:")
    print(f"   Successful: {len(analyses)}")
    print(f"   Failed: {len(failed_analyses)}")

    # Save results with all diffs
    results = {
        "total_analyzed": len(analyses),
        "total_attempted": len(all_results),
        "successful_analyses": analyses,
        "failed_analyses": failed_analyses,
        "all_results": all_results,  # Include everything for debugging
        "metadata": {
            "unique_solves_count": len(unique_solves),
            "models_compared": list(MODEL_CONFIG.keys()),
        },
    }

    # Create reports directory structure
    analysis_reports_dir = EVALUATION_REPORTS_DIR / "analysis"
    analysis_reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_reports_dir / f"{TARGET_MODEL}_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {output_path}")
    print(f"   Analyzed {len(analyses)} problems")
    print(f"   Target model: {TARGET_MODEL}")
    print(f"   All diffs and instance IDs included in JSON")


if __name__ == "__main__":
    main()
