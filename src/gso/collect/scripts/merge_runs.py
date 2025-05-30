from gso.utils.io import load_problems, save_problems
from gso.constants import *


def merge_results(latest_path: str, previous_path: str, output_path: str):
    """
    merge results from two execution runs of the same experiment, prioritizing the latest results
    and only merging when necessary based on validity criteria.

    Args:
        latest_path: Path to the latest experiment results JSON
        previous_path: Path to the previous experiment results JSON
        output_path: Path for the merged results JSON
    """
    latest_problems = load_problems(latest_path)
    previous_problems = load_problems(previous_path)

    # Create mappings by problem ID
    latest_map = {p.pid: p for p in latest_problems}
    previous_map = {p.pid: p for p in previous_problems}

    # merged problems list - start with all problems from latest
    merged_problems = []

    for pid, latest_prob in latest_map.items():
        if pid in previous_map:
            prev_prob = previous_map[pid]
            test_id_map = {}  # Maps (commit, old_test_id) -> new_test_id

            print(f"\nMerging problem: {pid}")

            # Merge tests by finding matching quick_hash
            latest_test_map = {}
            for test in latest_prob.tests:
                latest_test_map[test.quick_hash] = test

            # Merge Test samples or add new Tests
            for prev_test in prev_prob.tests:
                quick_hash = prev_test.quick_hash

                if quick_hash in latest_test_map:
                    latest_test = latest_test_map[quick_hash]
                    orig_count = len(latest_test.samples)

                    # Create mappings from old position to new position
                    for i, _ in enumerate(prev_test.samples):
                        new_id = orig_count + i
                        test_id_map[(quick_hash, i)] = new_id

                    latest_test.samples.extend(prev_test.samples)
                else:
                    latest_prob.tests.append(prev_test)
                    for i in range(len(prev_test.samples)):
                        test_id_map[(quick_hash, i)] = i

            # Merge results assuming only one run in both
            print("  Merging results:")
            try:
                latest_key = list(latest_prob.results.keys())[0]
                latest_results = latest_prob.results[latest_key]
                prev_key = list(prev_prob.results.keys())[0]
                prev_results = prev_prob.results[prev_key]
                print(f"  Found {len(latest_results)} results in latest run")
                print(f"  Found {len(prev_results)} results in previous run")

                # Start with all results from latest
                merged_results = latest_results.copy()

                for ct in prev_results:
                    test_file = ct["test_file"]
                    commit = ct["commit"]
                    test_id = ct["test_id"]

                    # Create a copy of the result
                    new_result = ct.copy()

                    # Update the test_id using our mapping
                    if (commit, test_id) in test_id_map:
                        new_test_id = test_id_map[(commit, test_id)]
                        print(f"    {test_file}: test_id {test_id} -> {new_test_id}")
                        new_result["test_id"] = new_test_id
                        new_result["test_file"] = test_file.replace(
                            str(test_id), str(new_test_id)
                        )
                    else:
                        print(
                            f"    {test_file}: keeping test_id {test_id} (no mapping found)"
                        )

                    merged_results.append(new_result)

                print(f"  Total merged results: {len(merged_results)}")
                latest_prob.results[latest_key] = merged_results

                # assert that there are no commit-test_id duplicates
                commits = set()
                for r in merged_results:
                    key = (r["commit"], r["test_id"])
                    assert key not in commits, f"Duplicate result: {key}"
                    commits.add(key)

            except Exception as e:
                print(f"  Error merging results for {pid}: {e}")

        merged_problems.append(latest_prob)
        print("=" * 50, "\n")

    save_problems(output_path, merged_problems)
    print(f"merged results saved to {output_path}")


exp_id = "pandas"
exp_dir = EXPS_DIR / f"{exp_id}"
previous_path = exp_dir / "v0" / f"{exp_id}_results.json"
latest_path = exp_dir / "v1" / f"{exp_id}_results.json"
output_path = exp_dir / f"{exp_id}_results_merged.json"

merge_results(latest_path, previous_path, output_path)
