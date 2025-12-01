import os

# Constants for the evaluation script
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
INSTALL_FAIL = ">>>>> Init Failed"
INSTALL_PASS = ">>>>> Init Succeeded"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_PASSED = ">>>>> Tests Passed"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
MAIN_REGRESS_WARNING = ">>>>> WARNING: Error in main branch! continuing..."
START_EVAL_PATCH = "Applying patch..."
START_BASE_OUTPUT = ">>>>> Start Base Output"
END_BASE_OUTPUT = ">>>>> End Base Output"
START_PATCH_OUTPUT = ">>>>> Start Patch Output"
END_PATCH_OUTPUT = ">>>>> End Patch Output"
START_COMMIT_OUTPUT = ">>>>> Start Commit Output"
END_COMMIT_OUTPUT = ">>>>> End Commit Output"
START_MAIN_OUTPUT = ">>>>> Start Main Output"
END_MAIN_OUTPUT = ">>>>> End Main Output"
TEST_DELIMITER = ">>>>> Test {i}"


APPLY_PATCH_HELPER = """
apply_patch() {{
    local patch_file=$1
    local applied=false
    
    # Ensure patch_file exists
    if [ ! -f "$patch_file" ]; then
        echo "Patch file does not exist: $patch_file"
        echo "{failed}"
        exit 1
    fi

    # reset, if any uncommited changes
    if ! git diff-index --quiet HEAD --; then
        echo "Uncommitted changes found. Resetting..."
        git reset --hard HEAD
    fi
    
    # file patterns to exclude
    local exclude_options="--exclude=.venv/* --exclude=.git/* --exclude=__pycache__/* --exclude=*.egg-info/* --exclude=*.json --exclude=*.txt --exclude=*.csv --exclude=*.log --exclude=*.pkl"

    # Try git apply with verbose output
    if git apply --verbose $exclude_options "$patch_file" 2>&1; then
        echo "Successfully applied patch using git apply --verbose"
        echo "{passed}"
        applied=true
    elif git apply --verbose --ignore-space-change $exclude_options "$patch_file" 2>&1; then
        echo "Successfully applied patch using git apply --verbose --ignore-space-change"
        echo "{passed}"
        applied=true
    elif git apply --verbose --ignore-space-change --reject $exclude_options "$patch_file" 2>&1; then
        echo "Successfully applied patch using git apply --verbose --ignore-space-change --reject"
        echo "{passed}"
        applied=true
    elif patch --batch --fuzz=5 -p1 -i "$patch_file" 2>&1; then
        echo "Successfully applied patch using patch command"
        echo "{passed}"
        applied=true
    fi

    if [ "$applied" = false ]; then
        echo "Failed to apply patch using all methods"
        echo "{failed}"
        exit 1
    fi
}}
"""

RUN_TESTS_HELPER = """
run_tests_for_commit() {{
    local test_file=$1
    local result_file=$2
    local flag=$3
    local prefix=$4
    local iterations={max_iterations}
    
    echo "Running test $test_file $iterations times..."
    for i in $(seq 1 $iterations); do
        echo "  Iteration $i/$iterations"
        status=0
        timeout {max_time}s python "$test_file" "$result_file" "$flag" --file_prefix "$prefix" || status=$?

        if [ $status -eq 124 ]; then
            # Exit code 124 indicates timeout
            echo "{timeout}"
            return 1

        elif [ $status -ne 0 ]; then
            # If main fails, warn and return
            if [[ "$result_file" == "main"* ]]; then
                echo "{warning}"
                echo "N/A" > "$result_file"
                return 0
            fi

            # Any other non-zero exit indicates test error
            echo "{error}"
            return 1
        fi
    done
    
    # If we get here, all iterations passed
    echo "{passed}"
    return 0
}}
"""

INSTALL_HELPER = """
install_repo() {{
    {{
        {install_commands} || {{
            echo "{failed}"
            return 1
        }}
        
        # If we get here, everything succeeded
        echo "{passed}"
        return 0
    }}
}}
"""

SKIP_MAIN = True  # Toggle to skip main branch testing
RUN_BASE = (
    """run_tests_for_commit /gso_test_{i}.py "base_{i}.txt" "--reference" "gso_{i}" """
)
RUN_PATCH = (
    """run_tests_for_commit /gso_test_{i}.py "result_{i}.txt" "--eqcheck" "gso_{i}" """
)
RUN_COMMIT = """run_tests_for_commit /gso_test_{i}.py "commit_{i}.txt" "--reference" "gso_{i}" """
RUN_MAIN = (
    """run_tests_for_commit /gso_test_{i}.py "main_{i}.txt" "--reference" "gso_{i}" """
)
PRINT_PERF = lambda i, f: f"""echo "{TEST_DELIMITER.format(i=i)}" && cat {f}_{i}.txt"""
MAX_TIME = lambda repo: (
    0 if repo in ["abetlen/llama-cpp-python", "huggingface/tokenizers"] else 300
)
MAX_ITERS = lambda repo: (
    1
    if repo
    in ["abetlen/llama-cpp-python", "microsoft/onnxruntime", "huggingface/tokenizers"]
    else 5
)


def get_eval_script(instance) -> str:
    install_commands = instance.install_commands
    opt_commit = instance.opt_commit
    reset_repo_commands = instance.reset_repo_commands
    test_count = instance.test_count
    repo = instance.repo

    # Avoid deleting result files
    if "git clean -xfd" in install_commands:
        install_commands.remove("git clean -xfd")

    run_base = "\n".join([RUN_BASE.format(i=i) for i in range(test_count)])

    # Set tokens
    setup_tokens = [
        f"export HF_TOKEN={os.getenv('HF_TOKEN')}" if os.getenv("HF_TOKEN") else "",
    ]

    eval_commands = [
        "source .venv/bin/activate",
        # ----------- set up tokens ------------
        'echo "Setting up tokens..."',
        *setup_tokens,
        # ----------- base and patch perf testing ------------
        'echo "Running performance test before patch..."',
        *[RUN_BASE.format(i=i) for i in range(test_count)],
        f'echo "{START_EVAL_PATCH}"',
        'apply_patch "/tmp/patch.diff"',
        'echo "Installing repo..."',
        "install_repo",
        'echo "Running performance test after patch..."',
        *[RUN_PATCH.format(i=i) for i in range(test_count)],
        f'echo "{START_BASE_OUTPUT}"',
        *[PRINT_PERF(i, "base") for i in range(test_count)],
        f'echo "{END_BASE_OUTPUT}"',
        f'echo "{START_PATCH_OUTPUT}"',
        *[PRINT_PERF(i, "result") for i in range(test_count)],
        f'echo "{END_PATCH_OUTPUT}"',
        # ----------- reset the repo to remote origin ------------
        f"{reset_repo_commands}",
        # ----------- commit and main perf testing ------------
        *(
            []
            if SKIP_MAIN
            else [
                'echo "Installing repo..."',
                "install_repo",
                'echo "Running performance test for main..."',
                *[RUN_MAIN.format(i=i) for i in range(test_count)],
                f'echo "{START_MAIN_OUTPUT}"',
                *[PRINT_PERF(i, "main") for i in range(test_count)],
                f'echo "{END_MAIN_OUTPUT}"',
            ]
        ),
        'echo "Checking out commit..."',
        f"git checkout {opt_commit}",
        'echo "Installing repo..."',
        "install_repo",
        'echo "Running performance test for commit..."',
        *[RUN_COMMIT.format(i=i) for i in range(test_count)],
        f'echo "{START_COMMIT_OUTPUT}"',
        *[PRINT_PERF(i, "commit") for i in range(test_count)],
        f'echo "{END_COMMIT_OUTPUT}"',
    ]

    return (
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",  # add x for debugging
                APPLY_PATCH_HELPER.format(
                    failed=APPLY_PATCH_FAIL, passed=APPLY_PATCH_PASS
                ),
                RUN_TESTS_HELPER.format(
                    timeout=TESTS_TIMEOUT,
                    warning=MAIN_REGRESS_WARNING,
                    error=TESTS_ERROR,
                    passed=TESTS_PASSED,
                    max_iterations=MAX_ITERS(repo),
                    max_time=MAX_TIME(repo),
                ),
                INSTALL_HELPER.format(
                    install_commands="\n\t\t".join(install_commands),
                    failed=INSTALL_FAIL,
                    passed=INSTALL_PASS,
                ),
            ]
        )
        + "\n"
        + "\n".join(eval_commands)
    )
