APPLY_PATCH_HELPER = """
apply_patch() {
    local patch_file=$1
    local applied=false
    
    # Ensure patch_file exists
    if [ ! -f "$patch_file" ]; then
        echo "Patch file does not exist: $patch_file"
        exit 1
    fi

    # Try git apply with verbose output
    if git apply --verbose "$patch_file" 2>&1; then
        echo "Successfully applied patch using git apply --verbose"
        applied=true
    elif git apply --verbose --reject "$patch_file" 2>&1; then
        echo "Successfully applied patch using git apply --verbose --reject"
        applied=true
    elif patch --batch --fuzz=5 -p1 -i "$patch_file" 2>&1; then
        echo "Successfully applied patch using patch command"
        applied=true
    fi

    if [ "$applied" = false ]; then
        echo "Failed to apply patch using all methods"
        exit 1
    fi
}
"""

RUN_TESTS_HELPER = """
run_tests_for_commit() {
    local test_file=$1
    local result_file=$2
    local flag=$3
    local prefix=$4
    local iterations=5
    
    echo "Running test $test_file $iterations times..."
    for i in $(seq 1 $iterations); do
        echo "  Iteration $i/$iterations"
        timeout 300s python "$test_file" "$result_file" "$flag" --file_prefix "$prefix"
    done
}
"""


def get_eval_script(install_commands) -> str:
    eval_commands = [
        "source .venv/bin/activate",
        'echo "Running performance test before patch..."',
        'run_tests_for_commit "/pyperf_test.py" "base.txt" "--reference" "pyperf"',
        'echo "Applying patch..."',
        'apply_patch "/tmp/patch.diff"',
    ]

    # Avoid deleting result files
    if "git clean -xfd" in install_commands:
        install_commands.remove("git clean -xfd")

    eval_commands += install_commands
    eval_commands += [
        'echo "Running performance test after patch..."',
        'run_tests_for_commit "/pyperf_test.py" "result.txt" "--eqcheck" "pyperf"',
    ]

    return (
        "\n".join(
            [
                "#!/bin/bash",
                "set -euxo pipefail",
                APPLY_PATCH_HELPER,
                RUN_TESTS_HELPER,
            ]
        )
        + "\n"
        + "\n".join(eval_commands)
    )
