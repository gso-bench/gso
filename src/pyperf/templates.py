SYSTEM_MESSAGE_DIFF = """You are an expert Python performance tester who writes performance tests during code review. 

You will be given a summary of a Pull Request performing some performance enhancements. This summary will contain the following information:
1. Repo Name
2. Title of the PR
3. Description of the PR
4. Files changed in the PR
5. A code diff of the changes made in the PR

Your task is to write a comprehensive performance test that will verify the performance enhancements. 
Depending on the optimization performed, you can measure performance along various axis of your choice such as latency, memory usage, etc.
You will return a performance test that estimates the performance of one version of the code. 
This test will be run separately on both, the original and the new code, and the results will be compared manually.

Additional Instructions:
1. Ignore any changes made to the test files.
2. Focus on testing the core optimizations performed
4. The performance test should be written in Python.
5. You can use any libraries you want.
6. Warmup: Before the actual measurements, perform operations to stabilize the system if required.
7. Repetitions: Run tests several times and calculate a more reliable average metric.

Respond in the following manner:

Reasoning: <Your thoughts on what optimizations were made and how you plan to test them>

Performance Test:
```
<Your performance test>
```
"""


PR_SUMMARY_DIFF = """
Repo Name: google/jax

Title: {}

Description: {}

Files changed: {}

Code Diff: {}

Please write a performance test for this PR.
"""


SYSTEM_MESSAGE_SLICE = """You are an expert Python performance tester who writes performance tests during code review. 

You will be given a function (and other code it depends on) in the following format:
1. Repo Name
2. Name of the function to test
5. Code of the function to test

Your task is to write a comprehensive performance test that will benchmark any performance enhancements performed on the function.
Depending on the optimization performed, you can measure performance along various axis of your choice such as latency, memory usage, etc.
You will return a performance test that estimates the performance of one version of the code. 
This test will be run separately on both, the original and new optimized code, and the results will be compared manually.

Additional Instructions:
1. Focus on testing the core optimizations performed
2. The performance test should be written in Python.
3. You can use any libraries you want.
4. Warmup: Before the actual measurements, perform operations to stabilize the system if required.
5. Repetitions: Run tests several times and calculate a more reliable average metric.

Respond in the following manner:

Reasoning: <Your thoughts on how you plan to test performance>

Performance Test:
```
<Your performance test>
```
"""


PR_SUMMARY_SLICE = """
Repo Name: google/jax

Function Name: {function_name}

Function Code: {sliced_context}

Please write a performance test for the function `{function_name}`.
"""


SETUP_SCRIPT = """
# Variable for base commit (state before PR)
BASE_COMMIT={base_commit}

# Clone the repository
git clone https://github.com/{repo_full_name}.git ~/buckets/my_repos/{repo_name}

# Move the cloned repo to r2e's local path
python ~/Projects/r2e/r2e/repo_builder/setup_repos.py --local_repo_path ~/buckets/my_repos/{repo_name}

# Go to the local repo
cd ~/buckets/local_repoeval_bucket/repos/LOCAL___{repo_name}

# Checkout the base commit
git checkout $BASE_COMMIT

# Build the repo
pip install -e . 
"""


APPLY_PATCH_SCRIPT = """
# Apply the patch to the target repository directory
git -C ~/buckets/local_repoeval_bucket/repos/LOCAL___{repo_name} apply {diff_file}

# Optionally, you might want to verify the patch application
# git -C ~/buckets/local_repoeval_bucket/repos/LOCAL___{repo_name} diff
"""
