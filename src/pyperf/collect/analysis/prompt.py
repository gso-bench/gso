PERF_ANALYSIS_MESSAGE = """You are an expert programmer who is annotating data for training and testing code language models.

You will be given a GitHub commit patch content. Your goal is to identify whether the commit is performance or optimization related. The commit should satisfy the following conditions:
1. The commit should modify at least one non-test file. It modify source code in a non-trivial manner and not just fix comments or documentation.
2. The changes need not directly mention performance or optimization in the commit message but should be related to performance optimization.
3. BE CRITICAL in your analysis. Just keywords like optimize or perf in the commit does not mean the commit is performance related. E.g., "add optimize flag", "fix get_optimize function", etc.
4. BE CRITICAL in your analysis: A commit message mentioning 'optimize' might be a fix to a function/module named 'optimize' and not a performance optimization. e.g., `optimize.linprog: make HiGHS default and deprecate old methods`.
5. The changes should not be related to bug fixes, simple refactoring, or adding new features.
6. The changes should preferably affect the performance of existing high-level or top-level APIs in the repo. This can be directly or indirectly via changes to internal APIs.
7. The commit should affect performance on CPU and should be testable without a GPU. Ignore commits related to GPU/TPU workloads only.
8. Ignore commits that only work on specific hardware (e.g., Qualcomm GPUs, Metal, Vulkan, etc.). Keep generalization and testability in mind.

Analyze the commit using natural language reasoning enclosed in [REASON] [/REASON] tags.
Then write YES or NO based on the conditions mentioned above enclosed in [ANSWER] [/ANSWER] tags.
Remember to close all tags properly.

Commit Information:
{diff_text}

Commit Message:
{message}
"""

PERF_IDENTIFY_API_DOCS = ""

PERF_IDENTIFY_API_SYSTEM = """You are an expert programmer who is annotating data for training and testing code language models.

You will be given a performance or optimization related GitHub commit patch content. Your goal is to identify a list of APIs (functions or methods of a class) that are affected by the commit. Some additional instructions:
1. The APIs should be high-level or top-level APIs in the repo. E.g., pd.read_csv (pandas), requests.get (requests), model.generate (transformers), etc.
2. By high/top-level, we mean APIs that are not internal helper functions.
3. If the commit affects multiple APIs, list them all separated by commas.
4. For methods, use the format "ClassName.method_name" (e.g., DataFrame.dropna).
5. NOTE: Find Affected PYTHON APIs only
    - Do not add backend APIs like C/C++/Rust functions. Instead, you MUST add the Python APIs that call them.
    - IMPORTANT: Just because a commit does not directly mention or update python APIs, does not mean changes to internal code do not affect any python APIs.
    - So, if any backend (e.g., C/C++/Rust) code affects any Python API or bindings, YOU MUST include that Python API in the list.
    - Especially in the case of repos with interfaces via python bindings (e.g., huggingface/tokenizers), find and include the python APIs affected by the commit.
6. Finally, if all else, and the commit does not affect any APIs, write "None".
"""

PERF_IDENTIFY_API_TASK = """Analyze the commit using natural language reasoning enclosed in [REASON] [/REASON] tags.
Then list the affected APIs (max 5 comma separated) enclosed in [APIS] [/APIS] tags.
Remember to close all tags properly.

Commit Information:
{diff_text}

Commit Message:
{message}
"""
