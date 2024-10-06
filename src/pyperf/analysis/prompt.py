PERF_ANALYSIS_MESSAGE = """You are an expert python programmer who is annotating data for training and testing code language models.

You will be given a GitHub commit patch content. Your goal is to identify whether the commit is performance or optimization related. The commit should satisfy the following conditions:
1. The commit should modify at least one non-test file.
2. The commit should modify source code in a non-trivial manner and not just fix comments or documentation.
3. The changes need not directly mention performance or optimization in the commit message but should be related to performance optimization.
4. The changes should not be related to bug fixes or simple refactoring.
5. The changes should preferably affect the performance of high-level or top-level APIs in the repo. This can be directly or indirectly via changes to internal APIs.

Analyze the commit using natural language reasoning enclosed in [REASON] [/REASON] tags.
Then write YES or NO based on the conditions mentioned above enclosed in [ANSWER] [/ANSWER] tags.
Remember to close all tags properly.

Commit Information:
{diff_text}

Commit Message:
{message}
"""
