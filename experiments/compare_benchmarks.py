import pandas as pd
from datetime import datetime
from pathlib import Path
from pyperf.data.commit import ParsedCommit
from pyperf.collect.analysis.parser import CommitParser
import subprocess


def get_num_patch_lines(patch: str) -> int:
    """Count the number of edited lines in non-test files from a patch.
    
    Args:
        patch: A string containing the patch content
        
    Returns:
        int: The total number of edited lines in non-test files
    """
        
    try:    
        # Parse the commit
        parser = CommitParser()
        commit = parser.parse_commit(
            old_commit_hash="0000000",  # Dummy hash
            new_commit_hash="0000001",  # Dummy hash
            diff_text=patch,
            commit_message="",  # Empty message
            commit_date=datetime.now(),
            repo_location=None
        )
        
        return commit.num_non_test_edited_lines
    except Exception as e:
        print(f"Error parsing commit: {e}")
        raise e
        # Fallback to simple line counting if parsing fails
        # lines = patch.split('\n')
        # return sum(1 for line in lines if line.startswith('+') or line.startswith('-'))


def get_file_names(patch: str) -> list[str]:
    """Get the list of file names modified in a patch.

    Args:
        patch: A string containing the patch content
        
    Returns:
        list[str]: A list of file names modified in the patch
    """
    try:
        # Parse the commit
        parser = CommitParser()
        commit = parser.parse_commit(
            old_commit_hash="0000000",  # Dummy hash
            new_commit_hash="0000001",  # Dummy hash
            diff_text=patch,
            commit_message="",  # Empty message
            commit_date=datetime.now(),
            repo_location=None
        )
        
        return [file_diff.path for file_diff in commit.file_diffs]
    except Exception as e:
        print(f"Error parsing commit: {e}")
        raise e
    
    

## swebench-multilingual
df1 = pd.read_parquet("hf://datasets/SWE-bench/SWE-bench_Multilingual/data/test-00000-of-00001.parquet")['patch']

## multi-swebench
df2 = pd.read_parquet("hf://datasets/SWE-bench/SWE-bench_Verified/data/test-00000-of-00001.parquet")['patch']


## multi-swebench-mini
df3 = pd.read_json("https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_mini/resolve/main/multi_swe_bench_mini.jsonl", lines=True)['fix_patch']


## pyperf
# TODO
# df4 = 

df1_lines = df1.apply(get_num_patch_lines)
df2_lines = df2.apply(get_num_patch_lines)
df3_lines = df3.apply(get_num_patch_lines)


print(df1_lines.describe())
print(df2_lines.describe())
print(df3_lines.describe())



## get list of file names modified in each commit

# df1_file_names = df1.apply(get_file_names)
# df2_file_names = df2.apply(get_file_names)
# # df3_file_names = df3.apply(get_file_names)

# # print(df3_file_names.describe())

# all_file_names = set(df1_file_names.explode().unique()) | set(df2_file_names.explode().unique())


# for file_name in all_file_names:
#     print(file_name)
