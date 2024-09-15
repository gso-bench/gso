import os
import ast
import json
from rich.text import Text
from rich.progress import Progress
from rich.panel import Panel
from rich.console import Console
import time

from utils import *

PR_NUMBER = 22375
TEST_PR_PATCH = False

dict_to_str = lambda x: json.dumps(x, indent=4)

repo_data = {
    "repo_id": None,
    "repo_path": "/home/manishs/buckets/local_repoeval_bucket/repos/jax",
}

function_data = {
    "funclass_names": ["_in1d"],
    "file_path": "jax/_src/numpy/setops.py",
}

tests = {"generated_tests": {}}

with open("sample_test.log", "r") as f:
    perf_test = f.read()

####### Setup the service #######

console = Console()
service = get_service()
service.setup_repo(dict_to_str(repo_data))
service.setup_function(dict_to_str(function_data))
service.setup_test(dict_to_str(tests))

out = service.init()
print("STDERR:\n", out["error"])
print("STDOUT:\n", out["output"])

####### Execute the performance test (BEFORE PATCH) #######

service.setup_codegen_mode()
out = service.execute(perf_test)

console.print()
console.print(Text("Before Optimization", style="bold green"))
print("STDERR:\n", out["error"])
print("STDOUT:\n", out["output"])


####### Execute the performance test (AFTER PATCH) #######

if TEST_PR_PATCH:
    os.system(f"sh logs/google___jax_{PR_NUMBER}/apply_patch.sh")
    service.setup_codegen_mode()
    out = service.execute(perf_test)

    console.print()
    console.print(Text("After Optimization", style="bold green"))
    print("STDERR:\n", out["error"])
    print("STDOUT:\n", out["output"])
