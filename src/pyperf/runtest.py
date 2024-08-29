import os
import ast
import json
from rich.text import Text
from rich.progress import Progress
from rich.panel import Panel
from rich.console import Console
import time

from utils import *

dict_to_str = lambda x: json.dumps(x, indent=4)

repo_data = {
    "repo_id": "jax",
    "repo_path": "/home/manishs/buckets/local_repoeval_bucket/repos/jax",
}

function_data = {
    "funclass_names": ["async_serialize"],
    "file_path": "jax/experimental/array_serialization/serialization.py",
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


# Apply the patch
os.system("sh logs/google___jax_22114/apply_patch.sh")


####### Execute the performance test (AFTER PATCH) #######

service.setup_codegen_mode()
out = service.execute(perf_test)

console.print()
console.print(Text("After Optimization", style="bold green"))
print("STDERR:\n", out["error"])
print("STDOUT:\n", out["output"])
