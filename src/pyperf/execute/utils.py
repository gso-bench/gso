import json

from rich.text import Text
from rich.console import Console
from rich.panel import Panel


console = Console()


def get_generated_tests(fut):
    header = Text("Equivalence Test", style="bold magenta")
    console.print(Panel(header, expand=False))
    console.print(fut.tests["test_0"])
    return json.dumps({"generated_tests": fut.tests})


def print_header(func_meth):
    print("\n\n")
    print("=" * 234)
    print("\n\n")
    header = Text("Function", style="bold yellow")
    console.print(Panel(header, expand=False))
    console.print(func_meth.code)
    console.print()
