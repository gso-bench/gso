import tiktoken
import subprocess
from pathlib import Path

tokenizer = tiktoken.encoding_for_model("gpt-4")


def count_tokens(context: str):
    return len(tokenizer.encode(context, disallowed_special=()))


def run_git_command(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(
        cmd, cwd=cwd, universal_newlines=True, errors="replace"
    ).strip()


def prompt_yes_no(message):
    response = input(f"{message} (y/n): ")
    return response.lower() in ["y", "yes"]
