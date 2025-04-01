import fire
import re
from datasets import load_dataset, Dataset

from r2e.llms.completions import LLMCompletions
from r2e.llms.llm_args import LLMArgs


SHORT_PLAN_PROMPT = """You are a performance testing expert. You will generate a description of a performance improving commit for a Python repository.
The description MUST be a single sentence with sufficient detail, and sound like a plan.

## Repo: {repo}
## Commit Message: {commit_message}

## Commit Diff:
{commit_diff}

Guidelines:
- Carefully read and try to understand the commit and interpret the changes made in the commit. Then, write a plan that describes the high-level idea of the optimization.
- The description should detail the high-level ideas of the optimization
- The description should be concise and clear
- The description should be specific to the commit and can describe the identified bottleneck if any
- Distill the ideas into a single sentence when there are multiple ideas being used
- Completely ignore changes to comments, documentation, testing, formatting, CI, etc.
- Only focus on core optimization ideas

Respond in the following format:
[[PLAN]]
To improve performance, we can <Your plan here>
[[PLAN]]
"""


DETAILED_PLAN_PROMPT = """You are a performance testing expert. You will generate a description of a performance improving commit for a Python repository.
The description MUST be a maximum of 5 point detailed description with sufficient detail, and sound like a plan.

## Repo: {repo}
## Commit Message: {commit_message}

## Commit Diff:
{commit_diff}

Guidelines:
- Carefully read and try to understand the commit and interpret the changes made in the commit. Then, write a plan that describes the high-level idea of the optimization.
- The description should detail the high-level ideas of the optimization
- The description should be concise and clear
- The description should be specific to the commit and can describe the identified bottleneck if any
- Distill the ideas into a maximum of 5 points when there are multiple ideas being used
- Completely ignore changes to comments, documentation, testing, formatting, CI, etc.
- Only focus on core optimization ideas

Respond in the following format:
[[PLAN]]
To improve performance, we can <Your plan here>
[[PLAN]]
"""


def get_generated_plans(outputs):
    results = []
    for output in outputs:
        plans = []
        for sample in output:
            plan = re.search(r"\[\[PLAN\]\](.*)\[\[PLAN\]\]", sample, re.DOTALL)
            plans.append(plan.group(1).strip())
        results.append(plans)
    return results


def prompt_o3_mini(instance, mode="short"):
    commit_diff = instance["gt_diff"]
    commit_message = instance["gt_commit_message"]
    repo = instance["repo"]

    if mode == "short":
        prompt = SHORT_PLAN_PROMPT.format(
            repo=repo, commit_message=commit_message, commit_diff=commit_diff
        )
    else:
        prompt = DETAILED_PLAN_PROMPT.format(
            repo=repo, commit_message=commit_message, commit_diff=commit_diff
        )

    return prompt


if __name__ == "__main__":
    args = fire.Fire(LLMArgs)
    repo = "numpy/numpy"
    dataset = load_dataset("manishs/pyperf-extended", split="test")
    dataset = dataset.filter(lambda x: repo == x["repo"])
    print(f"Dataset size: {len(dataset)}")

    # SHORT PLANS
    short_payloads = []
    for row in dataset:
        prompt = prompt_o3_mini(row, mode="short")
        short_payloads.append([{"role": "user", "content": prompt}])

    short_payloads = [p for p in short_payloads for _ in range(args.n)]
    outputs = LLMCompletions.get_llm_completions(args, short_payloads)
    outputs = [item for sublist in outputs for item in sublist]
    grouped_outputs = []
    for i in range(0, len(outputs), args.n):
        grouped_outputs.append(outputs[i : i + args.n])
    short_plans = get_generated_plans(grouped_outputs)
    # print(short_plans)

    # DETAILED PLANS
    detailed_payloads = []
    for row in dataset:
        prompt = prompt_o3_mini(row, mode="detailed")
        detailed_payloads.append([{"role": "user", "content": prompt}])

    detailed_payloads = [p for p in detailed_payloads for _ in range(args.n)]
    outputs = LLMCompletions.get_llm_completions(args, detailed_payloads)
    detailed_outputs = [item for sublist in outputs for item in sublist]
    grouped_outputs = []
    for i in range(0, len(detailed_outputs), args.n):
        grouped_outputs.append(detailed_outputs[i : i + args.n])
    detailed_plans = get_generated_plans(grouped_outputs)
    # print(detailed_plans)

    # Convert the dataset to a dictionary
    dataset_dict = dataset.to_dict()
    dataset_dict["short_plans"] = short_plans
    dataset_dict["detailed_plans"] = detailed_plans
    new_dataset = Dataset.from_dict(dataset_dict)
    new_dataset.push_to_hub("manishs/pyperf-planned", split="test", private=True)
