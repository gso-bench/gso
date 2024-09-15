from r2e.llms.llm_args import LLMArgs
from pydantic import Field


class PerfTestGenArgs(LLMArgs):
    context_type: str = Field(
        "sliced",
        description="The context type to use for the language model",
    )

    max_context_size: int = Field(
        6000,
        description="The maximum context size",
    )

    in_file: str = Field(
        None,
        description="The input file for the test generator",
    )

    exp_id: str = Field(
        "temp",
        description="Experiment ID used for prefixing the generated tests file.",
    )
