from r2e.llms.llm_args import LLMArgs
from pydantic import Field


class PerfExpGenArgs(LLMArgs):
    exp_id: str = Field(
        "temp",
        description="Experiment ID used for prefixing the generated tests file.",
    )



