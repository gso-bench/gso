from r2e.llms.llm_args import LLMArgs
from pydantic import Field


class PerfExpGenArgs(LLMArgs):
    exp_id: str = Field("exp", description="Experiment ID.")
    model_name: str = Field("gpt-4o", description="Model name.")
