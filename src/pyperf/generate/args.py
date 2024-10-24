from r2e.llms.llm_args import LLMArgs
from pydantic import Field


class PerfExpGenArgs(LLMArgs):
    yaml_path: str = Field(..., description="Path to the experiment YAML file.")
    model_name: str = Field("gpt-4o", description="Model name.")
