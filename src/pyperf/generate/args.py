from r2e.llms.llm_args import LLMArgs
from pydantic import Field


class PerfExpGenArgs(LLMArgs):
    yaml_path: str = Field(..., description="Path to the experiment YAML file.")
    quickcheck: bool = Field(False, description="Run quickcheck after generating test.")
    model_name: str = Field("gpt-4o", description="Model name.")
    cache_batch_size: int = Field(100, description="Batch size for caching.")
    max_year: int = Field(2016, description="Maximum year for commits.")

    @classmethod
    def parse(cls, *args, **kwargs):
        if args and not kwargs.get("yaml_path"):
            kwargs["yaml_path"] = args[0]
        return cls(**kwargs)
