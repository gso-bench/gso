from r2e.llms.llm_args import LLMArgs
from pydantic import Field


class PerfExpGenArgs(LLMArgs):
    yaml_path: str = Field(..., description="Path to the experiment YAML file.")
    model_name: str = Field("gpt-4o", description="Model name.")
    cache_batch_size: int = Field(100, description="Batch size for caching.")
    max_year: int = Field(2016, description="Maximum year for commits.")
    min_loc: int = Field(0, description="Minimum LOC for a commit.")
    multiprocess: int = Field(30, description="Num parallel processes for generation.")
    n: int = Field(5, description="Num samples per generation.")
    api: str = Field(None, description="API to generate tests for.")

    @classmethod
    def parse(cls, *args, **kwargs):
        if args and not kwargs.get("yaml_path"):
            kwargs["yaml_path"] = args[0]
        return cls(**kwargs)


class OversampleArgs(LLMArgs):
    exp_id: str = Field(None, description="Experiment ID.")
    model_name: str = Field("gpt-4o", description="Model name.")
    cache_batch_size: int = Field(100, description="Batch size for caching.")
    max_year: int = Field(2016, description="Maximum year for commits.")
    min_loc: int = Field(0, description="Minimum LOC for a commit.")
    multiprocess: int = Field(30, description="Num parallel processes for generation.")
    n: int = Field(5, description="Num samples per generation.")
    api: str = Field(None, description="API to generate tests for.")

    @classmethod
    def parse(cls, *args, **kwargs):
        if args and not kwargs.get("yaml_path"):
            kwargs["yaml_path"] = args[0]
        return cls(**kwargs)
