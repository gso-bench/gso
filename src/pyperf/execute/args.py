from pydantic import BaseModel, Field


class PerfTestRunArgs(BaseModel):
    multiprocess: int = Field(
        20,
        description="The number of processes to use for executing the functions and methods",
    )

    batch_size: int = Field(
        100,
        description="The number of functions to run before writing the output to the file",
    )

    port: int = Field(3006, description="The port to use for the execution service")

    timeout_per_task: int = Field(
        180, description="The timeout for the execution service in seconds"
    )

    in_file: str = Field(
        None,
        description="The input file for the test runner. Usually {exp_id}_generate.json",
    )

    exp_id: str = Field(
        "temp",
        description="Experiment ID used for prefixing the generated test execution file.",
    )
