from pydantic import BaseModel, Field, HttpUrl


class Problem(BaseModel):
    pid: str = Field(default="test", description="ID of the problem")

    # vm info
    cloud: str = Field(default="gcp", description="Cloud provider")
    region: str = Field(default="us-central1", description="Cloud region")
    instance_type: str = Field(default="n2-standard-16", description="Instance type")

    # repo info # TODO: eventually replace with the repo model
    repo_url: HttpUrl = Field(..., description="Repository URL")
    repo_name: str = Field(..., description="Repository name")

    # commit info
    before_commit: str = Field(..., description="Commit hash for before test")
    after_commit: str = Field(..., description="Commit hash for after test")

    # commands
    setup_commands: list[str] = Field(
        default_factory=list, description="Setup commands to run"
    )
    install_commands: list[str] = Field(
        default_factory=list, description="Install commands to run"
    )

    # test code #TODO: eventually replace with the test model
    test: str = Field(..., description="Test code to run")
