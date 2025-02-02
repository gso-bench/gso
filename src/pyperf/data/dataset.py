from dataclasses import dataclass


@dataclass
class PyPerfInstance:
    instance_id: str
    repo: str
    base_commit: str
    api: str
    test_script: str
    hints_text: str
    setup_commands: list[str]
    install_commands: list[str]
    created_at: str
    arch: str = "x86_64"
    instance_image_tag: str = "latest"

    @property
    def instance_image_key(self):
        key = f"pyperf.eval.{self.arch}.{self.instance_id.lower()}:{self.instance_image_tag}"
        return key

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")
