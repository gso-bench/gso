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
    def install_repo_script(self):
        env_name = "testbed"
        repo_directory = f"/{env_name}"
        repo_setup = [
            f"git clone -o origin https://github.com/{self.repo} {repo_directory}",
            f"chmod -R 777 {repo_directory}",  # nonroot user can run tests
            f"cd {repo_directory}",
            f"git reset --hard {self.base_commit}",
            # Remove remote so agent can't see newer commits
            f"git remote remove origin",
        ]

        return (
            "\n".join(
                ["#!/bin/bash", "set -euxo pipefail"]
                + repo_setup
                + self.install_commands
            )
            + "\n"
        )

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"pyperf.eval.{self.instance_id}"
        return f"pyperf.eval.{self.instance_id.lower()}.{run_id}"
