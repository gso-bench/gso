import json
import os
import shutil
import tempfile
from pathlib import Path
from string import Template

from pyperf.utils.io import load_problems


class SkyGen:
    """Generate a skypilot YAML file for a given problem"""

    @staticmethod
    def load_yaml_template(template_path):
        with open(template_path, "r") as f:
            return Template(f.read())

    @staticmethod
    def create_yaml_content(template, problem):
        setup_commands = "\n  ".join(problem.setup_commands)
        install_commands = " && ".join(problem.install_commands)

        result = template.safe_substitute(
            id=problem.pid,
            cloud=problem.cloud,
            region=problem.region,
            instance_type=problem.instance_type,
            setup_commands=setup_commands,
            repo_url=problem.repo_url,
            repo_name=problem.repo_name,
            before_commit=problem.before_commit,
            after_commit=problem.after_commit,
            install_commands_before=install_commands,
            install_commands_after=install_commands,
        )
        return result

    @staticmethod
    def create_temp_directory_with_files(problem, yaml_template):
        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            # Create and write the YAML file
            yaml_content = SkyGen.create_yaml_content(yaml_template, problem)
            yaml_path = os.path.join(temp_dir, f"{problem.pid}_task.yaml")
            with open(yaml_path, "w") as yaml_file:
                yaml_file.write(yaml_content)

            # Create and write the test.py file
            test_script_path = os.path.join(temp_dir, "test.py")
            with open(test_script_path, "w") as test_file:
                test_file.write(problem.test)

        return temp_dir


# Example usage
if __name__ == "__main__":
    data = load_problems("input.json")
    problem = data[0]

    yaml_template = SkyGen.load_yaml_template("template.yaml")
    temp_dir = SkyGen.create_temp_directory_with_files(problem, yaml_template)

    shutil.rmtree(temp_dir)
