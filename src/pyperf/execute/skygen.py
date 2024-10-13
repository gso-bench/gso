import json
import os
import shutil
import tempfile
from pathlib import Path
from string import Template


class SkyGen:
    """Generate a skypilot YAML file for a given problem"""

    @staticmethod
    def load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_yaml_template(template_path):
        with open(template_path, "r") as f:
            return Template(f.read())

    @staticmethod
    def create_yaml_content(template, data):
        repo_name = Path(data["repo_url"]).stem
        setup_commands = "\n  ".join(data["setup_commands"])
        install_commands = " && ".join(data["install_commands"])

        result = template.safe_substitute(
            name=data["name"],
            cloud=data.get("cloud", "gcp"),
            region=data.get("region", "us-central1"),
            instance_type=data.get("instance_type", "n2-standard-16"),
            setup_commands=setup_commands,
            repo_url=data["repo_url"],
            repo_name=repo_name,
            before_commit=data["before_commit"],
            after_commit=data["after_commit"],
            install_commands_before=install_commands,
            install_commands_after=install_commands,
        )
        return result

    @staticmethod
    def create_temp_directory_with_files(data, yaml_template_path="template.yaml"):
        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            # Create and write the YAML file
            yaml_content = SkyGen.create_yaml_content(yaml_template, data)
            yaml_path = os.path.join(temp_dir, f"{data['name']}_task.yaml")
            with open(yaml_path, "w") as yaml_file:
                yaml_file.write(yaml_content)

            # Create and write the test.py file
            test_script_path = os.path.join(temp_dir, "test.py")
            with open(test_script_path, "w") as test_file:
                test_file.write(data["test_code"])

        return temp_dir


# Example usage
if __name__ == "__main__":
    json_file_path = "input.json"
    SkyGen.process_json_input(json_file_path)

    data = SkyGen.load_json(json_file_path)
    data = data[0]
    yaml_template = SkyGen.load_yaml_template(yaml_template_path)
    temp_dir = SkyGen.create_temp_directory_with_files(data, yaml_template)

    shutil.rmtree(temp_dir)
