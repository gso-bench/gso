import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from string import Template

from pyperf.logger import logger


class SkyManager:
    """Generate and manage skypilot tasks for perf testing"""

    @staticmethod
    def load_template(template_path):
        with open(template_path, "r") as f:
            return Template(f.read())

    @staticmethod
    def create_yaml_content(template, problem):
        setup_commands = "\n  ".join(problem.setup_commands)
        install_commands = "\n  ".join(problem.install_commands)

        result = template.safe_substitute(
            id=problem.pid,
            cloud=problem.cloud,
            region=problem.region,
            instance_type=problem.instance_type,
            setup_commands=setup_commands,
            repo_url=problem.repo.repo_url,
            repo_name=problem.repo.repo_name,
            base_commit=problem.base_commit,
            target_commit=problem.target_commit,
            install_commands_before=install_commands,
            install_commands_after=install_commands,
            file_before="results_a.txt",
            file_after="results_b.txt",
        )
        return result

    @staticmethod
    def create_workspace(problem, yaml_template):
        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            # Create and write the YAML file
            yaml_content = SkyManager.create_yaml_content(yaml_template, problem)
            yaml_path = os.path.join(temp_dir, f"{problem.pid}_task.yaml")
            with open(yaml_path, "w") as yaml_file:
                yaml_file.write(yaml_content)

            # Create and write the test.py file
            test_script_path = os.path.join(temp_dir, "test.py")
            with open(test_script_path, "w") as test_file:
                test_file.write(problem.test)

            logger.info(f"Created workspace: {temp_dir}")

        return temp_dir

    @staticmethod
    def launch_task(task_yaml, workspace, cluster="sky-pyperf", interactive=False):
        cmd = ["sky", "launch", "-c", cluster, task_yaml]
        if not interactive:
            cmd.append("--detach-setup")
            cmd.append("--detach-run")

        subprocess.run(
            cmd, cwd=workspace, input="Y\n" if not interactive else None, text=True
        )
        logger.info(f"Launched task: {task_yaml}")

    @staticmethod
    def exec_task(task_yaml, workspace, cluster="sky-pyperf"):
        subprocess.run(["sky", "exec", cluster, task_yaml], cwd=workspace)
        logger.info(f"Execed task: {task_yaml}")

    @staticmethod
    def is_complete(workspace, cluster="sky-pyperf"):
        result = subprocess.run(
            ["sky", "logs", "--status", cluster], cwd=workspace, capture_output=True
        )
        return "SUCCEEDED" in result.stdout.decode("utf-8")

    @staticmethod
    def get_results(workspace, cluster="sky-pyperf"):
        subprocess.run(
            ["rsync", "-Pavz", f"{cluster}:~/sky_workdir/results_*", "."], cwd=workspace
        )

        try:
            with open(os.path.join(workspace, "results_a.txt"), "r") as f:
                results_a = f.read()
        except FileNotFoundError:
            logger.error(f"results_a.txt not found in {cluster}")
            return None

        try:
            with open(os.path.join(workspace, "results_b.txt"), "r") as f:
                results_b = f.read()
        except FileNotFoundError:
            logger.error(f"results_b.txt not found in {cluster}")
            return None

        # NOTE(@manish): cleaning up so on subsequent launches,
        # we don't move previous results to the workspace; simplifies result logging
        os.remove(os.path.join(workspace, "results_a.txt"))
        os.remove(os.path.join(workspace, "results_b.txt"))

        result = f"Cluster: {cluster}\n\nA:\n{results_a}\nB:\n{results_b}"
        logger.info(f"{result}")
        return result

    @staticmethod
    def cleanup_workspace(workspace):
        shutil.rmtree(workspace)
        logger.info(f"Deleted workspace: {workspace}")

    @staticmethod
    def cleanup_cluster(cluster):
        subprocess.run(["sky", "down", cluster])
        logger.info(f"Deleted cluster: {cluster}")

    @staticmethod
    def cleanup_all_clusters():
        subprocess.run(["sky", "down", "-a"])
        logger.info(f"Deleted all clusters")
