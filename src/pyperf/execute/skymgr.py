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
            repo_url=problem.repo_url,
            repo_name=problem.repo_name,
            before_commit=problem.before_commit,
            after_commit=problem.after_commit,
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
    def launch_task(task_yaml, workspace, cluster="sky-pyperf"):
        # TODO: use --down to autotear the cluster after task is done
        # TODO: use --detach-setup/--detach-run to run w/ interactive setup --> can get logs from sky logs
        subprocess.run(["sky", "launch", "-c", cluster, task_yaml], cwd=workspace)
        logger.info(f"Launched task: {task_yaml}")

    @staticmethod
    def exec_task(task_yaml, workspace, cluster="sky-pyperf"):
        subprocess.run(["sky", "exec", cluster, task_yaml], cwd=workspace)
        logger.info(f"Execed task: {task_yaml}")

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
