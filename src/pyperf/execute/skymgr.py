import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from string import Template

from pyperf.constants import *
from pyperf.logger import logger


class SkyManager:
    """Generate and manage skypilot tasks for perf testing"""

    @staticmethod
    def load_template(template_path):
        with open(template_path, "r") as f:
            return Template(f.read())

    @staticmethod
    def build_templates(temp_dir, task, phase1, phase2, problem):
        setup_commands = "\n  ".join(problem.setup_commands)
        install_commands = "\n        ".join(problem.install_commands)
        candidates = " ".join(t.quick_hash for t in problem.tests)

        task = task.safe_substitute(
            id=problem.pid,
            cloud=problem.cloud,
            region=problem.region,
            instance_type=problem.instance_type,
            setup_commands=setup_commands,
            repo_url=problem.repo.repo_url,
            repo_name=problem.repo.repo_name,
            candidates=candidates,
        )

        phase1 = phase1.safe_substitute(
            repo_name=problem.repo.repo_name, install_commands=install_commands
        )

        phase2 = phase2.safe_substitute(
            repo_name=problem.repo.repo_name,
            install_commands=install_commands,
            target_commit=problem.target_commit,
            file_before="results_a.txt",
            file_after="results_b.txt",
        )

        with open(temp_dir / f"{problem.pid}_task.yaml", "w") as yaml_file:
            yaml_file.write(task)

        with open(temp_dir / f"phase1.sh", "w") as phase1_file:
            phase1_file.write(phase1)

        with open(temp_dir / f"phase2.sh", "w") as phase2_file:
            phase2_file.write(phase2)

    @staticmethod
    def create_workspace(problem) -> Path:
        task = SkyManager.load_template(SKYGEN_TEMPLATE)
        phase1 = SkyManager.load_template(PHASE1_TEMPLATE)
        phase2 = SkyManager.load_template(PHASE2_TEMPLATE)

        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            temp_dir = Path(temp_dir)

            # Create and write tamplates (task, phase1, phase2) to workspace
            SkyManager.build_templates(temp_dir, task, phase1, phase2, problem)

            # each candidate commit is a subdirectory in the workspace
            for commit_tests in problem.tests:
                commit_dir = temp_dir / commit_tests.quick_hash
                commit_dir.mkdir(parents=True, exist_ok=True)

                # write sampled tests for each commit
                for i, sample in enumerate(commit_tests.samples):
                    with open(commit_dir / f"test_{i}.py", "w") as test_file:
                        test_file.write(sample)

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
        stdout, stderr = result.stdout.decode("utf-8"), result.stderr.decode("utf-8")
        if stderr:
            logger.error(stderr)
            raise Exception(stderr)

        return "SUCCEEDED" in result.stdout.decode("utf-8")

    @staticmethod
    def get_results(workspace, cluster="sky-pyperf"):
        subprocess.run(
            ["rsync", "-Pavz", f"{cluster}:~/sky_workdir/results_*", "."], cwd=workspace
        )

        subprocess.run(
            ["rsync", "-Pavz", f"{cluster}:~/sky_workdir/working_*", "."], cwd=workspace
        )

        results_a, results_b, metadata, working_test = None, None, None, None
        try:
            with open(workspace / "results_a.txt", "r") as f:
                results_a = f.read()

            with open(workspace / "results_b.txt", "r") as f:
                results_b = f.read()

            with open(workspace / "working_pair.txt", "r") as f:
                commit, test_file = f.read().split(" ")
                metadata = {"commit": commit, "test_file": test_file.split("/")[-1]}

            with open(workspace / "working_test.py", "r") as f:
                working_test = f.read()
        except FileNotFoundError as e:
            logger.error(f"{cluster}: {str(e)}")

        result_str = f"Cluster: {cluster}\nMetadata:{metadata}\n\nA:\n{results_a}\nB:\n{results_b}"
        result = {
            "base": results_a,
            "target": results_b,
            "metadata": metadata,
            "test": working_test,
        }
        logger.info(f"{result_str}")
        return result_str, result

    @staticmethod
    def cleanup_workspace(workspace):
        shutil.rmtree(workspace)
        logger.info(f"Deleted workspace: {workspace}")

    @staticmethod
    def cleanup_cluster(cluster, interactive=False):
        subprocess.run(
            ["sky", "down", cluster], input="Y\n" if not interactive else None
        )
        logger.info(f"Deleted cluster: {cluster}")

    @staticmethod
    def cleanup_all_clusters(interactive=False):
        subprocess.run(
            ["sky", "down", "-a"], input="Y\n" if not interactive else None, text=True
        )
        logger.info(f"Deleted all clusters")
