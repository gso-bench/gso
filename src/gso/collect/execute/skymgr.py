import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from string import Template

from gso.collect.execute.helpers import zip_results, add_tokens_to_installs
from gso.harness.environment.patches import apply_patches_to_tests
from gso.constants import *
from gso.logger import logger


class SkyManager:
    """Generate and manage skypilot tasks for perf testing"""

    @staticmethod
    def load_template(template_path):
        with open(template_path, "r") as f:
            return Template(f.read())

    @staticmethod
    def build_templates(temp_dir, task, phase1, phase2, problem):
        setup_commands = "\n  ".join(problem.setup_commands)
        install_commands = add_tokens_to_installs(problem.install_commands)
        install_commands = "\n        ".join(install_commands)
        candidates = " ".join(t.quick_hash for t in problem.tests)

        task = task.safe_substitute(
            id=problem.pid.replace("__", ""),
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
            run_target_tests="false",
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
                test_samples = commit_tests.samples

                # optional: uncomment to not apply patched requests.get function
                test_samples = apply_patches_to_tests("requests", test_samples)

                # write sampled tests for each commit
                for i, sample in enumerate(test_samples):
                    with open(commit_dir / f"test_{i}.py", "w") as test_file:
                        test_file.write(sample)

            logger.info(f"Created workspace: {temp_dir}")

        return temp_dir

    @staticmethod
    def launch_task(task_yaml, workspace, cluster="sky-gso", interactive=False):
        cmd = ["sky", "launch", "-c", cluster, task_yaml]
        if not interactive:
            subprocess.run(
                cmd + ["--detach-setup", "--detach-run", "--yes"],
                cwd=workspace,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        else:
            subprocess.run(cmd, cwd=workspace, text=True)
        logger.info(f"Launched {task_yaml} on cluster: {cluster} from {workspace}")

    @staticmethod
    def is_complete(workspace, cluster="sky-gso"):
        result = subprocess.run(
            ["sky", "logs", "--status", cluster], cwd=workspace, capture_output=True
        )
        stdout, stderr = result.stdout.decode("utf-8"), result.stderr.decode("utf-8")
        if stderr:
            logger.error(stderr)
            raise Exception(stderr)

        elif "FAILED" in result.stdout.decode("utf-8"):
            logger.warning(f"is_complete: cluster {cluster} failed")
            return True

        return "SUCCEEDED" in result.stdout.decode("utf-8")

    @staticmethod
    def get_results(workspace, cluster="sky-gso"):
        subprocess.run(
            ["rsync", "-Pavz", f"{cluster}:~/sky_workdir/results/*", "./results/"],
            cwd=workspace,
        )

        if not (workspace / "results").exists():
            return f"Cluster: {cluster}: no results!", []

        file_groups = zip_results(workspace / "results")
        results = []

        for identifier, files in file_groups.items():
            commit, test_file = identifier.split("_", 1)
            base_file = files.get("base")
            commit_file = files.get("commit")
            target_file = files.get("target")
            meta_file = files.get("meta")

            if base_file and meta_file:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    meta["test_id"] = int(meta["test_file"][:-3].split("_")[-1])

                with open(base_file, "r") as f:
                    meta["base_result"] = f.read()

                if target_file:
                    with open(target_file, "r") as f:
                        meta["target_result"] = f.read()

                if commit_file:
                    with open(commit_file, "r") as f:
                        meta["commit_result"] = f.read()

                results.append(meta)

        return f"Cluster: {cluster}: results returned!", results

    @staticmethod
    def cleanup_workspace(workspace):
        shutil.rmtree(workspace)
        logger.info(f"Deleted workspace: {workspace}")

    @staticmethod
    def cleanup_cluster(cluster, interactive=False):
        cmd = ["sky", "down", cluster]
        if not interactive:
            cmd.append("--yes")
        subprocess.run(cmd)
        logger.info(f"Deleted cluster: {cluster}")

    @staticmethod
    def cleanup_all_clusters(interactive=False):
        cmd = ["sky", "down", "-a"]
        if not interactive:
            cmd.append("--yes")
        subprocess.run(cmd)
        logger.info(f"Deleted all clusters")
