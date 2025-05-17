import time
import shutil
import argparse
import asyncio
from tqdm.asyncio import tqdm as atqdm
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import functools

from gso.collect.execute.skymgr import SkyManager
from gso.utils.io import load_problems, save_problems
from gso.data import Problem
from gso.constants import *

DEBUG_FLAG = False


@dataclass
class TaskState:
    problem: Problem
    workspace: str
    cluster: str
    run_index: int
    is_complete: bool = False
    results_collected: bool = False
    launching: bool = False
    cleaning: bool = False


class ExecutionManager:
    def __init__(
        self,
        exp_id: str,
        exp_dir: Path,
        problems: list[Problem],
        machines: int,
        runs: int,
        interactive: bool = False,
    ):
        self.exp_id = exp_id
        self.exp_dir = exp_dir
        self.problems = problems
        self.machines = machines
        self.runs = runs
        self.cluster_counter = 0
        self.tasks: dict[str, TaskState] = {}
        self.active_clusters: set[str] = set()
        self.completed_clusters: set[str] = set()
        self.interactive = interactive

        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=machines * 2)
        self.last_progress_print = 0

    def get_next_cluster_name(self) -> str:
        with self.lock:
            name = f"sky-gso-{self.exp_id}-{self.cluster_counter}"
            self.cluster_counter += 1
            return name

    def initialize_problems(self):
        """Set up tasks and workspaces for all problems"""
        for prob in self.problems:
            for run_idx in range(self.runs):
                cluster = self.get_next_cluster_name()
                try:
                    wspace = SkyManager.create_workspace(prob)
                except Exception as e:
                    continue

                self.tasks[cluster] = TaskState(
                    problem=prob, workspace=wspace, cluster=cluster, run_index=run_idx
                )

    async def launch_task_async(self, cluster: str, state: TaskState):
        """Asynchronously launch a task"""
        if state.launching:
            return

        state.launching = True
        try:
            # Run the launch task in a thread pool to not block
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                functools.partial(
                    SkyManager.launch_task,
                    f"{state.problem.pid}_task.yaml",
                    state.workspace,
                    cluster=cluster,
                    interactive=self.interactive,
                ),
            )

            self.active_clusters.add(cluster)
            print(
                f"Launched {state.problem.pid} (run {state.run_index + 1}/{self.runs}) on cluster {cluster}",
                flush=True,
            )
        except Exception as e:
            print(f"Error launching task on cluster {cluster}: {e}", flush=True)
        finally:
            state.launching = False

    async def cleanup_cluster_async(self, cluster: str, state: TaskState):
        """Asynchronously cleanup a cluster"""
        if state.cleaning:
            return

        state.cleaning = True
        try:
            # Run the cleanup in a thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                functools.partial(
                    SkyManager.cleanup_cluster, cluster, interactive=False
                ),
            )
            print(f"Cleaned up cluster {cluster}")
        except Exception as e:
            print(f"Error cleaning up cluster {cluster}: {e}")
        finally:
            state.cleaning = False

    async def launch_available_tasks(self):
        """Launch tasks that can be executed based on machine availability"""
        with self.lock:
            if len(self.active_clusters) >= self.machines:
                return

            available_slots = self.machines - len(self.active_clusters)
            pending_tasks = [
                (cluster, state)
                for cluster, state in self.tasks.items()
                if cluster not in self.active_clusters
                and cluster not in self.completed_clusters
                and not state.launching
            ]

            # Launch tasks concurrently
            launch_tasks = []
            for cluster, state in pending_tasks[:available_slots]:
                task = self.launch_task_async(cluster, state)
                launch_tasks.append(task)

            if launch_tasks:
                await asyncio.gather(*launch_tasks)

    async def check_completion(self):
        """Check for completed tasks and collect results"""
        cleanup_tasks = []
        completed = set()

        for cluster in self.active_clusters:
            state = self.tasks[cluster]
            if not state.is_complete:
                try:
                    is_complete = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        functools.partial(
                            SkyManager.is_complete, state.workspace, cluster
                        ),
                    )
                except Exception as e:
                    print(f"WARNING: error is_complete: {cluster} Assuming complete")
                    is_complete = True

                if is_complete:
                    state.is_complete = True
                    completed.add(cluster)

                    res_str, results = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        functools.partial(
                            SkyManager.get_results, state.workspace, cluster
                        ),
                    )

                    result_key = f"{cluster}_run{state.run_index}"
                    state.problem.add_results(key=result_key, results=results)
                    state.results_collected = True
                    print(
                        f"Collected results from {cluster} (run {state.run_index + 1}/{self.runs})",
                        flush=True,
                    )

                    cleanup_tasks.append(self.cleanup_cluster_async(cluster, state))

        # Update tracking sets
        with self.lock:
            self.active_clusters -= completed
            self.completed_clusters |= completed

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)

    async def run(self):
        """Main execution loop using async/await"""
        count = 0
        results_json = f"{self.exp_id}_results{'_DEBUG' if DEBUG_FLAG else ''}.json"
        try:
            while not self.all_tasks_complete():
                print("======== Phase: Launch =========", flush=True)
                await self.launch_available_tasks()
                print("---------------------------------\n\n", flush=True)

                print("======== Phase: Execution =========", flush=True)
                await self.check_completion()
                print(self.get_progress_summary(), flush=True)
                print("---------------------------------\n\n", flush=True)
                await asyncio.sleep(5)

                # Save on every 10 new problems that are completed
                if len(self.completed_clusters) - count >= 10:
                    save_problems(self.exp_dir / results_json, self.problems)
                    count = len(self.completed_clusters)
        finally:
            self.thread_pool.shutdown(wait=True)
            self.cleanup_all()
            save_problems(self.exp_dir / results_json, self.problems)

    def cleanup_all(self):
        for state in self.tasks.values():
            if state.workspace:
                SkyManager.cleanup_workspace(state.workspace)

    def all_tasks_complete(self) -> bool:
        return all(state.results_collected for state in self.tasks.values())

    def get_progress_summary(self) -> str:
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_clusters)
        active_tasks = len(self.active_clusters)
        pending_tasks = total_tasks - completed_tasks - active_tasks

        return (
            f"Progress: {completed_tasks}/{total_tasks} tasks completed | "
            f"Active: {active_tasks} | Pending: {pending_tasks}"
        )


async def async_main(
    exp_id: str,
    machines: int,
    runs: int,
    specific_api: str | None = None,
    exp_yaml: str | None = None,
    interactive: bool = False,
):
    exp_dir = EXPS_DIR / f"{exp_id}"
    all_problems = load_problems(exp_dir / f"{exp_id}_problems.json")

    if exp_yaml is not None:
        if os.path.exists(exp_yaml):
            with open(exp_yaml, "r") as f:
                exp_data = yaml.safe_load(f)
            for p in all_problems:
                p.target_commit = exp_data.get("target_commit", "main")
                p.install_commands = exp_data.get("install_commands", [])
        else:
            raise ValueError(f"Experiment YAML file provided but not found: {exp_yaml}")

    if specific_api:
        problems = [p for p in all_problems if p.api == specific_api]
        if not problems:
            raise ValueError(f"No problem found for API: {specific_api}")
    else:
        problems = all_problems

    manager = ExecutionManager(exp_id, exp_dir, problems, machines, runs, interactive)
    manager.initialize_problems()
    try:
        await manager.run()
    finally:
        SkyManager.cleanup_all_clusters()
        if not DEBUG_FLAG:
            save_problems(exp_dir / f"{exp_id}_results.json", problems)


def main(
    exp_id: str,
    machines: int,
    runs: int,
    specific_api: str | None = None,
    exp_yaml: str | None = None,
    interactive: bool = False,
):
    asyncio.run(async_main(exp_id, machines, runs, specific_api, exp_yaml, interactive))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute tasks with SkyManager")
    parser.add_argument("-e", "--exp_id", type=str, help="Experiment ID", required=True)
    parser.add_argument(
        "-a", "--api", type=str, help="Specific API", required=False, default=None
    )
    parser.add_argument(
        "-yp",
        "--exp_yaml",
        type=str,
        help="Path to experiment YAML",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-m", "--machines", type=int, default=2, help="Max concurrent machines"
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=1, help="Number of times to run each problem"
    )
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()

    main(
        args.exp_id, args.machines, args.runs, args.api, args.exp_yaml, args.interactive
    )
