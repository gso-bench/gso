import json
import argparse
from pathlib import Path
from collections import defaultdict

from gso.data import PerformanceCommit, PerfAnalysis, APICommitMap
from gso.constants import *


class APIAnalyzer:
    def __init__(self):
        self.api_to_commits: dict[str, list[PerformanceCommit]] = defaultdict(list)

    def load_analysis(self, input_file: Path) -> PerfAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return PerfAnalysis(**data)

    def create_api_to_commits_map(self, analysis: PerfAnalysis) -> APICommitMap:
        self.commit_analysis = analysis
        for commit in analysis.performance_commits:
            for api in commit.apis:
                if api == "None":
                    continue
                self.api_to_commits[api].append(commit)

        # Sort by number of commits
        self.api_to_commits = dict(
            sorted(
                self.api_to_commits.items(), key=lambda item: len(item[1]), reverse=True
            )
        )

        return APICommitMap(
            repo_url=analysis.repo_url,
            repo_owner=analysis.repo_owner,
            repo_name=analysis.repo_name,
            api_to_commits=self.api_to_commits,
        )

    def get_commits_for_api(self, api: str) -> list[PerformanceCommit]:
        return self.api_to_commits.get(api, [])

    def api_commit_map(self) -> dict[str, list[dict]]:
        sorted_apis = sorted(
            self.api_to_commits.items(), key=lambda item: len(item[1]), reverse=True
        )

        summary = defaultdict(list)
        for api, commits in sorted_apis:
            for c in commits:
                summary[api].append(
                    {
                        "commit_hash": c.commit_hash,
                        "subject": c.subject,
                        "date": c.date.date(),
                    }
                )
        return summary

    def print_api_summary(self) -> None:
        sorted_apis = self.api_commit_map()
        for api, commits in sorted_apis.items():
            print(f"API: {api}")
            print(f"Number of affecting commits: {len(commits)}")
            print("Affecting commits:")
            for commit in commits:
                print(f"  - {commit['commit_hash'][:8]}: {commit['subject']}")
            print()

    @staticmethod
    def save_map(map: APICommitMap, output_file: Path) -> None:
        with open(output_file, "w") as f:
            f.write(map.model_dump_json(indent=2))

    @staticmethod
    def load_map(input_file: Path) -> APICommitMap:
        with open(input_file, "r") as f:
            data = json.load(f)
        return APICommitMap(**data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify APIs affected by commits")
    parser.add_argument("repo_name", type=str, help="Name of the repository")
    args = parser.parse_args()

    repo_name = args.repo_name
    input_file = ANALYSIS_COMMITS_DIR / f"{repo_name}_commits.json"

    analyzer = APIAnalyzer()
    commit_analysis = analyzer.load_analysis(input_file)
    ac_map = analyzer.create_api_to_commits_map(commit_analysis)

    # save the map to a file
    output_file = ANALYSIS_APIS_DIR / f"{repo_name}_ac_map.json"
    ANALYSIS_APIS_DIR.mkdir(parents=True, exist_ok=True)
    analyzer.save_map(ac_map, output_file)

    # Example usage
    # analyzer.print_api_summary()
