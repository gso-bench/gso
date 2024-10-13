import json
from pathlib import Path
from collections import defaultdict

from pyperf.data import PerformanceCommit, RepositoryAnalysis
from pyperf.constants import ANALYSIS_DIR


class APIAnalyzer:
    def __init__(self):
        self.api_to_commits: dict[str, list[PerformanceCommit]] = defaultdict(list)

    def load_analysis(self, input_file: Path) -> RepositoryAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return RepositoryAnalysis(**data)

    def create_api_to_commits_map(self, analysis: RepositoryAnalysis) -> None:
        self.commit_analysis = analysis
        for commit in analysis.performance_commits:
            for api in commit.apis:
                if api == "None":
                    continue
                self.api_to_commits[api].append(commit)

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
                    {"commit_hash": c.commit_hash, "subject": c.subject}
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


if __name__ == "__main__":
    repo_name = "Pillow"
    output_file = ANALYSIS_DIR / "commits" / f"{repo_name}_commits.json"

    analyzer = APIAnalyzer()
    commit_analysis = analyzer.load_analysis(output_file)
    analyzer.create_api_to_commits_map(commit_analysis)

    # Example usage
    analyzer.print_api_summary()
