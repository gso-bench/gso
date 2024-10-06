import json
from pathlib import Path
from collections import defaultdict

from pyperf.analysis.data.models import PerformanceCommit, RepositoryAnalysis
from pyperf.constants import ANALYSIS_DIR


class APIAnalyzer:
    def __init__(self):
        self.api_to_commits: Dict[str, list[PerformanceCommit]] = defaultdict(list)

    def load_analysis(self, input_file: Path) -> RepositoryAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return RepositoryAnalysis(**data)

    def create_api_to_commits_map(self, analysis: RepositoryAnalysis) -> None:
        for commit in analysis.performance_commits:
            for api in commit.apis:
                if api == "None":
                    continue
                self.api_to_commits[api].append(commit)

    def get_commits_for_api(self, api: str) -> list[PerformanceCommit]:
        return self.api_to_commits.get(api, [])

    def print_api_summary(self) -> None:
        sorted_apis = sorted(
            self.api_to_commits.items(), key=lambda item: len(item[1]), reverse=True
        )
        for api, commits in sorted_apis:
            print(f"API: {api}")
            print(f"Number of affecting commits: {len(commits)}")
            print("Affecting commits:")
            for commit in commits:
                print(f"  - {commit.commit_hash[:8]}: {commit.subject}")
            print()


if __name__ == "__main__":
    repo_name = "Pillow"
    output_file = ANALYSIS_DIR / "commits" / f"{repo_name}_commits.json"

    analyzer = APIAnalyzer()
    loaded_analysis = analyzer.load_analysis(output_file)
    analyzer.create_api_to_commits_map(loaded_analysis)

    # Example usage
    analyzer.print_api_summary()
