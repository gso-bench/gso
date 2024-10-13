from datetime import datetime
from pydantic import BaseModel

from pyperf.data.parsing import FileDiff


class ParsedCommit(BaseModel):
    file_diffs: list[FileDiff]
    old_commit_hash: str
    new_commit_hash: str
    commit_message: str

    def get_patch(self) -> str:
        patch = ""
        for file_diff in self.file_diffs:
            patch += file_diff.get_patch()

        return patch

    @property
    def num_files(self) -> int:
        return len(self.file_diffs)

    @property
    def num_test_files(self) -> int:
        return sum(file_diff.is_test_file for file_diff in self.file_diffs)

    @property
    def num_non_test_files(self) -> int:
        return self.num_files - self.num_test_files

    @property
    def num_hunks(self) -> int:
        return sum(len(file_diff.hunks) for file_diff in self.file_diffs)

    @property
    def num_edited_lines(self) -> int:
        return sum(
            hunk.line_group.num_edited
            for file_diff in self.file_diffs
            for hunk in file_diff.hunks
        )

    @property
    def num_non_test_edited_lines(self) -> int:
        return sum(
            hunk.line_group.num_edited
            for file_diff in self.file_diffs
            if not file_diff.is_test_file
            for hunk in file_diff.hunks
        )

    @property
    def is_bugfix(self) -> bool:
        return (
            "fix" in self.commit_message.lower() or "bug" in self.commit_message.lower()
        )

    @property
    def is_feature(self) -> bool:
        return (
            "feature" in self.commit_message.lower()
            or "add" in self.commit_message.lower()
        )

    @property
    def is_refactor(self) -> bool:
        return "refactor" in self.commit_message.lower()

    @property
    def commit_date(self) -> datetime:
        # TODO: implement this, for now return dummy value
        return datetime.now()
