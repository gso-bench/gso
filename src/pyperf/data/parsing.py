from datetime import datetime
from pydantic import BaseModel, Field


class Range(BaseModel):
    start: int
    length: int | None = None


class UnitHunkDescriptor(BaseModel):
    old_range: Range
    new_range: Range
    section: str


class Line(BaseModel):
    content: str


class ContextLine(Line):
    pass


class LeftLine(Line):
    pass


class RightLine(Line):
    pass


class NoteLine(Line):
    pass


class LineGroup(BaseModel):
    pre_context_lines: list[ContextLine] = Field(default_factory=list)
    left_lines: list[LeftLine] = Field(default_factory=list)
    right_lines: list[RightLine] = Field(default_factory=list)
    post_context_lines: list[ContextLine] = Field(default_factory=list)
    note_line: NoteLine | None = None

    @property
    def num_deleted(self) -> int:
        return len(self.left_lines)

    @property
    def num_added(self) -> int:
        return len(self.right_lines)

    @property
    def num_context(self) -> int:
        return len(self.pre_context_lines) + len(self.post_context_lines)

    @property
    def num_edited(self) -> int:
        return self.num_deleted + self.num_added


class UniHunk(BaseModel):
    descriptor: UnitHunkDescriptor
    line_group: LineGroup

    @property
    def is_import_hunk(self) -> bool:
        for line in self.line_group.left_lines + self.line_group.right_lines:
            if line.content.startswith("import") or len(line.strip()) == 0:  # type: ignore
                continue
            return False
        return True


class FileInfo(BaseModel):
    path: str
    timestamp: datetime


class FileDiff(BaseModel):
    old_file: FileInfo
    new_file: FileInfo
    old_commit_hash: str
    new_commit_hash: str
    hunks: list[UniHunk]

    @property
    def path(self) -> str:
        return self.new_file.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or "tests" in self.path.split("/")
        )

    def get_patch(self) -> str:
        patch = f"diff --git a/{self.old_file.path} b/{self.new_file.path}\n"
        patch += f"index {self.old_commit_hash}..{self.new_commit_hash}\n"
        patch += f"--- a/{self.old_file.path}\n"
        patch += f"+++ b/{self.new_file.path}\n"
        for hunk in self.hunks:
            patch += f"@@ -{hunk.descriptor.old_range.start},{hunk.descriptor.old_range.length} +{hunk.descriptor.new_range.start},{hunk.descriptor.new_range.length} @@ {hunk.descriptor.section}\n"
            for line in hunk.line_group.pre_context_lines:
                patch += f" {line.content}\n"
            for line in hunk.line_group.left_lines:
                patch += f"-{line.content}\n"
            for line in hunk.line_group.right_lines:
                patch += f"+{line.content}\n"
            for line in hunk.line_group.post_context_lines:
                patch += f" {line.content}\n"
            if hunk.line_group.note_line:
                patch += f" \\\\ {hunk.line_group.note_line.content}\n"

            patch += "\n"
        patch += "\n\n"
        return patch
