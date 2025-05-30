# type: ignore

import re
from pathlib import Path
from datetime import datetime

from gso.data import *
from gso.data.parsing import *

_ignore_extensions = ["json", "txt", "lock"]
_ignore_directories = [".venv", ".git", "__pycache__", ".pytest_cache"]
should_ignore = lambda path, ext: ext not in _ignore_extensions and (
    not any([d in path for d in _ignore_directories])
)


class SimplePatchParser:
    def parse_patch(
        self,
        diff_text: str,
        old_commit_hash: str,
    ) -> ParsedCommit:
        """
        Parse a diff message and return a ParsedCommit object
        Only requires the diff text and the old commit hash
        """
        file_diffs = []
        current_file_diff: FileDiff | None = None
        current_hunk: UniHunk | None = None

        for line in diff_text.split("\n"):
            if line.startswith("diff --git"):
                if current_file_diff:
                    path = current_file_diff.header.file.path
                    ext = Path(path).suffix.lstrip(".")
                    if should_ignore(path, ext):
                        file_diffs.append(current_file_diff)
                current_file_diff = self._parse_file_diff_header(line)
                current_hunk = None
            elif current_file_diff:
                if line.startswith("@@ "):
                    current_hunk = self._parse_hunk_header(line)
                    current_file_diff.hunks.append(current_hunk)
                elif current_hunk is None:
                    if line.startswith("index"):
                        self._parse_index_line(current_file_diff, line)
                    elif any(
                        [
                            line.startswith(mode_prefix)
                            for mode_prefix in [
                                "old mode",
                                "new mode",
                                "deleted file mode",
                                "new file mode",
                            ]
                        ]
                    ):
                        if current_file_diff.header.misc_line:
                            current_file_diff.header.misc_line += f"\n{line}"
                        else:
                            current_file_diff.header.misc_line = line
                    elif line.startswith("+++ "):
                        current_file_diff.plus_file = FileInfo(path=line[4:])
                    elif line.startswith("--- "):
                        current_file_diff.minus_file = FileInfo(path=line[4:])
                    elif line.startswith("Binary files"):
                        current_file_diff.is_binary_file = True
                        current_file_diff.binary_line = line
                else:
                    self._parse_hunk_line(current_hunk, line)

        if current_file_diff:
            path = current_file_diff.header.file.path
            ext = Path(path).suffix.lstrip(".")
            if should_ignore(path, ext):
                file_diffs.append(current_file_diff)

        # We don't have commit message, date or new hash, so use defaults
        return ParsedCommit(
            file_diffs=file_diffs,
            old_commit_hash=old_commit_hash,
            new_commit_hash="HEAD",  # Use HEAD as default new commit
            commit_message="",  # Empty commit message
            commit_date=datetime.now(),  # Current time as default
        )

    def _parse_file_diff_header(self, header: str) -> FileDiff:
        """Extract file path information from the diff header"""
        match = re.match(r"diff --git a/(\S+) b/(\S+)", header)
        if not match:
            return FileDiff(
                old_file_content="",
                new_file_content="",
                header=FileDiffHeader(file=FileInfo(path="")),
            )

        old_path, new_path = match.groups()

        # We don't attempt to get file contents since we don't have repo access
        header = FileDiffHeader(
            file=FileInfo(path=old_path),
        )
        return FileDiff(
            old_file_content="",  # We don't retrieve content in simplified version
            new_file_content="",  # We don't retrieve content in simplified version
            header=header,
        )

    def _parse_index_line(self, file_diff: FileDiff, line: str):
        """Parse the index line which contains hash information"""
        if line.startswith("index"):
            parts = line.split()
            if len(parts) < 2:
                return

            hashes = parts[1].split("..")
            if len(hashes) != 2:
                return

            file_diff.index_line = IndexLine(
                old_commit_hash=hashes[0],
                new_commit_hash=hashes[1],
                mode=parts[2] if len(parts) > 2 else "",
            )

    def _parse_hunk_header(self, header: str) -> UniHunk:
        """Parse the @@ line that describes line numbers affected"""
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", header)
        if not match:
            raise ValueError(f"Invalid hunk header: {header}")

        old_start, old_length, new_start, new_length, section = match.groups()
        return UniHunk(
            descriptor=UnitHunkDescriptor(
                old_range=Range(
                    start=int(old_start), length=int(old_length) if old_length else None
                ),
                new_range=Range(
                    start=int(new_start), length=int(new_length) if new_length else None
                ),
                section=section.lstrip(),
            ),
            line_group=LineGroup(),
        )

    def _parse_hunk_line(self, hunk: UniHunk, line: str):
        """Parse a line within a hunk (context, added, removed)"""
        if line == "" or line.startswith(" "):
            context_line = Line(content=line[1:] if line else "", type=LineType.CONTEXT)
            hunk.line_group.all_lines.append(context_line)
        elif line.startswith("-"):
            left_line = Line(content=line[1:], type=LineType.DELETED)
            hunk.line_group.all_lines.append(left_line)
        elif line.startswith("+"):
            right_line = Line(content=line[1:], type=LineType.ADDED)
            hunk.line_group.all_lines.append(right_line)
        elif line.startswith("\\"):
            note_line = Line(content=line[2:], type=LineType.NOTE)
            hunk.line_group.all_lines.append(note_line)
