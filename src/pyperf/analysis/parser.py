import re
from datetime import datetime

from pyperf.analysis.data.models import *


class DiffParser:
    def parse_git_diff(
        self,
        old_commit_hash: str,
        new_commit_hash: str,
        diff_text: str,
        commit_message: str,
        commit_date: datetime,
    ) -> ParsedCommit:
        file_diffs = []
        current_file_diff: FileDiff | None = None
        current_hunk: UniHunk | None = None
        hunk_state: str | None = None  # Can be 'pre', 'change', 'post'

        for line in diff_text.split("\n"):
            if line.startswith("diff --git"):
                if current_file_diff:
                    file_diffs.append(current_file_diff)
                current_file_diff = self.parse_file_diff_header(line)
                current_hunk = None
                hunk_state = None
            elif current_file_diff:
                if line.startswith("index"):
                    self.parse_file_diff_content(file_diff=current_file_diff, line=line)
                elif line.startswith("@@"):
                    current_hunk = self.parse_hunk_header(line)
                    current_file_diff.hunks.append(current_hunk)
                    hunk_state = "pre"
                elif current_hunk is not None:
                    hunk_state = self.parse_hunk_line(current_hunk, line, hunk_state)

        if current_file_diff:
            file_diffs.append(current_file_diff)

        return ParsedCommit(
            file_diffs=file_diffs,
            old_commit_hash=old_commit_hash,
            new_commit_hash=new_commit_hash,
            commit_message=commit_message,
            commit_date=commit_date,
        )

    def parse_file_diff_header(self, header: str) -> FileDiff:
        match = re.match(r"diff --git a/(\S+) b/(\S+)", header)
        if not match:
            raise ValueError(f"Invalid diff header: {header}")

        old_path, new_path = match.groups()
        return FileDiff(
            old_file=FileInfo(
                path=old_path, timestamp=datetime.now()  # Placeholder timestamp...
            ),
            new_file=FileInfo(
                path=new_path, timestamp=datetime.now()  # Placeholder timestamp...
            ),
            old_commit_hash="",  # To be filled later
            new_commit_hash="",  # To be filled later
            hunks=[],
        )

    def parse_file_diff_content(self, file_diff: FileDiff, line: str):
        if line.startswith("index"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid index line: {line}")
            hashes = parts[1].split("..")
            if len(hashes) != 2:
                raise ValueError(f"Invalid index hashes: {parts[1]}")
            file_diff.old_commit_hash, file_diff.new_commit_hash = hashes

    def parse_hunk_header(self, header: str) -> UniHunk:
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
                section=section.strip(),
            ),
            line_group=LineGroup(),
        )

    def parse_hunk_line(
        self, hunk: UniHunk, line: str, state: str | None
    ) -> str | None:
        """
        Parses a single line within a hunk and updates the LineGroup accordingly.
        Returns the updated state.
        """
        if line == "" or line.startswith(" "):
            context_line = ContextLine(content=line[1:])
            if state == "pre":
                hunk.line_group.pre_context_lines.append(context_line)
            elif state == "change":
                # After changes, any context lines are post-context
                hunk.line_group.post_context_lines.append(context_line)
                return "post"
            elif state == "post":
                hunk.line_group.post_context_lines.append(context_line)
        elif line.startswith("-"):
            left_line = LeftLine(content=line[1:])
            hunk.line_group.left_lines.append(left_line)
            if state != "change":
                state = "change"
        elif line.startswith("+"):
            right_line = RightLine(content=line[1:])
            hunk.line_group.right_lines.append(right_line)
            if state != "change":
                state = "change"
        elif line.startswith("\\"):
            # Note lines are typically for things like "No newline at end of file"
            note_line = NoteLine(content=line[1:].strip())
            hunk.line_group.note_line = note_line

        return state

    def parse_diff(
        self,
        old_commit_hash: str,
        new_commit_hash: str,
        diff_text: str,
        commit_message: str,
        commit_date: datetime,
    ) -> ParsedCommit:
        return self.parse_git_diff(
            old_commit_hash, new_commit_hash, diff_text, commit_message, commit_date
        )
