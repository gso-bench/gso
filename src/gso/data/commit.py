from datetime import datetime
from pydantic import BaseModel, Field

from gso.data.parsing import FileDiff, UniHunk
from gso.data.entities import EntityType, Entity


class ParsedCommit(BaseModel):
    """
    Represents a parsed commit, with all of its file diffs
    Contains metadata about the commit, such as the commit message and date
    """

    file_diffs: list[FileDiff]
    old_commit_hash: str
    new_commit_hash: str
    commit_message: str
    commit_date: datetime
    metadata: dict = Field(default_factory=dict)

    def get_patch(self, test_file: bool = True, non_test_file: bool = True) -> str:
        patch = ""
        for file_diff in self.file_diffs:
            if file_diff.is_test_file and test_file:
                patch += file_diff.get_patch()
            if not file_diff.is_test_file and non_test_file:
                patch += file_diff.get_patch()

        return patch

    @property
    def file_name_list(self) -> list[str]:
        return [file_diff.path for file_diff in self.file_diffs]

    @property
    def file_extension_set(self) -> set[str]:
        return {file_diff.path.split(".")[-1] for file_diff in self.file_diffs}

    @property
    def is_only_python_edit(self) -> bool:
        return self.file_extension_set == {"py"}

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
        return

    @property
    def all_hunks(self) -> list[UniHunk]:
        return [hunk for file_diff in self.file_diffs for hunk in file_diff.hunks]

    @property
    def are_all_insert_hunks(self) -> bool:
        return all(hunk.is_insert_hunk for hunk in self.all_hunks)

    @property
    def are_all_delete_hunks(self) -> bool:
        return all(hunk.is_delete_hunk for hunk in self.all_hunks)

    @property
    def are_all_import_hunks(self) -> bool:
        return all(hunk.is_import_hunk for hunk in self.all_hunks)

    @property
    def are_all_insertdelete_hunks(self) -> bool:
        return all(
            hunk.is_insert_hunk or hunk.is_delete_hunk for hunk in self.all_hunks
        )

    def get_diff_by_file_name(self, file_name: str) -> FileDiff:
        for file_diff in self.file_diffs:
            if file_diff.path == file_name:
                return file_diff
        raise ValueError(f"File {file_name} not found in commit")

    def get_hunk_entity_set(
        self, entity_property_name: str, allow_test_file: bool, ignore_statements: bool
    ) -> set[Entity]:
        return {
            entity  # type: ignore
            for file_diff in self.file_diffs
            for hunk in file_diff.hunks
            for entity in getattr(hunk, entity_property_name)
            if allow_test_file or not file_diff.is_test_file
            if not ignore_statements or entity.type != EntityType.STATEMENT  # type: ignore
        }

    def edited_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "edited_entities", allow_test_file, ignore_statements
        )

    def added_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "added_entities", allow_test_file, ignore_statements
        )

    def deleted_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "deleted_entities", allow_test_file, ignore_statements
        )

    def modified_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "modified_entities", allow_test_file, ignore_statements
        )

    def num_edited_entities(self, allow_test_file=True, ignore_statements=True) -> int:
        return len(self.edited_entities(allow_test_file, ignore_statements))

    def num_added_entities(self, allow_test_file=True, ignore_statements=True) -> int:
        return len(self.added_entities(allow_test_file, ignore_statements))

    def num_deleted_entities(self, allow_test_file=True, ignore_statements=True) -> int:
        return len(self.deleted_entities(allow_test_file, ignore_statements))

    def num_modified_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> int:
        return len(self.modified_entities(allow_test_file, ignore_statements))

    def num_method_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.METHOD
            for entity in self.edited_entities(allow_test_file)
        )

    def num_function_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.FUNCTION
            for entity in self.edited_entities(allow_test_file)
        )

    def num_class_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.CLASS
            for entity in self.edited_entities(allow_test_file)
        )

    def num_statement_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.STATEMENT
            for entity in self.edited_entities(allow_test_file, ignore_statements=False)
        )
