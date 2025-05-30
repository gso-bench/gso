from enum import Enum
from pydantic import BaseModel, Field
from gso.data.entities import EntityType, Entity


class Range(BaseModel):
    start: int
    length: int | None = None

    def get_patch(self) -> str:
        if self.length is None:
            return f"{self.start}"
        return f"{self.start},{self.length}"


class UnitHunkDescriptor(BaseModel):
    old_range: Range
    new_range: Range
    section: str

    def get_patch(self) -> str:
        content = f"@@ -{self.old_range.get_patch()} +{self.new_range.get_patch()} @@"
        if self.section:
            content += f" {self.section}"
        return content


class LineType(Enum):
    CONTEXT = "context"
    ADDED = "added"
    DELETED = "deleted"
    NOTE = "note"


class Line(BaseModel):
    content: str
    type: LineType


class LineGroup(BaseModel):
    all_lines: list[Line] = Field(default_factory=list)

    @property
    def num_deleted(self) -> int:
        return sum(line.type == LineType.DELETED for line in self.all_lines)

    @property
    def num_added(self) -> int:
        return sum(line.type == LineType.ADDED for line in self.all_lines)

    @property
    def num_context(self) -> int:
        return sum(line.type == LineType.CONTEXT for line in self.all_lines)

    @property
    def lr_lines(self) -> list[Line]:
        return [
            line
            for line in self.all_lines
            if line.type in [LineType.DELETED, LineType.CONTEXT]
        ]

    @property
    def num_edited(self) -> int:
        return self.num_deleted + self.num_added


class UniHunk(BaseModel):
    descriptor: UnitHunkDescriptor
    line_group: LineGroup
    modified_entities: set[Entity] = Field(default_factory=set)
    added_entities: set[Entity] = Field(default_factory=set)
    deleted_entities: set[Entity] = Field(default_factory=set)

    @property
    def is_import_hunk(self) -> bool:
        for line in self.line_group.lr_lines:
            if len(line.content.strip()) == 0:
                continue
            if line.content.startswith("import"):
                continue
            if line.content.startswith("from ") and "import" in line.content:
                continue
            return False
        return True

    @property
    def is_insert_hunk(self) -> bool:
        return self.line_group.num_deleted == 0

    @property
    def is_delete_hunk(self) -> bool:
        return self.line_group.num_added == 0

    @property
    def edited_entities(self) -> set[Entity]:
        return self.modified_entities.union(self.added_entities).union(
            self.deleted_entities
        )

    @property
    def num_edited_entities(self) -> int:
        return len(self.edited_entities)

    @property
    def num_modified_entities(self) -> int:
        return len(self.modified_entities)

    @property
    def num_added_entities(self) -> int:
        return len(self.added_entities)

    @property
    def num_deleted_entities(self) -> int:
        return len(self.deleted_entities)

    @property
    def num_method_entities(self) -> int:
        return sum(entity.type == EntityType.METHOD for entity in self.edited_entities)

    @property
    def num_function_entities(self) -> int:
        return sum(
            entity.type == EntityType.FUNCTION for entity in self.edited_entities
        )

    @property
    def num_class_entities(self) -> int:
        return sum(entity.type == EntityType.CLASS for entity in self.edited_entities)

    @property
    def edit_transcends_single_location(self) -> bool:
        return (self.num_function_entities + self.num_class_entities > 1) or (
            self.num_method_entities > 1
        )


class FileInfo(BaseModel):
    path: str


class FileDiffHeader(BaseModel):
    file: FileInfo
    misc_line: str | None = None

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or "tests" in self.path.split("/")
        )

    def get_patch(self) -> str:
        patch = f"diff --git a/{self.file.path} b/{self.file.path}\n"
        if self.misc_line:
            patch += self.misc_line + "\n"
        return patch


class IndexLine(BaseModel):
    old_commit_hash: str
    new_commit_hash: str
    mode: str

    def get_patch(self) -> str:
        return f"index {self.old_commit_hash}..{self.new_commit_hash}{' ' if self.mode else ''}{self.mode}\n"


class FileDiff(BaseModel):
    old_file_content: str
    new_file_content: str
    header: FileDiffHeader
    index_line: IndexLine | None = None
    is_binary_file: bool = False
    binary_line: str | None = None
    minus_file: FileInfo | None = None
    plus_file: FileInfo | None = None
    hunks: list[UniHunk] = []

    @property
    def path(self) -> str:
        return self.header.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or "tests" in self.path.split("/")
        )

    def get_patch(self) -> str:
        patch = self.header.get_patch()
        if self.index_line:
            patch += self.index_line.get_patch()
        if self.is_binary_file:
            patch += self.binary_line + "\n"

        if self.minus_file and self.plus_file:
            patch += f"--- {self.minus_file.path}\n"
            patch += f"+++ {self.plus_file.path}\n"
        for hunk in self.hunks:
            patch += hunk.descriptor.get_patch() + "\n"
            for line in hunk.line_group.all_lines:
                if line.type == LineType.CONTEXT:
                    patch += f" {line.content}\n"
                elif line.type == LineType.ADDED:
                    patch += f"+{line.content}\n"
                elif line.type == LineType.DELETED:
                    patch += f"-{line.content}\n"
                elif line.type == LineType.NOTE:
                    patch += f"\\ {line.content}\n"

        return patch

    @property
    def is_python_file(self) -> bool:
        return self.path.endswith(".py")

    @property
    def num_hunks(self) -> int:
        return len(self.hunks)

    @property
    def edited_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.edited_entities}

    @property
    def added_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.added_entities}

    @property
    def deleted_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.deleted_entities}

    @property
    def modified_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.modified_entities}

    @property
    def num_edited_entities(self) -> int:
        return len(self.edited_entities)

    @property
    def num_added_entities(self) -> int:
        return len(self.added_entities)

    @property
    def num_deleted_entities(self) -> int:
        return len(self.deleted_entities)

    @property
    def num_modified_entities(self) -> int:
        return len(self.modified_entities)

    @property
    def num_method_entities(self) -> int:
        return sum(entity.type == EntityType.METHOD for entity in self.edited_entities)

    @property
    def num_function_entities(self) -> int:
        return sum(
            entity.type == EntityType.FUNCTION for entity in self.edited_entities
        )

    @property
    def num_class_entities(self) -> int:
        return sum(entity.type == EntityType.CLASS for entity in self.edited_entities)
