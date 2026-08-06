from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentPage:
    document_name: str
    source_path: Path
    file_type: str
    text: str
    page_number: int | None = None
    unit_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    document_name: str
    source_path: Path
    file_type: str
    pages: list[DocumentPage]

    @property
    def total_units(self) -> int:
        return len(self.pages)

    @property
    def total_characters(self) -> int:
        return sum(len(page.text) for page in self.pages)