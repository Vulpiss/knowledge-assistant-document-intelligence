from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcessedDocumentUnit:
    document_id: str
    document_name: str
    source_path: Path
    file_type: str

    text: str

    page_number: int | None
    unit_number: int | None

    raw_characters: int
    clean_characters: int

    metadata: dict[str, Any] = field(default_factory=dict)