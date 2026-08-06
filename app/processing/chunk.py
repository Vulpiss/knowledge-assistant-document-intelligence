from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    source_path: Path
    file_type: str

    text: str

    chunk_index: int
    page_number: int | None
    unit_number: int | None

    characters: int

    metadata: dict[str, Any] = field(default_factory=dict)