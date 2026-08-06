from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    point_id: str
    score: float

    chunk_id: str
    document_id: str
    document_name: str
    source_path: str
    file_type: str

    text: str

    chunk_index: int
    page_number: int | None
    unit_number: int | None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )