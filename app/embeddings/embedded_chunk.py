from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.processing.chunk import DocumentChunk


@dataclass
class EmbeddedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    source_path: Path
    file_type: str

    text: str
    embedding: list[float]

    chunk_index: int
    page_number: int | None
    unit_number: int | None

    characters: int
    embedding_model: str
    embedding_dimension: int

    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: DocumentChunk,
        embedding: list[float],
        embedding_model: str,
    ) -> "EmbeddedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            source_path=chunk.source_path,
            file_type=chunk.file_type,
            text=chunk.text,
            embedding=embedding,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            unit_number=chunk.unit_number,
            characters=chunk.characters,
            embedding_model=embedding_model,
            embedding_dimension=len(embedding),
            metadata={
                **chunk.metadata,
                "embedding_model": embedding_model,
                "embedding_dimension": len(embedding),
            },
        )