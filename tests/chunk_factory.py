from typing import Any

from app.retrieval.retrieved_chunk import RetrievedChunk


def make_chunk(
    *,
    document_name: str,
    text: str,
    score: float = 0.5,
    chunk_id: str = "chunk-1",
    metadata: dict[str, Any] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=f"point-{chunk_id}",
        score=score,
        chunk_id=chunk_id,
        document_id=f"document-{document_name}",
        document_name=document_name,
        source_path=f"data/{document_name}",
        file_type="txt",
        text=text,
        chunk_index=0,
        page_number=None,
        unit_number=1,
        metadata=metadata or {},
    )
