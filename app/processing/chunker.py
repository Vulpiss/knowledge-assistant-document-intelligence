import hashlib
import re

from loguru import logger

from app.core.config import config
from app.processing.chunk import DocumentChunk
from app.processing.processed_document import ProcessedDocumentUnit


class RecursiveTextChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk_units(self, units: list[ProcessedDocumentUnit]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for unit in units:
            unit_chunks = self._chunk_single_unit(unit)
            chunks.extend(unit_chunks)

        logger.info(
            "Chunking completed | units={} | chunks={} | chunk_size={} | overlap={}",
            len(units),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        return chunks

    def _chunk_single_unit(self, unit: ProcessedDocumentUnit) -> list[DocumentChunk]:
        text_parts = self._split_text(unit.text)
        packed_chunks = self._pack_text_parts(text_parts)

        document_chunks: list[DocumentChunk] = []

        for index, chunk_text in enumerate(packed_chunks, start=1):
            chunk_id = self._build_chunk_id(
                document_id=unit.document_id,
                unit_number=unit.unit_number,
                page_number=unit.page_number,
                chunk_index=index,
                text=chunk_text,
            )

            metadata = {
                **unit.metadata,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunk_characters": len(chunk_text),
            }

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=unit.document_id,
                    document_name=unit.document_name,
                    source_path=unit.source_path,
                    file_type=unit.file_type,
                    text=chunk_text,
                    chunk_index=index,
                    page_number=unit.page_number,
                    unit_number=unit.unit_number,
                    characters=len(chunk_text),
                    metadata=metadata,
                )
            )

        return document_chunks

    def _split_text(self, text: str) -> list[str]:
        if not text:
            return []

        paragraphs = re.split(r"\n\s*\n", text)

        parts: list[str] = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) <= self.chunk_size:
                parts.append(paragraph)
                continue

            sentences = self._split_long_paragraph(paragraph)
            parts.extend(sentences)

        return parts

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentence_candidates = re.split(r"(?<=[.!?])\s+", paragraph)

        parts: list[str] = []

        for sentence in sentence_candidates:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= self.chunk_size:
                parts.append(sentence)
            else:
                parts.extend(self._hard_split(sentence))

        return parts

    def _hard_split(self, text: str) -> list[str]:
        return [
            text[start:start + self.chunk_size].strip()
            for start in range(0, len(text), self.chunk_size)
            if text[start:start + self.chunk_size].strip()
        ]

    def _pack_text_parts(self, parts: list[str]) -> list[str]:
        chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            if not current_chunk:
                current_chunk = part
                continue

            candidate = f"{current_chunk}\n\n{part}"

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = f"{overlap_text}\n\n{part}".strip() if overlap_text else part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        if self.chunk_overlap == 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        return text[-self.chunk_overlap:].strip()

    def _build_chunk_id(
        self,
        document_id: str,
        unit_number: int | None,
        page_number: int | None,
        chunk_index: int,
        text: str,
    ) -> str:
        raw_id = (
            f"{document_id}|"
            f"unit={unit_number}|"
            f"page={page_number}|"
            f"chunk={chunk_index}|"
            f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
        )

        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]