from dataclasses import dataclass

from loguru import logger

from app.core.config import config
from app.retrieval.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True)
class ContextSource:
    source_id: int
    document_name: str
    page_number: int | None
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True)
class BuiltContext:
    text: str
    sources: tuple[ContextSource, ...]


class ContextBuilder:
    def __init__(
        self,
        max_characters: int | None = None,
    ) -> None:
        self.max_characters = (
            max_characters
            if max_characters is not None
            else config.max_context_characters
        )

        if self.max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than 0."
            )

    def build(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> BuiltContext:
        blocks: list[str] = []
        sources: list[ContextSource] = []
        used_characters = 0

        for source_id, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            page_display = (
                str(chunk.page_number)
                if chunk.page_number is not None
                else "brak"
            )

            metadata_lines = self._build_metadata_lines(
                chunk
            )

            header = (
                f"[SOURCE {source_id}]\n"
                f"document: {chunk.document_name}\n"
                f"page: {page_display}\n"
                f"chunk_id: {chunk.chunk_id}\n"
                f"{metadata_lines}"
                f"content:\n"
            )

            footer = f"\n[/SOURCE {source_id}]"

            remaining_characters = (
                self.max_characters
                - used_characters
                - len(header)
                - len(footer)
            )

            if remaining_characters <= 0:
                break

            source_text = chunk.text.strip()

            if len(source_text) > remaining_characters:
                source_text = (
                    source_text[:remaining_characters]
                    .rsplit(" ", maxsplit=1)[0]
                    .rstrip()
                )

            if not source_text:
                continue

            block = f"{header}{source_text}{footer}"

            blocks.append(block)
            used_characters += len(block)

            sources.append(
                ContextSource(
                    source_id=source_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                    text=source_text,
                )
            )

        logger.info(
            "Context built | sources={} | characters={}",
            len(sources),
            used_characters,
        )

        return BuiltContext(
            text="\n\n".join(blocks),
            sources=tuple(sources),
        )

    @staticmethod
    def _build_metadata_lines(
        chunk: RetrievedChunk,
    ) -> str:
        fields = (
            (
                "title",
                chunk.metadata.get("document_title"),
            ),
            (
                "status",
                chunk.metadata.get("document_status"),
            ),
            (
                "version",
                chunk.metadata.get("document_version"),
            ),
            (
                "valid_from",
                chunk.metadata.get("document_valid_from"),
            ),
            (
                "valid_until",
                chunk.metadata.get("document_valid_until"),
            ),
        )

        return "".join(
            f"{name}: {value.strip()}\n"
            for name, value in fields
            if isinstance(value, str) and value.strip()
        )
