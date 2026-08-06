from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.core.config import config
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.loader_factory import DocumentLoaderFactory
from app.processing.chunk import DocumentChunk
from app.processing.chunker import RecursiveTextChunker
from app.processing.document_processor import DocumentProcessor
from app.storage.vector_store import QdrantVectorStore


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


@dataclass(frozen=True)
class IndexingSummary:
    documents: int
    processed_units: int
    chunks: int
    stored_points: int
    total_points: int


class IndexingService:
    def __init__(
        self,
        loader_factory: DocumentLoaderFactory | None = None,
        document_processor: DocumentProcessor | None = None,
        chunker: RecursiveTextChunker | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.loader_factory = (
            loader_factory or DocumentLoaderFactory()
        )

        self.document_processor = (
            document_processor or DocumentProcessor()
        )

        self.chunker = (
            chunker or RecursiveTextChunker()
        )

        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

    def index(
        self,
        rebuild: bool = False,
    ) -> IndexingSummary:
        config.ensure_directories()

        document_paths = self._find_document_paths(
            config.raw_documents_dir
        )

        if not document_paths:
            raise RuntimeError(
                "Nie znaleziono dokumentów PDF, DOCX ani TXT "
                f"w katalogu: {config.raw_documents_dir}"
            )

        logger.info(
            "Starting document indexing | documents={} | rebuild={}",
            len(document_paths),
            rebuild,
        )

        all_chunks: list[DocumentChunk] = []
        total_processed_units = 0

        for document_path in document_paths:
            loaded_document = self.loader_factory.load(
                document_path
            )

            logger.info(
                "Loaded document | name={} | type={} | units={} | characters={}",
                loaded_document.document_name,
                loaded_document.file_type,
                loaded_document.total_units,
                loaded_document.total_characters,
            )

            processed_units = self.document_processor.process(
                loaded_document
            )

            total_processed_units += len(processed_units)

            chunks = self.chunker.chunk_units(
                processed_units
            )

            logger.info(
                "Document chunked | document={} | chunks={}",
                loaded_document.document_name,
                len(chunks),
            )

            all_chunks.extend(chunks)

        if not all_chunks:
            raise RuntimeError(
                "Dokumenty zostały odczytane, ale nie utworzono "
                "żadnych chunków."
            )

        embedded_chunks = self.embedding_service.embed_chunks(
            all_chunks
        )

        if not embedded_chunks:
            raise RuntimeError(
                "Nie udało się utworzyć embeddingów."
            )

        vector_store = QdrantVectorStore()

        try:
            if rebuild:
                vector_store.delete_collection()

            stored_points = vector_store.upsert_embeddings(
                embedded_chunks
            )

            total_points = vector_store.count_points()

        finally:
            vector_store.close()

        summary = IndexingSummary(
            documents=len(document_paths),
            processed_units=total_processed_units,
            chunks=len(all_chunks),
            stored_points=stored_points,
            total_points=total_points,
        )

        logger.info(
            "Indexing completed | documents={} | units={} | "
            "chunks={} | stored_points={} | total_points={}",
            summary.documents,
            summary.processed_units,
            summary.chunks,
            summary.stored_points,
            summary.total_points,
        )

        return summary

    def _find_document_paths(
        self,
        directory: Path,
    ) -> list[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if self._is_valid_document_path(path)
        )

    @staticmethod
    def _is_valid_document_path(
        path: Path,
    ) -> bool:
        if not path.is_file():
            return False

        if path.name.startswith("."):
            return False

        if (
            path.suffix.lower()
            not in SUPPORTED_DOCUMENT_EXTENSIONS
        ):
            return False

        try:
            file_size = path.stat().st_size
        except OSError:
            logger.warning(
                "Cannot read file metadata. Skipping: {}",
                path,
            )
            return False

        if file_size == 0:
            logger.warning(
                "Skipping empty file: {}",
                path,
            )
            return False

        return True