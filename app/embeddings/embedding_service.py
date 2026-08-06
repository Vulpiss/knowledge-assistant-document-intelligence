from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import config
from app.embeddings.embedded_chunk import EmbeddedChunk
from app.processing.chunk import DocumentChunk


class EmbeddingService:

    def embed_query(self, query: str) -> list[float]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty.")

        model = self._load_model()

        logger.info(
            "Generating query embedding | characters={} | model={}",
            len(normalized_query),
            self.model_name,
        )

        try:
            embedding = model.encode(
                normalized_query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            logger.exception("Failed to generate query embedding.")
            raise RuntimeError(
                "Failed to generate query embedding."
            ) from error

        embedding_list = embedding.tolist()

        logger.info(
            "Query embedding generated | dimension={}",
            len(embedding_list),
        )

        return embedding_list

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or config.embedding_model_name
        self.batch_size = batch_size or config.embedding_batch_size
        self.model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            logger.info("Loading embedding model: {}", self.model_name)

            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as error:
                logger.exception("Failed to load embedding model: {}", self.model_name)
                raise RuntimeError(
                    f"Failed to load embedding model: {self.model_name}"
                ) from error

            logger.info("Embedding model loaded successfully.")

        return self.model

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            logger.warning("No chunks provided for embedding.")
            return []

        model = self._load_model()

        texts = [
            self._build_embedding_text(chunk)
            for chunk in chunks
        ]

        logger.info(
            "Generating embeddings | chunks={} | batch_size={} | model={}",
            len(chunks),
            self.batch_size,
            self.model_name,
        )

        try:
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            logger.exception("Failed to generate embeddings.")
            raise RuntimeError("Failed to generate embeddings.") from error

        embedded_chunks: list[EmbeddedChunk] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            embedding_list = embedding.tolist()

            embedded_chunks.append(
                EmbeddedChunk.from_chunk(
                    chunk=chunk,
                    embedding=embedding_list,
                    embedding_model=self.model_name,
                )
            )

        logger.info(
            "Embeddings generated successfully | chunks={} | dimension={}",
            len(embedded_chunks),
            embedded_chunks[0].embedding_dimension if embedded_chunks else 0,
        )

        return embedded_chunks

    @staticmethod
    def _build_embedding_text(
        chunk: DocumentChunk,
    ) -> str:
        document_label = (
            chunk.document_name
            .rsplit(".", maxsplit=1)[0]
            .replace("_", " ")
        )

        context_fields = (
            ("Dokument", document_label),
            (
                "Tytuł",
                chunk.metadata.get("document_title"),
            ),
            (
                "Status",
                chunk.metadata.get("document_status"),
            ),
            (
                "Wersja",
                chunk.metadata.get("document_version"),
            ),
        )

        context_lines = [
            f"{label}: {value}"
            for label, value in context_fields
            if isinstance(value, str) and value.strip()
        ]

        return "\n".join(
            [
                *context_lines,
                "Treść:",
                chunk.text,
            ]
        )
