import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models

from app.core.config import config
from app.embeddings.embedded_chunk import EmbeddedChunk

from app.storage.vector_search_result import VectorSearchResult


class QdrantVectorStore:
    def __init__(
        self,
        database_path: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.database_path = database_path or config.qdrant_path
        self.collection_name = (
            collection_name or config.qdrant_collection
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Opening local Qdrant database | path={} | collection={}",
            self.database_path,
            self.collection_name,
        )

        self.client = QdrantClient(
            path=str(self.database_path)
        )

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than 0"
            )

        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            self._validate_collection_dimension(vector_size)

            logger.info(
                "Qdrant collection already exists: {}",
                self.collection_name,
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        logger.info(
            "Created Qdrant collection | name={} | dimension={} | distance=cosine",
            self.collection_name,
            vector_size,
        )

    def upsert_embeddings(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> int:
        if not embedded_chunks:
            logger.warning(
                "No embedded chunks provided to vector store."
            )
            return 0

        dimensions = {
            chunk.embedding_dimension
            for chunk in embedded_chunks
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All embeddings must have the same dimension. "
                f"Received dimensions: {sorted(dimensions)}"
            )

        vector_size = dimensions.pop()
        self.ensure_collection(vector_size)

        points = [
            models.PointStruct(
                id=self._build_point_id(chunk.chunk_id),
                vector=chunk.embedding,
                payload=self._build_payload(chunk),
            )
            for chunk in embedded_chunks
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        logger.info(
            "Embedded chunks stored in Qdrant | collection={} | points={}",
            self.collection_name,
            len(points),
        )

        return len(points)

    def search(
            self,
            query_vector: list[float],
            limit: int = 5,
    ) -> list[VectorSearchResult]:
        if not query_vector:
            raise ValueError(
                "Query vector cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "Search limit must be greater than 0."
            )

        if not self.client.collection_exists(
                collection_name=self.collection_name
        ):
            raise RuntimeError(
                f"Qdrant collection does not exist: "
                f"{self.collection_name}"
            )

        logger.info(
            "Searching Qdrant | collection={} | "
            "dimension={} | limit={}",
            self.collection_name,
            len(query_vector),
            limit,
        )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as error:
            logger.exception(
                "Qdrant search failed | collection={}",
                self.collection_name,
            )
            raise RuntimeError(
                "Failed to search Qdrant collection."
            ) from error

        results = [
            VectorSearchResult(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

        logger.info(
            "Qdrant search completed | results={}",
            len(results),
        )

        return results

    def delete_collection(self) -> bool:
        if not self.client.collection_exists(
                collection_name=self.collection_name
        ):
            logger.info(
                "Qdrant collection does not exist. "
                "Nothing to delete: {}",
                self.collection_name,
            )
            return False

        deleted = self.client.delete_collection(
            collection_name=self.collection_name
        )

        logger.warning(
            "Qdrant collection deleted | collection={}",
            self.collection_name,
        )

        return bool(deleted)

    def count_points(self) -> int:
        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):
            return 0

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def close(self) -> None:
        self.client.close()

        logger.info(
            "Qdrant database connection closed."
        )

    def _validate_collection_dimension(
        self,
        expected_dimension: int,
    ) -> None:
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )

        vectors_config = (
            collection_info.config.params.vectors
        )

        existing_dimension = getattr(
            vectors_config,
            "size",
            None,
        )

        if (
            existing_dimension is not None
            and existing_dimension != expected_dimension
        ):
            raise ValueError(
                "Qdrant collection dimension does not match "
                "the embedding model dimension. "
                f"Collection={existing_dimension}, "
                f"embedding={expected_dimension}. "
                "Use a new collection or rebuild the existing one."
            )

    @staticmethod
    def _build_point_id(chunk_id: str) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"ai-knowledge-assistant:{chunk_id}",
            )
        )

    @staticmethod
    def _build_payload(
        chunk: EmbeddedChunk,
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "source_path": str(chunk.source_path),
            "file_type": chunk.file_type,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "unit_number": chunk.unit_number,
            "characters": chunk.characters,
            "embedding_model": chunk.embedding_model,
            "embedding_dimension": chunk.embedding_dimension,
            "metadata": chunk.metadata,
        }