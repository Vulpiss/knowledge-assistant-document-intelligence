from typing import Any

from loguru import logger

from app.core.config import config
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.reranker import HybridReranker
from app.retrieval.retrieved_chunk import RetrievedChunk
from app.storage.vector_search_result import VectorSearchResult
from app.storage.vector_store import QdrantVectorStore


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        top_k: int | None = None,
        candidate_pool: int | None = None,
        reranker: HybridReranker | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = (
            config.retrieval_top_k
            if top_k is None
            else top_k
        )
        self.candidate_pool = (
            config.retrieval_candidate_pool
            if candidate_pool is None
            else candidate_pool
        )
        self.reranker = reranker or HybridReranker(
            dense_weight=config.retrieval_dense_weight,
            lexical_weight=config.retrieval_lexical_weight,
            document_weight=config.retrieval_document_weight,
            archive_penalty=config.retrieval_archive_penalty,
            max_chunks_per_document=(
                config.retrieval_max_chunks_per_document
            ),
        )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if self.candidate_pool <= 0:
            raise ValueError(
                "candidate_pool must be greater than 0"
            )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        result_limit = (
            self.top_k
            if top_k is None
            else top_k
        )

        if result_limit <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        candidate_limit = max(
            result_limit,
            self.candidate_pool,
        )

        logger.info(
            "Starting hybrid retrieval | query_characters={} | "
            "candidate_pool={} | top_k={}",
            len(normalized_query),
            candidate_limit,
            result_limit,
        )

        query_vector = (
            self.embedding_service.embed_query(
                normalized_query
            )
        )

        search_results = self.vector_store.search(
            query_vector=query_vector,
            limit=candidate_limit,
        )

        candidate_chunks = [
            self._map_search_result(result)
            for result in search_results
        ]
        retrieved_chunks = self.reranker.rerank(
            query=normalized_query,
            chunks=candidate_chunks,
            limit=result_limit,
        )

        logger.info(
            "Hybrid retrieval completed | candidates={} | results={}",
            len(candidate_chunks),
            len(retrieved_chunks),
        )

        return retrieved_chunks

    def _map_search_result(
        self,
        result: VectorSearchResult,
    ) -> RetrievedChunk:
        payload = result.payload

        metadata = payload.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        return RetrievedChunk(
            point_id=result.point_id,
            score=result.score,
            chunk_id=str(
                payload.get("chunk_id", "")
            ),
            document_id=str(
                payload.get("document_id", "")
            ),
            document_name=str(
                payload.get("document_name", "")
            ),
            source_path=str(
                payload.get("source_path", "")
            ),
            file_type=str(
                payload.get("file_type", "")
            ),
            text=str(
                payload.get("text", "")
            ),
            chunk_index=self._to_int(
                payload.get("chunk_index"),
                default=0,
            ),
            page_number=self._to_optional_int(
                payload.get("page_number")
            ),
            unit_number=self._to_optional_int(
                payload.get("unit_number")
            ),
            metadata=metadata,
        )

    @staticmethod
    def _to_int(
        value: Any,
        default: int,
    ) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_optional_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None
