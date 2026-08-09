from dataclasses import dataclass
from typing import cast

from app.core.config import (
    CORPUS_PROFILES,
    CorpusName,
    CorpusProfile,
    config,
)
from app.embeddings.embedding_service import EmbeddingService
from app.generation.answer_models import GeneratedAnswer
from app.generation.answer_service import AnswerService
from app.generation.context_builder import ContextBuilder
from app.generation.llm_client import OllamaLLMClient
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.retriever import Retriever
from app.storage.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class CorpusStatus:
    corpus: CorpusName
    collection: str
    documents_directory: str
    points: int


class EmptyCorpusError(RuntimeError):
    pass


def get_corpus_profile(corpus: str) -> CorpusProfile:
    if corpus not in CORPUS_PROFILES:
        raise ValueError(f"Nieznany corpus: {corpus}")

    corpus_name = cast(CorpusName, corpus)
    return CORPUS_PROFILES[corpus_name]


def get_corpus_status(corpus: str) -> CorpusStatus:
    profile = get_corpus_profile(corpus)
    vector_store = _open_vector_store(profile)

    try:
        points = vector_store.count_points()
    finally:
        vector_store.close()

    return CorpusStatus(
        corpus=profile.name,
        collection=profile.qdrant_collection,
        documents_directory=str(profile.raw_documents_dir),
        points=points,
    )


def answer_question(
    *,
    corpus: str,
    question: str,
    embedding_service: EmbeddingService,
) -> GeneratedAnswer:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Pytanie nie może być puste.")

    profile = get_corpus_profile(corpus)
    vector_store = _open_vector_store(profile)

    try:
        points = vector_store.count_points()

        if points == 0:
            raise EmptyCorpusError(
                "Kolekcja Qdrant jest pusta. Uruchom indeksowanie "
                f"dla corpusu {profile.name}."
            )

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            top_k=config.answer_top_k,
        )
        answer_service = AnswerService(
            retriever=retriever,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_client=OllamaLLMClient(),
            top_k=config.answer_top_k,
        )

        return answer_service.answer(normalized_question)
    finally:
        vector_store.close()


def _open_vector_store(
    profile: CorpusProfile,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        database_path=config.qdrant_path,
        collection_name=profile.qdrant_collection,
    )
