from dataclasses import dataclass
from typing import cast

import requests

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
from app.services.document_library import (
    DocumentInfo,
    DocumentLibrary,
)
from app.services.indexing_service import (
    IndexingService,
    IndexingSummary,
)
from app.storage.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class CorpusStatus:
    corpus: CorpusName
    collection: str
    documents_directory: str
    documents: int
    points: int


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    model_available: bool
    message: str


class EmptyCorpusError(RuntimeError):
    pass


def get_corpus_profile(corpus: str) -> CorpusProfile:
    if corpus not in CORPUS_PROFILES:
        raise ValueError(f"Nieznany corpus: {corpus}")

    corpus_name = cast(CorpusName, corpus)
    return CORPUS_PROFILES[corpus_name]


def get_corpus_status(corpus: str) -> CorpusStatus:
    profile = get_corpus_profile(corpus)
    documents = DocumentLibrary(
        profile.raw_documents_dir
    ).list_documents()
    vector_store = _open_vector_store(profile)

    try:
        points = vector_store.count_points()
    finally:
        vector_store.close()

    return CorpusStatus(
        corpus=profile.name,
        collection=profile.qdrant_collection,
        documents_directory=str(profile.raw_documents_dir),
        documents=len(documents),
        points=points,
    )


def get_ollama_status(
    timeout_seconds: float = 3.0,
) -> OllamaStatus:
    endpoint = f"{config.ollama_base_url.rstrip('/')}/api/tags"

    try:
        response = requests.get(
            endpoint,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return OllamaStatus(
            available=False,
            model_available=False,
            message="Ollama nie odpowiada.",
        )

    raw_models = (
        payload.get("models", [])
        if isinstance(payload, dict)
        else []
    )
    models = raw_models if isinstance(raw_models, list) else []
    model_names = {
        name
        for model in models
        if isinstance(model, dict)
        for name in (model.get("name"), model.get("model"))
        if isinstance(name, str)
    }
    model_available = config.ollama_model in model_names

    if model_available:
        message = f"Ollama i model {config.ollama_model} są gotowe."
    else:
        message = (
            "Ollama działa, ale brakuje modelu "
            f"{config.ollama_model}."
        )

    return OllamaStatus(
        available=True,
        model_available=model_available,
        message=message,
    )


def list_production_documents() -> list[DocumentInfo]:
    return _production_library().list_documents()


def save_production_document(
    file_name: str,
    content: bytes,
) -> DocumentInfo:
    return _production_library().save_document(
        file_name,
        content,
    )


def delete_production_document(file_name: str) -> bool:
    return _production_library().delete_document(file_name)


def rebuild_production_index(
    *,
    embedding_service: EmbeddingService,
) -> IndexingSummary:
    profile = CORPUS_PROFILES["production"]
    documents = _production_library().list_documents()

    if not documents:
        vector_store = _open_vector_store(profile)

        try:
            vector_store.delete_collection()
        finally:
            vector_store.close()

        return IndexingSummary(
            documents=0,
            processed_units=0,
            chunks=0,
            stored_points=0,
            total_points=0,
        )

    indexing_service = IndexingService(
        embedding_service=embedding_service,
        documents_directory=profile.raw_documents_dir,
        database_path=config.qdrant_path,
        collection_name=profile.qdrant_collection,
    )

    return indexing_service.index(rebuild=True)


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
            if profile.name == "production":
                message = (
                    "Indeks prywatnych dokumentów jest pusty. "
                    "Dodaj dokumenty i przebuduj indeks w panelu bocznym."
                )
            else:
                message = (
                    "Kolekcja Qdrant jest pusta. Uruchom indeksowanie "
                    f"dla corpusu {profile.name}."
                )

            raise EmptyCorpusError(
                message
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


def _production_library() -> DocumentLibrary:
    return DocumentLibrary(
        CORPUS_PROFILES["production"].raw_documents_dir
    )
