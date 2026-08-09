from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.core.config import CORPUS_PROFILES, config
from app.core.logger import setup_logger
from app.embeddings.embedding_service import EmbeddingService
from app.generation.answer_models import GeneratedAnswer
from app.interfaces.web_runtime import (
    EmptyCorpusError,
    answer_question,
    get_corpus_status,
)


APP_TITLE = "Knowledge Assistant"
HISTORY_KEY = "knowledge_assistant_histories"

EXAMPLE_QUESTIONS = {
    "v1": (
        "Ile dni wcześniej należy złożyć wniosek urlopowy?",
        "Po jakim czasie wymieniany jest laptop służbowy?",
        "Ile znaków powinno mieć hasło?",
    ),
    "v2": (
        "Aktualna polityka wymaga hasła 12- czy 14-znakowego?",
        "Jaki jest tygodniowy limit pracy zdalnej?",
        "Jaka jest miesięczna premia pracownika?",
    ),
}


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    setup_logger()
    _apply_styles()
    _initialize_state()

    corpus = _render_sidebar()
    history = _history_for(corpus)

    st.title("📚 Knowledge Assistant")
    st.caption(
        "Zadawaj pytania do dokumentów. Każda odpowiedź jest "
        "oparta na wyszukanych źródłach."
    )

    _render_history(history)

    example_question = _render_examples(
        corpus=corpus,
        show=not history,
    )
    typed_question = st.chat_input(
        "Zadaj pytanie do dokumentów…"
    )
    question = example_question or typed_question

    if question:
        _process_question(
            corpus=corpus,
            question=question,
            history=history,
        )


@st.cache_resource(show_spinner=False)
def _embedding_service() -> EmbeddingService:
    return EmbeddingService()


def _initialize_state() -> None:
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = {
            corpus: []
            for corpus in CORPUS_PROFILES
        }


def _history_for(corpus: str) -> list[dict[str, Any]]:
    histories = st.session_state[HISTORY_KEY]

    if corpus not in histories:
        histories[corpus] = []

    return histories[corpus]


def _render_sidebar() -> str:
    with st.sidebar:
        st.header("Ustawienia")
        corpus = st.selectbox(
            "Zestaw dokumentów",
            options=tuple(CORPUS_PROFILES),
            index=1,
            format_func=lambda value: value.upper(),
            help=(
                "V1 to podstawowy corpus. V2 zawiera trudniejsze "
                "konflikty wersji i testy bezpieczeństwa."
            ),
        )

        profile = CORPUS_PROFILES[corpus]
        st.caption(f"Kolekcja: `{profile.qdrant_collection}`")

        try:
            status = get_corpus_status(corpus)

            if status.points > 0:
                st.success(
                    f"Indeks gotowy: {status.points} chunków",
                    icon="✅",
                )
            else:
                st.warning("Indeks jest pusty.", icon="⚠️")
        except Exception as error:
            st.error(
                _friendly_error(error),
                icon="🚫",
            )

        st.divider()

        if st.button(
            "Wyczyść rozmowę",
            use_container_width=True,
        ):
            st.session_state[HISTORY_KEY][corpus] = []
            st.rerun()

        st.caption(
            f"Model: `{config.ollama_model}`  \n"
            "Dane pozostają na tym komputerze."
        )

    return corpus


def _render_history(history: list[dict[str, Any]]) -> None:
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
                continue

            _render_assistant_payload(message)


def _render_examples(
    *,
    corpus: str,
    show: bool,
) -> str | None:
    if not show:
        return None

    st.subheader("Przykładowe pytania")
    selected: str | None = None

    for index, question in enumerate(
        EXAMPLE_QUESTIONS[corpus]
    ):
        if st.button(
            question,
            key=f"example_{corpus}_{index}",
            use_container_width=True,
        ):
            selected = question

    return selected


def _process_question(
    *,
    corpus: str,
    question: str,
    history: list[dict[str, Any]],
) -> None:
    normalized_question = question.strip()

    if not normalized_question:
        return

    history.append(
        {
            "role": "user",
            "content": normalized_question,
        }
    )

    with st.chat_message("user"):
        st.write(normalized_question)

    with st.chat_message("assistant"):
        with st.spinner("Szukam w dokumentach i przygotowuję odpowiedź…"):
            try:
                generated = answer_question(
                    corpus=corpus,
                    question=normalized_question,
                    embedding_service=_embedding_service(),
                )
                payload = _answer_to_payload(generated)
            except Exception as error:
                payload = {
                    "role": "assistant",
                    "error": _friendly_error(error),
                }

        _render_assistant_payload(payload)

    history.append(payload)


def _answer_to_payload(
    answer: GeneratedAnswer,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": answer.answer,
        "insufficient_context": answer.insufficient_context,
        "citations": [
            asdict(citation)
            for citation in answer.citations
        ],
    }


def _render_assistant_payload(
    payload: dict[str, Any],
) -> None:
    if error := payload.get("error"):
        st.error(error, icon="🚫")
        return

    answer = payload.get("content", "")

    if payload.get("insufficient_context"):
        st.warning(answer, icon="ℹ️")
    else:
        st.write(answer)

    citations = payload.get("citations", [])

    if citations:
        st.caption(
            f"Odpowiedź oparta na {len(citations)} "
            f"{'źródle' if len(citations) == 1 else 'źródłach'}."
        )

        for citation in citations:
            _render_citation(citation)


def _render_citation(citation: dict[str, Any]) -> None:
    source_id = citation["source_id"]
    document = citation["document_name"]

    with st.expander(f"Źródło {source_id}: {document}"):
        page_number = citation.get("page_number")
        page_display = (
            str(page_number)
            if page_number is not None
            else "nie dotyczy"
        )
        st.caption(
            f"Strona: {page_display} · "
            f"Dopasowanie: {citation['score']:.3f}"
        )
        st.write(citation["excerpt"])

        with st.popover("Szczegóły techniczne"):
            st.code(citation["chunk_id"], language=None)


def _friendly_error(error: Exception) -> str:
    message = str(error)
    normalized = message.casefold()

    if isinstance(error, EmptyCorpusError):
        return message

    if (
        "already accessed" in normalized
        or "alreadylocked" in normalized
    ):
        return (
            "Lokalna baza Qdrant jest używana przez inne polecenie. "
            "Zakończ tryb ask lub ewaluację i spróbuj ponownie."
        )

    if "connection" in normalized or "ollama" in normalized:
        return (
            "Nie udało się połączyć z Ollama. Sprawdź, czy Ollama "
            f"działa i czy model {config.ollama_model} jest dostępny."
        )

    return f"Nie udało się przygotować odpowiedzi: {message}"


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 860px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 0.35rem 0.6rem;
            margin-bottom: 0.75rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
