from __future__ import annotations

import sys
import os
import threading
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
    delete_production_document,
    get_corpus_status,
    get_ollama_status,
    list_production_documents,
    pull_ollama_model,
    rebuild_production_index,
    save_production_document,
)


APP_TITLE = "Knowledge Assistant"
HISTORY_KEY = "knowledge_assistant_histories"
DOCUMENT_NOTICE_KEY = "knowledge_assistant_document_notice"
UPLOAD_VERSION_KEY = "knowledge_assistant_upload_version"
SELECTED_DOCUMENT_KEY = "knowledge_assistant_selected_document"
DELETE_CONFIRMATION_KEY = "knowledge_assistant_delete_confirmation"

CORPUS_OPTIONS = (
    ("production",)
    if config.app_env == "production"
    else ("production", "v2", "v1")
)
CORPUS_LABELS = {
    "production": "MOJE DOKUMENTY",
    "v2": "V2 — TEST ZAAWANSOWANY",
    "v1": "V1 — TEST PODSTAWOWY",
}

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

    if UPLOAD_VERSION_KEY not in st.session_state:
        st.session_state[UPLOAD_VERSION_KEY] = 0


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
            options=CORPUS_OPTIONS,
            index=0,
            format_func=lambda value: CORPUS_LABELS[value],
            help=(
                "Moje dokumenty to prywatna baza użytkowa. "
                "V1 i V2 pozostają zestawami testowymi."
            ),
        )

        profile = CORPUS_PROFILES[corpus]
        st.caption(f"Kolekcja: `{profile.qdrant_collection}`")

        if corpus == "production":
            _render_document_manager()

        try:
            status = get_corpus_status(corpus)

            st.caption(
                f"Dokumenty: {status.documents} · "
                f"Chunki: {status.points}"
            )

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
        _render_ollama_status()
        st.divider()

        if st.button(
            "Wyczyść rozmowę",
            use_container_width=True,
        ):
            st.session_state[HISTORY_KEY][corpus] = []
            st.rerun()

        if config.app_env == "production":
            if st.button(
                "Zamknij aplikację",
                use_container_width=True,
            ):
                st.success("Aplikacja jest zamykana…")
                threading.Timer(0.5, lambda: os._exit(0)).start()

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

    if corpus == "production":
        st.info(
            "Korzystasz z prywatnej bazy. Dodaj dokumenty w panelu "
            "po lewej, przebuduj indeks i zadaj własne pytanie."
        )
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


@st.cache_data(ttl=10, show_spinner=False)
def _cached_ollama_status():
    return get_ollama_status()


def _render_ollama_status() -> None:
    status = _cached_ollama_status()

    if status.available and status.model_available:
        st.success(status.message, icon="✅")
    elif status.available:
        st.warning(status.message, icon="⚠️")

        if st.button(
            f"Pobierz model {config.ollama_model}",
            use_container_width=True,
        ):
            _download_ollama_model()
    else:
        st.error(status.message, icon="🚫")
        st.link_button(
            "Zainstaluj Ollama",
            "https://ollama.com/download/windows",
            use_container_width=True,
        )


def _download_ollama_model() -> None:
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    try:
        for update in pull_ollama_model():
            status_placeholder.caption(update.status)

            if update.fraction is not None:
                progress_bar.progress(update.fraction)
    except Exception as error:
        st.error(
            f"Nie udało się pobrać modelu: {error}",
            icon="🚫",
        )
        return

    progress_bar.progress(1.0)
    status_placeholder.caption("Model został pobrany.")
    _cached_ollama_status.clear()
    st.success(
        f"Model {config.ollama_model} jest gotowy.",
        icon="✅",
    )


def _render_document_manager() -> None:
    st.subheader("Moje dokumenty")
    _render_document_notice()

    uploaded_files = st.file_uploader(
        "Dodaj pliki TXT, PDF lub DOCX",
        type=("txt", "pdf", "docx"),
        accept_multiple_files=True,
        key=(
            "production_upload_"
            f"{st.session_state[UPLOAD_VERSION_KEY]}"
        ),
        help="Maksymalny rozmiar pojedynczego pliku: 20 MB.",
    )

    if st.button(
        "Zapisz dokumenty",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        saved_names: list[str] = []
        errors: list[str] = []

        for uploaded_file in uploaded_files or []:
            try:
                saved = save_production_document(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                saved_names.append(saved.name)
            except Exception as error:
                errors.append(f"{uploaded_file.name}: {error}")

        if saved_names:
            message = (
                f"Zapisano {len(saved_names)} dokumentów. "
                "Przebuduj indeks, aby uwzględnić je w odpowiedziach."
            )
            notice_type = "success" if not errors else "warning"
        else:
            message = "Nie zapisano żadnego dokumentu."
            notice_type = "error"

        if errors:
            message += "\n\n" + "\n".join(
                f"- {error}" for error in errors
            )

        _set_document_notice(notice_type, message)
        st.session_state[UPLOAD_VERSION_KEY] += 1
        st.rerun()

    documents = list_production_documents()

    if documents:
        st.caption(f"Zapisane dokumenty: {len(documents)}")

        for document in documents:
            st.write(
                f"• `{document.name}` "
                f"({_format_file_size(document.size_bytes)})"
            )

        selected_document = st.selectbox(
            "Dokument do usunięcia",
            options=[document.name for document in documents],
            index=None,
            placeholder="Wybierz dokument",
            key=SELECTED_DOCUMENT_KEY,
            on_change=_reset_delete_confirmation,
        )
        deletion_confirmed = st.checkbox(
            "Potwierdzam usunięcie wybranego dokumentu",
            disabled=selected_document is None,
            key=DELETE_CONFIRMATION_KEY,
        )

        if st.button(
            "Usuń dokument",
            use_container_width=True,
            disabled=(
                selected_document is None
                or not deletion_confirmed
            ),
        ):
            try:
                deleted = delete_production_document(
                    selected_document
                )
            except Exception as error:
                _set_document_notice("error", str(error))
            else:
                if deleted:
                    _set_document_notice(
                        "success",
                        "Dokument został usunięty. Przebuduj indeks, "
                        "aby usunąć jego treść z odpowiedzi.",
                    )
                else:
                    _set_document_notice(
                        "warning",
                        "Dokument nie istnieje.",
                    )

            st.rerun()
    else:
        st.caption("Nie dodano jeszcze własnych dokumentów.")

    rebuild_label = (
        "Przebuduj indeks"
        if documents
        else "Wyczyść pusty indeks"
    )

    if st.button(
        rebuild_label,
        use_container_width=True,
        type="primary",
        help=(
            "Tworzy indeks od nowa wyłącznie dla prywatnej "
            "kolekcji production."
        ),
    ):
        try:
            with st.spinner(
                "Przetwarzam dokumenty i przebudowuję indeks…"
            ):
                summary = rebuild_production_index(
                    embedding_service=_embedding_service(),
                )
        except Exception as error:
            _set_document_notice(
                "error",
                _friendly_error(error),
            )
        else:
            if summary.documents:
                message = (
                    "Indeks jest gotowy: "
                    f"{summary.documents} dokumentów, "
                    f"{summary.total_points} chunków."
                )
            else:
                message = "Pusty indeks został wyczyszczony."

            _set_document_notice("success", message)

        st.rerun()


def _render_document_notice() -> None:
    notice = st.session_state.pop(DOCUMENT_NOTICE_KEY, None)

    if not notice:
        return

    notice_type, message = notice
    renderer = getattr(st, notice_type, st.info)
    renderer(message)


def _set_document_notice(
    notice_type: str,
    message: str,
) -> None:
    st.session_state[DOCUMENT_NOTICE_KEY] = (
        notice_type,
        message,
    )


def _reset_delete_confirmation() -> None:
    st.session_state[DELETE_CONFIRMATION_KEY] = False


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"

    size_kilobytes = size_bytes / 1024

    if size_kilobytes < 1024:
        return f"{size_kilobytes:.1f} KB"

    return f"{size_kilobytes / 1024:.1f} MB"


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
