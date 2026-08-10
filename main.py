import argparse
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from typing import cast

from loguru import logger

from app.core.config import (
    CORPUS_PROFILES,
    EVALUATION_CORPORA,
    CorpusName,
    config,
)
from app.core.logger import setup_logger


CommandHandler = Callable[[Namespace], None]


def run_index_command(
    args: Namespace,
) -> None:
    from app.services.indexing_service import (
        IndexingService,
    )

    indexing_service = IndexingService()

    summary = indexing_service.index(
        rebuild=args.rebuild
    )

    print()
    print("=" * 70)
    print("INDEXING COMPLETED")
    print("=" * 70)
    print(f"Corpus: {config.active_corpus}")
    print(f"Kolekcja: {config.qdrant_collection}")
    print(f"Dokumenty: {summary.documents}")
    print(
        f"Przetworzone jednostki: "
        f"{summary.processed_units}"
    )
    print(f"Chunki: {summary.chunks}")
    print(
        f"Zapisane w tym przebiegu: "
        f"{summary.stored_points}"
    )
    print(
        f"Wszystkie punkty w kolekcji: "
        f"{summary.total_points}"
    )


def run_ask_command(
    args: Namespace,
) -> None:
    from app.embeddings.embedding_service import (
        EmbeddingService,
    )
    from app.generation.answer_service import (
        AnswerService,
    )
    from app.generation.context_builder import (
        ContextBuilder,
    )
    from app.generation.llm_client import (
        OllamaLLMClient,
    )
    from app.generation.prompt_builder import (
        PromptBuilder,
    )
    from app.interfaces.cli import run_answer_debug
    from app.retrieval.retriever import Retriever
    from app.storage.vector_store import (
        QdrantVectorStore,
    )

    vector_store = QdrantVectorStore()

    try:
        total_points = vector_store.count_points()

        if total_points == 0:
            print()
            print(
                "Kolekcja Qdrant jest pusta: "
                f"{config.qdrant_collection}"
            )
            print("Najpierw uruchom:")
            print(
                "python main.py index "
                f"--corpus {config.active_corpus} --rebuild"
            )
            return

        logger.info(
            "Existing index loaded | collection={} | points={}",
            config.qdrant_collection,
            total_points,
        )

        embedding_service = EmbeddingService()

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        answer_service = AnswerService(
            retriever=retriever,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm_client=OllamaLLMClient(),
        )

        run_answer_debug(
            answer_service=answer_service
        )

    finally:
        vector_store.close()


def run_evaluate_answers_command(
    args: Namespace,
) -> None:
    from app.evaluation.answer_eval import (
        run_answer_evaluation,
    )

    run_answer_evaluation(
        dataset_path=config.answer_eval_file,
        report_path=config.answer_eval_results_file,
    )


def run_evaluate_command(
    args: Namespace,
) -> None:
    from app.evaluation.retrieval_eval import (
        run_retrieval_evaluation,
    )

    run_retrieval_evaluation(
        dataset_path=config.retrieval_eval_file
    )


def run_release_check_command(
    args: Namespace,
) -> None:
    from app.evaluation.release_gate import (
        run_release_check,
    )

    run_release_check(
        baseline_path=args.baseline,
        report_path=args.report,
    )


def add_corpus_arguments(
    command_parser: ArgumentParser,
    *,
    include_documents_dir: bool = False,
    include_eval_file: bool = False,
    include_results_file: bool = False,
    corpus_choices: tuple[CorpusName, ...] | None = None,
) -> None:
    choices = corpus_choices or tuple(CORPUS_PROFILES)

    command_parser.add_argument(
        "--corpus",
        choices=choices,
        default="v1",
        help=(
            "Wybiera spójny katalog dokumentów i kolekcję "
            "Qdrant. V1 i V2 mają także dane ewaluacyjne "
            "(domyślnie: v1)."
        ),
    )
    command_parser.add_argument(
        "--collection",
        help=(
            "Opcjonalnie nadpisuje nazwę kolekcji Qdrant "
            "wybranego corpusu."
        ),
    )

    if include_documents_dir:
        command_parser.add_argument(
            "--documents-dir",
            type=Path,
            help=(
                "Opcjonalnie nadpisuje katalog dokumentów "
                "wybranego corpusu."
            ),
        )

    if include_eval_file:
        command_parser.add_argument(
            "--eval-file",
            type=Path,
            help=(
                "Opcjonalnie nadpisuje plik danych "
                "dla bieżącej ewaluacji."
            ),
        )

    if include_results_file:
        command_parser.add_argument(
            "--results-file",
            type=Path,
            help=(
                "Opcjonalnie nadpisuje plik raportu "
                "ewaluacji odpowiedzi."
            ),
        )


def configure_command(args: Namespace) -> None:
    if args.command == "release-check":
        return

    corpus_name = cast(CorpusName, args.corpus)

    config.activate_corpus(
        corpus_name,
        raw_documents_dir=getattr(
            args,
            "documents_dir",
            None,
        ),
        qdrant_collection=args.collection,
    )

    eval_file = getattr(args, "eval_file", None)

    if args.command == "evaluate" and eval_file:
        config.retrieval_eval_file = eval_file

    if args.command == "evaluate-answers":
        if eval_file:
            config.answer_eval_file = eval_file
            config.answer_eval_results_file = (
                eval_file.with_name(
                    f"{eval_file.stem}_results.json"
                )
            )

        results_file = getattr(
            args,
            "results_file",
            None,
        )

        if results_file:
            config.answer_eval_results_file = results_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AI Knowledge Assistant",
        description=(
            "Document Intelligence system with "
            "indexing, retrieval and grounded answers."
        ),
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help=(
            "Przetwarza dokumenty i zapisuje "
            "embeddingi w Qdrant."
        ),
    )
    add_corpus_arguments(
        index_parser,
        include_documents_dir=True,
    )
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Usuwa wybraną kolekcję Qdrant "
            "i buduje jej indeks od nowa."
        ),
    )
    index_parser.set_defaults(
        handler=run_index_command
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help=(
            "Uruchamia tryb zadawania pytań "
            "do istniejącego indeksu."
        ),
    )
    add_corpus_arguments(ask_parser)
    ask_parser.set_defaults(
        handler=run_ask_command
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help=(
            "Uruchamia automatyczną ewaluację retrieval."
        ),
    )
    add_corpus_arguments(
        evaluate_parser,
        include_eval_file=True,
        corpus_choices=EVALUATION_CORPORA,
    )
    evaluate_parser.set_defaults(
        handler=run_evaluate_command
    )

    answer_evaluate_parser = subparsers.add_parser(
        "evaluate-answers",
        help=(
            "Uruchamia automatyczną ewaluację "
            "odpowiedzi, cytowań i odmów."
        ),
    )
    add_corpus_arguments(
        answer_evaluate_parser,
        include_eval_file=True,
        include_results_file=True,
        corpus_choices=EVALUATION_CORPORA,
    )
    answer_evaluate_parser.set_defaults(
        handler=run_evaluate_answers_command
    )

    release_check_parser = subparsers.add_parser(
        "release-check",
        help=(
            "Uruchamia bramkę wydania dla corpusów v1 i v2 "
            "oraz wykrywa regresje względem baseline."
        ),
    )
    release_check_parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "data/eval/release_baseline_v1.json"
        ),
        help="Plik z zamrożonymi progami jakości.",
    )
    release_check_parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/eval/release_check_results.json"
        ),
        help="Plik wynikowy bramki wydania.",
    )
    release_check_parser.set_defaults(
        handler=run_release_check_command
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_command(args)
    setup_logger()

    handler: CommandHandler = args.handler

    logger.info(
        "Starting {} | command={} | environment={} | "
        "corpus={} | documents_dir={} | collection={}",
        config.app_name,
        args.command,
        config.app_env,
        config.active_corpus,
        config.raw_documents_dir,
        config.qdrant_collection,
    )

    try:
        handler(args)

    except KeyboardInterrupt:
        logger.warning(
            "Command interrupted by user."
        )

    except Exception as error:
        logger.exception(
            "Command failed | command={}",
            args.command,
        )

        print()
        print(f"Wystąpił błąd: {error}")

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
