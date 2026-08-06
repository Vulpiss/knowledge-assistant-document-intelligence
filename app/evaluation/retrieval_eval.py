from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.core.config import config
from app.core.logger import setup_logger
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.eval_dataset import (
    RetrievalEvalCase,
    load_retrieval_eval_cases,
)
from app.retrieval.retriever import Retriever
from app.storage.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class RetrievalEvalResult:
    case_id: str
    question: str
    expected_document: str
    document_rank: int | None
    phrase_rank: int | None
    top_document: str | None
    top_score: float | None


@dataclass(frozen=True)
class RetrievalEvalSummary:
    total_cases: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    phrase_recall_at_1: float
    mrr: float


class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Retriever,
        max_k: int = 5,
    ) -> None:
        if max_k <= 0:
            raise ValueError(
                "max_k must be greater than 0."
            )

        self.retriever = retriever
        self.max_k = max_k

    def evaluate_case(
        self,
        eval_case: RetrievalEvalCase,
    ) -> RetrievalEvalResult:
        results = self.retriever.retrieve(
            query=eval_case.question,
            top_k=self.max_k,
        )

        document_rank: int | None = None
        phrase_rank: int | None = None

        expected_phrase = (
            eval_case.expected_phrase.casefold()
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if (
                document_rank is None
                and result.document_name
                == eval_case.expected_document
            ):
                document_rank = rank

            if (
                phrase_rank is None
                and result.document_name
                == eval_case.expected_document
                and expected_phrase
                in result.text.casefold()
            ):
                phrase_rank = rank

        top_document = (
            results[0].document_name
            if results
            else None
        )

        top_score = (
            results[0].score
            if results
            else None
        )

        return RetrievalEvalResult(
            case_id=eval_case.case_id,
            question=eval_case.question,
            expected_document=eval_case.expected_document,
            document_rank=document_rank,
            phrase_rank=phrase_rank,
            top_document=top_document,
            top_score=top_score,
        )

    def evaluate(
        self,
        eval_cases: list[RetrievalEvalCase],
    ) -> list[RetrievalEvalResult]:
        results: list[RetrievalEvalResult] = []

        for index, eval_case in enumerate(
            eval_cases,
            start=1,
        ):
            logger.info(
                "Evaluating retrieval case {}/{} | case_id={}",
                index,
                len(eval_cases),
                eval_case.case_id,
            )

            results.append(
                self.evaluate_case(eval_case)
            )

        return results


def calculate_recall_at_k(
    results: list[RetrievalEvalResult],
    k: int,
) -> float:
    if not results:
        return 0.0

    successful = sum(
        1
        for result in results
        if (
            result.document_rank is not None
            and result.document_rank <= k
        )
    )

    return successful / len(results)


def calculate_phrase_recall_at_k(
    results: list[RetrievalEvalResult],
    k: int,
) -> float:
    if not results:
        return 0.0

    successful = sum(
        1
        for result in results
        if (
            result.phrase_rank is not None
            and result.phrase_rank <= k
        )
    )

    return successful / len(results)


def calculate_mrr(
    results: list[RetrievalEvalResult],
) -> float:
    if not results:
        return 0.0

    reciprocal_rank_sum = sum(
        1 / result.document_rank
        if result.document_rank is not None
        else 0
        for result in results
    )

    return reciprocal_rank_sum / len(results)


def print_detailed_results(
    results: list[RetrievalEvalResult],
) -> None:
    print()
    print("=" * 80)
    print("RETRIEVAL EVALUATION — SZCZEGÓŁY")
    print("=" * 80)

    for result in results:
        status = (
            "PASS"
            if result.document_rank == 1
            else "FAIL"
        )

        score_display = (
            f"{result.top_score:.4f}"
            if result.top_score is not None
            else "brak"
        )

        print()
        print(f"[{status}] {result.case_id}")
        print(f"Pytanie: {result.question}")
        print(
            f"Oczekiwany dokument: "
            f"{result.expected_document}"
        )
        print(
            f"Dokument na pozycji 1: "
            f"{result.top_document}"
        )
        print(
            f"Pozycja właściwego dokumentu: "
            f"{result.document_rank}"
        )
        print(
            f"Pozycja właściwej frazy: "
            f"{result.phrase_rank}"
        )
        print(f"Najlepszy score: {score_display}")


def calculate_summary(
    results: list[RetrievalEvalResult],
) -> RetrievalEvalSummary:
    recall_at_1 = calculate_recall_at_k(
        results,
        1,
    )
    recall_at_3 = calculate_recall_at_k(
        results,
        3,
    )
    recall_at_5 = calculate_recall_at_k(
        results,
        5,
    )

    phrase_recall_at_1 = (
        calculate_phrase_recall_at_k(
            results,
            1,
        )
    )

    mrr = calculate_mrr(results)

    return RetrievalEvalSummary(
        total_cases=len(results),
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        phrase_recall_at_1=phrase_recall_at_1,
        mrr=mrr,
    )


def print_summary(
    summary: RetrievalEvalSummary,
) -> None:
    print()
    print("=" * 80)
    print("RETRIEVAL EVALUATION — PODSUMOWANIE")
    print("=" * 80)
    print(f"Liczba przypadków: {summary.total_cases}")
    print(f"Recall@1: {summary.recall_at_1:.2%}")
    print(f"Recall@3: {summary.recall_at_3:.2%}")
    print(f"Recall@5: {summary.recall_at_5:.2%}")
    print(
        f"Phrase Recall@1: "
        f"{summary.phrase_recall_at_1:.2%}"
    )
    print(f"MRR: {summary.mrr:.4f}")


def run_retrieval_evaluation(
    dataset_path: Path | None = None,
) -> RetrievalEvalSummary:
    dataset_path = (
        dataset_path
        or config.retrieval_eval_file
    )

    logger.info(
        "Loading retrieval evaluation dataset: {}",
        dataset_path,
    )

    eval_cases = load_retrieval_eval_cases(
        dataset_path
    )

    logger.info(
        "Evaluation cases loaded: {}",
        len(eval_cases),
    )

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore()

    try:
        total_points = vector_store.count_points()

        if total_points == 0:
            raise RuntimeError(
                "Kolekcja Qdrant jest pusta. "
                "Najpierw uruchom: python main.py index"
            )

        logger.info(
            "Evaluation collection ready | points={}",
            total_points,
        )

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            top_k=5,
        )

        evaluator = RetrievalEvaluator(
            retriever=retriever,
            max_k=5,
        )

        results = evaluator.evaluate(
            eval_cases
        )

        summary = calculate_summary(results)

        print_detailed_results(results)
        print_summary(summary)

        return summary

    finally:
        vector_store.close()


def main() -> None:
    setup_logger()
    run_retrieval_evaluation()


if __name__ == "__main__":
    main()
