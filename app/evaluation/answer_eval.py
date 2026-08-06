import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from loguru import logger

from app.core.config import config
from app.core.logger import setup_logger
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.answer_eval_dataset import (
    AnswerEvalCase,
    load_answer_eval_cases,
)
from app.generation.answer_service import AnswerService
from app.generation.context_builder import ContextBuilder
from app.generation.llm_client import OllamaLLMClient
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.retriever import Retriever
from app.storage.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class AnswerEvalResult:
    case_id: str
    question: str
    should_refuse: bool

    actual_refusal: bool
    answer: str

    accepted_answer_phrases: tuple[str, ...]
    expected_documents: tuple[str, ...]
    citation_documents: tuple[str, ...]

    answer_match: bool | None
    citation_match: bool | None

    passed: bool
    latency_seconds: float
    error: str | None


@dataclass(frozen=True)
class AnswerEvalSummary:
    total_cases: int
    passed_cases: int
    failed_cases: int

    answerable_cases: int
    refusal_cases: int

    overall_pass_rate: float
    answer_accuracy: float
    citation_accuracy: float
    refusal_accuracy: float
    grounded_answer_rate: float
    hallucination_rate: float
    execution_error_rate: float

    average_latency_seconds: float


class AnswerEvaluator:
    def __init__(
        self,
        answer_service: AnswerService,
    ) -> None:
        self.answer_service = answer_service

    def evaluate(
        self,
        eval_cases: list[AnswerEvalCase],
    ) -> list[AnswerEvalResult]:
        results: list[AnswerEvalResult] = []

        for index, eval_case in enumerate(
            eval_cases,
            start=1,
        ):
            logger.info(
                "Evaluating answer case {}/{} | case_id={}",
                index,
                len(eval_cases),
                eval_case.case_id,
            )

            result = self.evaluate_case(
                eval_case
            )

            results.append(result)

        return results

    def evaluate_case(
        self,
        eval_case: AnswerEvalCase,
    ) -> AnswerEvalResult:
        started_at = perf_counter()

        try:
            generated_answer = self.answer_service.answer(
                eval_case.question
            )

        except Exception as error:
            latency_seconds = (
                perf_counter() - started_at
            )

            logger.exception(
                "Answer evaluation case failed | case_id={}",
                eval_case.case_id,
            )

            return AnswerEvalResult(
                case_id=eval_case.case_id,
                question=eval_case.question,
                should_refuse=eval_case.should_refuse,
                actual_refusal=False,
                answer="",
                accepted_answer_phrases=(
                    eval_case.accepted_answer_phrases
                ),
                expected_documents=(
                    eval_case.expected_documents
                ),
                citation_documents=(),
                answer_match=None,
                citation_match=None,
                passed=False,
                latency_seconds=latency_seconds,
                error=str(error),
            )

        latency_seconds = perf_counter() - started_at

        citation_documents = tuple(
            sorted(
                {
                    citation.document_name
                    for citation
                    in generated_answer.citations
                }
            )
        )

        if eval_case.should_refuse:
            passed = (
                generated_answer.insufficient_context
                and not generated_answer.citations
            )

            answer_match: bool | None = None
            citation_match: bool | None = None

        else:
            answer_match = self._answer_matches(
                answer=generated_answer.answer,
                accepted_phrases=(
                    eval_case.accepted_answer_phrases
                ),
            )

            citation_match = self._citations_match(
                citation_documents=citation_documents,
                expected_documents=(
                    eval_case.expected_documents
                ),
            )

            passed = (
                not generated_answer.insufficient_context
                and answer_match
                and citation_match
            )

        return AnswerEvalResult(
            case_id=eval_case.case_id,
            question=eval_case.question,
            should_refuse=eval_case.should_refuse,
            actual_refusal=(
                generated_answer.insufficient_context
            ),
            answer=generated_answer.answer,
            accepted_answer_phrases=(
                eval_case.accepted_answer_phrases
            ),
            expected_documents=(
                eval_case.expected_documents
            ),
            citation_documents=citation_documents,
            answer_match=answer_match,
            citation_match=citation_match,
            passed=passed,
            latency_seconds=latency_seconds,
            error=None,
        )

    @staticmethod
    def _answer_matches(
        answer: str,
        accepted_phrases: tuple[str, ...],
    ) -> bool:
        normalized_answer = _normalize_text(
            answer
        )

        return any(
            _normalize_text(phrase)
            in normalized_answer
            for phrase in accepted_phrases
        )

    @staticmethod
    def _citations_match(
        citation_documents: tuple[str, ...],
        expected_documents: tuple[str, ...],
    ) -> bool:
        actual_set = set(
            citation_documents
        )

        expected_set = set(
            expected_documents
        )

        return actual_set == expected_set


def calculate_summary(
    results: list[AnswerEvalResult],
) -> AnswerEvalSummary:
    answerable_results = [
        result
        for result in results
        if not result.should_refuse
    ]

    refusal_results = [
        result
        for result in results
        if result.should_refuse
    ]

    passed_cases = sum(
        result.passed
        for result in results
    )

    answer_correct = sum(
        bool(result.answer_match)
        and not result.actual_refusal
        and result.error is None
        for result in answerable_results
    )

    citations_correct = sum(
        bool(result.citation_match)
        and not result.actual_refusal
        and result.error is None
        for result in answerable_results
    )

    grounded_answers = sum(
        bool(result.answer_match)
        and bool(result.citation_match)
        and not result.actual_refusal
        and result.error is None
        for result in answerable_results
    )

    correct_refusals = sum(
        result.actual_refusal
        and not result.citation_documents
        and result.error is None
        for result in refusal_results
    )

    hallucinations = sum(
        not result.actual_refusal
        and result.error is None
        for result in refusal_results
    )

    execution_errors = sum(
        result.error is not None
        for result in results
    )

    total_latency = sum(
        result.latency_seconds
        for result in results
    )

    return AnswerEvalSummary(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=(
            len(results) - passed_cases
        ),
        answerable_cases=len(
            answerable_results
        ),
        refusal_cases=len(
            refusal_results
        ),
        overall_pass_rate=_safe_divide(
            passed_cases,
            len(results),
        ),
        answer_accuracy=_safe_divide(
            answer_correct,
            len(answerable_results),
        ),
        citation_accuracy=_safe_divide(
            citations_correct,
            len(answerable_results),
        ),
        refusal_accuracy=_safe_divide(
            correct_refusals,
            len(refusal_results),
        ),
        grounded_answer_rate=_safe_divide(
            grounded_answers,
            len(answerable_results),
        ),
        hallucination_rate=_safe_divide(
            hallucinations,
            len(refusal_results),
        ),
        execution_error_rate=_safe_divide(
            execution_errors,
            len(results),
        ),
        average_latency_seconds=_safe_divide(
            total_latency,
            len(results),
        ),
    )


def print_detailed_results(
    results: list[AnswerEvalResult],
) -> None:
    print()
    print("=" * 80)
    print("ANSWER EVALUATION — SZCZEGÓŁY")
    print("=" * 80)

    for result in results:
        if result.error:
            status = "ERROR"
        elif result.passed:
            status = "PASS"
        else:
            status = "FAIL"

        print()
        print(f"[{status}] {result.case_id}")
        print(f"Pytanie: {result.question}")
        print(
            f"Oczekiwana odmowa: "
            f"{result.should_refuse}"
        )
        print(
            f"Rzeczywista odmowa: "
            f"{result.actual_refusal}"
        )
        print(
            f"Odpowiedź: {result.answer}"
        )

        if result.answer_match is not None:
            print(
                f"Poprawna fraza: "
                f"{result.answer_match}"
            )

        if result.citation_match is not None:
            print(
                f"Poprawne cytowania: "
                f"{result.citation_match}"
            )

        print(
            f"Oczekiwane dokumenty: "
            f"{list(result.expected_documents)}"
        )
        print(
            f"Cytowane dokumenty: "
            f"{list(result.citation_documents)}"
        )
        print(
            f"Czas: "
            f"{result.latency_seconds:.2f} s"
        )

        if result.error:
            print(
                f"Błąd: {result.error}"
            )


def print_summary(
    summary: AnswerEvalSummary,
) -> None:
    print()
    print("=" * 80)
    print("ANSWER EVALUATION — PODSUMOWANIE")
    print("=" * 80)
    print(
        f"Liczba przypadków: "
        f"{summary.total_cases}"
    )
    print(
        f"PASS: {summary.passed_cases}"
    )
    print(
        f"FAIL: {summary.failed_cases}"
    )
    print(
        f"Overall Pass Rate: "
        f"{summary.overall_pass_rate:.2%}"
    )
    print(
        f"Answer Accuracy: "
        f"{summary.answer_accuracy:.2%}"
    )
    print(
        f"Citation Accuracy: "
        f"{summary.citation_accuracy:.2%}"
    )
    print(
        f"Refusal Accuracy: "
        f"{summary.refusal_accuracy:.2%}"
    )
    print(
        f"Grounded Answer Rate: "
        f"{summary.grounded_answer_rate:.2%}"
    )
    print(
        f"Hallucination Rate: "
        f"{summary.hallucination_rate:.2%}"
    )
    print(
        f"Execution Error Rate: "
        f"{summary.execution_error_rate:.2%}"
    )
    print(
        f"Średni czas odpowiedzi: "
        f"{summary.average_latency_seconds:.2f} s"
    )


def save_report(
    results: list[AnswerEvalResult],
    summary: AnswerEvalSummary,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = {
        "generated_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "model": config.ollama_model,
        "collection": config.qdrant_collection,
        "summary": asdict(summary),
        "results": [
            asdict(result)
            for result in results
        ],
    }

    report_path.write_text(
        json.dumps(
            report_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Answer evaluation report saved: {}",
        report_path,
    )


def run_answer_evaluation(
    dataset_path: Path | None = None,
    report_path: Path | None = None,
) -> AnswerEvalSummary:
    dataset_path = (
        dataset_path
        or config.answer_eval_file
    )

    report_path = (
        report_path
        or config.answer_eval_results_file
    )

    logger.info(
        "Loading answer evaluation dataset: {}",
        dataset_path,
    )

    eval_cases = load_answer_eval_cases(
        dataset_path
    )

    logger.info(
        "Answer evaluation cases loaded: {}",
        len(eval_cases),
    )

    vector_store = QdrantVectorStore()

    try:
        total_points = (
            vector_store.count_points()
        )

        if total_points == 0:
            raise RuntimeError(
                "Kolekcja Qdrant jest pusta. "
                "Najpierw uruchom: "
                "python main.py index"
            )

        logger.info(
            "Answer evaluation index ready | points={}",
            total_points,
        )

        embedding_service = EmbeddingService()

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

        evaluator = AnswerEvaluator(
            answer_service=answer_service
        )

        results = evaluator.evaluate(
            eval_cases
        )

        summary = calculate_summary(
            results
        )

        print_detailed_results(
            results
        )

        print_summary(
            summary
        )

        save_report(
            results=results,
            summary=summary,
            report_path=report_path,
        )

        return summary

    finally:
        vector_store.close()


def _normalize_text(
    value: str,
) -> str:
    normalized_whitespace = re.sub(
        r"\s+",
        " ",
        value,
    )

    return (
        normalized_whitespace
        .strip()
        .casefold()
    )


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    setup_logger()
    run_answer_evaluation()


if __name__ == "__main__":
    main()
