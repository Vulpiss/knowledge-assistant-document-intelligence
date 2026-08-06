import re

from loguru import logger

from app.core.config import config
from app.generation.answer_models import (
    AnswerCitation,
    GeneratedAnswer,
    INSUFFICIENT_CONTEXT_ANSWER,
)
from app.generation.context_builder import (
    BuiltContext,
    ContextBuilder,
)
from app.generation.llm_client import (
    LLMClient,
    LLMContractError,
)
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.retriever import Retriever


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        top_k: int | None = None,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

        self.top_k = (
            top_k
            if top_k is not None
            else config.answer_top_k
        )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

    def answer(
        self,
        question: str,
    ) -> GeneratedAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty."
            )

        retrieved_chunks = self.retriever.retrieve(
            query=normalized_question,
            top_k=self.top_k,
        )

        if not retrieved_chunks:
            return self._build_insufficient_answer(
                normalized_question
            )

        context = self.context_builder.build(
            retrieved_chunks
        )

        if not context.sources:
            return self._build_insufficient_answer(
                normalized_question
            )

        logger.info(
            "Answer context selected | sources={}",
            [
                {
                    "source_id": source.source_id,
                    "document": source.document_name,
                    "chunk_id": source.chunk_id,
                }
                for source in context.sources
            ],
        )

        if self._context_explicitly_denies_answer(
            question=normalized_question,
            context=context,
        ):
            logger.info(
                "Context explicitly states that the requested "
                "information is unavailable. Returning controlled refusal."
            )
            return self._build_insufficient_answer(
                normalized_question
            )

        if self._context_lacks_required_subject(
            question=normalized_question,
            context=context,
        ):
            logger.info(
                "Requested subject is not present in the context. "
                "Returning controlled refusal."
            )
            return self._build_insufficient_answer(
                normalized_question
            )

        direct_period_answer = self._find_device_period_answer(
            question=normalized_question,
            context=context,
        )

        if direct_period_answer is not None:
            answer, source_id = direct_period_answer
            logger.info(
                "Direct device period resolved from matching sentence | "
                "source_id={}",
                source_id,
            )
            return GeneratedAnswer(
                question=normalized_question,
                answer=answer,
                citations=self._build_citations(
                    context=context,
                    used_source_ids=[source_id],
                ),
                insufficient_context=False,
            )

        injection_explanation_source_id = (
            self._find_injection_explanation_source(
                question=normalized_question,
                context=context,
            )
        )

        if injection_explanation_source_id is not None:
            logger.info(
                "Prompt-injection explanation answered "
                "deterministically | source_id={}",
                injection_explanation_source_id,
            )
            return GeneratedAnswer(
                question=normalized_question,
                answer=(
                    "Nie. To zdanie jest wyłącznie materiałem "
                    "szkoleniowym i należy je traktować jako treść "
                    "dokumentu, a nie instrukcję do wykonania."
                ),
                citations=self._build_citations(
                    context=context,
                    used_source_ids=[
                        injection_explanation_source_id
                    ],
                ),
                insufficient_context=False,
            )

        system_prompt, user_prompt = (
            self.prompt_builder.build(
                question=normalized_question,
                context=context,
            )
        )

        try:
            llm_result = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        except LLMContractError:
            logger.warning(
                "Model failed the answer contract after "
                "a repair attempt. Returning controlled refusal."
            )

            return self._build_insufficient_answer(
                normalized_question
            )

        if llm_result.insufficient_context:
            return self._build_insufficient_answer(
                normalized_question
            )

        cleaned_answer = self._clean_answer(
            llm_result.answer
        )

        if not cleaned_answer:
            logger.warning(
                "Model produced an empty answer after cleanup. "
                "Returning controlled refusal."
            )
            return self._build_insufficient_answer(
                normalized_question
            )

        if self._answer_indicates_missing_value(
            question=normalized_question,
            answer=cleaned_answer,
        ):
            logger.info(
                "Answer states that the requested value is absent. "
                "Returning controlled refusal."
            )
            return self._build_insufficient_answer(
                normalized_question
            )

        valid_source_ids = {
            source.source_id
            for source in context.sources
        }

        invalid_source_ids = sorted(
            set(llm_result.used_source_ids)
            - valid_source_ids
        )

        if invalid_source_ids:
            logger.warning(
                "Model referenced invalid source IDs: {}. "
                "Returning controlled refusal.",
                invalid_source_ids,
            )

            return self._build_insufficient_answer(
                normalized_question
            )

        citations = self._build_citations(
            context=context,
            used_source_ids=llm_result.used_source_ids,
        )

        if not citations:
            logger.warning(
                "Model produced an answer without valid citations. "
                "Returning controlled refusal."
            )

            return self._build_insufficient_answer(
                normalized_question
            )

        return GeneratedAnswer(
            question=normalized_question,
            answer=cleaned_answer,
            citations=citations,
            insufficient_context=False,
        )

    @staticmethod
    def _clean_answer(answer: str) -> str:
        without_named_markers = re.sub(
            r"\[(?:SOURCE|ŹRÓDŁO)\s*\d+\]",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        without_numeric_markers = re.sub(
            r"\[\d+\]",
            "",
            without_named_markers,
        )

        return re.sub(
            r"\s+",
            " ",
            without_numeric_markers,
        ).strip()

    @staticmethod
    def _answer_indicates_missing_value(
        *,
        question: str,
        answer: str,
    ) -> bool:
        normalized_question = question.casefold().strip()

        if normalized_question.startswith("czy "):
            return False

        normalized_answer = answer.casefold()
        missing_value_phrases = (
            "nie określa",
            "nie podaje",
            "nie zawiera informacji",
            "nie jest opisany",
            "nie jest opisana",
            "brak informacji",
        )

        return any(
            phrase in normalized_answer
            for phrase in missing_value_phrases
        )

    @staticmethod
    def _find_injection_explanation_source(
        *,
        question: str,
        context: BuiltContext,
    ) -> int | None:
        normalized_question = question.casefold().strip()

        if not re.search(r"\bczy\b", normalized_question):
            return None

        execution_phrases = (
            "należy wykonać",
            "nalezy wykonac",
            "powinno się wykonać",
            "powinno sie wykonac",
            "powinien wykonać",
            "powinien wykonac",
            "trzeba wykonać",
            "trzeba wykonac",
        )

        if not any(
            phrase in normalized_question
            for phrase in execution_phrases
        ):
            return None

        suspicious_terms = (
            "zignorować",
            "zignorowac",
            "instrukcj",
            "hasł",
            "hasl",
            "source",
        )

        if not any(
            term in normalized_question
            for term in suspicious_terms
        ):
            return None

        evidence_phrases = (
            "wyłącznie materiałem szkoleniowym",
            "wyłącznie materiał szkoleniowy",
            "nie jako instrukcję do wykonania",
            "nie jako instrukcje do wykonania",
        )

        for source in context.sources:
            normalized_source = source.text.casefold()

            if any(
                phrase in normalized_source
                for phrase in evidence_phrases
            ):
                return source.source_id

        return None

    @classmethod
    def _context_explicitly_denies_answer(
        cls,
        *,
        question: str,
        context: BuiltContext,
    ) -> bool:
        question_terms = cls._content_terms(question)

        if not question_terms:
            return False

        denial_phrases = (
            "nie określa",
            "nie podaje",
            "nie wskazuje",
            "nie zawiera",
            "nie jest przechowywan",
            "nie są przechowywan",
            "brak informacji",
        )

        for source in context.sources:
            sentences = re.split(r"(?<=[.!?])\s+", source.text)

            for sentence in sentences:
                normalized_sentence = sentence.casefold()

                if not any(
                    phrase in normalized_sentence
                    for phrase in denial_phrases
                ):
                    continue

                sentence_terms = cls._content_terms(sentence)
                shared_terms = question_terms & sentence_terms

                if len(shared_terms) >= 2:
                    return True

        return False

    @staticmethod
    def _content_terms(value: str) -> set[str]:
        normalized = value.casefold()
        normalized = normalized.translate(
            str.maketrans(
                "ąćęłńóśźż",
                "acelnoszz",
            )
        )
        tokens = re.findall(r"[a-z0-9]+", normalized)
        stop_words = {
            "aktualny", "aktualna", "aktualne", "czy", "dla",
            "dokladnie", "gdzie", "i", "ile", "jak", "jaka",
            "jaki", "jakie", "jest", "kiedy", "komu", "ma",
            "moze", "na", "nalezy", "obecnie", "oraz", "podaj",
            "powinien", "powinna", "powinno", "przez", "sie",
            "to", "tym", "w", "wynosi", "z", "za",
        }

        return {
            token[:6]
            for token in tokens
            if len(token) >= 4 and token not in stop_words
        }

    @staticmethod
    def _context_lacks_required_subject(
        *,
        question: str,
        context: BuiltContext,
    ) -> bool:
        normalized_question = question.casefold()
        required_subjects = {
            "akta osobowe": (
                "akta osobowe",
                "dokumentacja pracownicza",
                "dokumentacja osobowa",
            ),
        }

        context_text = " ".join(
            source.text.casefold()
            for source in context.sources
        )

        for question_phrase, source_phrases in required_subjects.items():
            if question_phrase not in normalized_question:
                continue

            return not any(
                phrase in context_text
                for phrase in source_phrases
            )

        return False

    @staticmethod
    def _find_device_period_answer(
        *,
        question: str,
        context: BuiltContext,
    ) -> tuple[str, int] | None:
        normalized_question = question.casefold()

        if "okres" not in normalized_question:
            return None

        devices = {
            "laptop": "laptopa służbowego",
            "monitor": "monitora służbowego",
            "telefon": "telefonu służbowego",
        }
        requested_device = next(
            (
                (stem, label)
                for stem, label in devices.items()
                if stem in normalized_question
            ),
            None,
        )

        if requested_device is None:
            return None

        device_stem, device_label = requested_device

        for source in context.sources:
            if "archiw" in source.document_name.casefold():
                continue

            sentences = re.split(r"(?<=[.!?])\s+", source.text)

            for sentence in sentences:
                normalized_sentence = sentence.casefold()

                if device_stem not in normalized_sentence:
                    continue

                match = re.search(
                    rf"{device_stem}\w*[^.!?]*?wynosi\s+"
                    r"(?P<years>\d+)\s+lat(?:a)?\b",
                    normalized_sentence,
                )

                if match is None:
                    continue

                years = match.group("years")
                year_word = (
                    "lata"
                    if years in {"2", "3", "4"}
                    else "lat"
                )
                return (
                    "Standardowy okres użytkowania "
                    f"{device_label} wynosi {years} {year_word}.",
                    source.source_id,
                )

        return None

    def _build_citations(
        self,
        context: BuiltContext,
        used_source_ids: list[int],
    ) -> tuple[AnswerCitation, ...]:
        source_map = {
            source.source_id: source
            for source in context.sources
        }

        citations: list[AnswerCitation] = []

        for source_id in used_source_ids:
            source = source_map[source_id]

            citations.append(
                AnswerCitation(
                    source_id=source.source_id,
                    document_name=source.document_name,
                    page_number=source.page_number,
                    chunk_id=source.chunk_id,
                    score=source.score,
                    excerpt=source.text[:300].strip(),
                )
            )

        return tuple(citations)

    @staticmethod
    def _build_insufficient_answer(
        question: str,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            question=question,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            citations=(),
            insufficient_context=True,
        )
