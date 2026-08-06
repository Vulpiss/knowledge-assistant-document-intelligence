from dataclasses import dataclass
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


INSUFFICIENT_CONTEXT_ANSWER = (
    "Nie znalazłem wystarczających informacji "
    "w dostarczonych dokumentach."
)


class LLMAnswerPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    answer: str = Field(
        min_length=1,
        description=(
            "Odpowiedź przygotowana wyłącznie na podstawie źródeł. "
            "Przy braku odpowiedzi musi zawierać dokładny tekst odmowy."
        ),
    )

    used_source_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Numery źródeł użytych do przygotowania odpowiedzi. "
            "Lista musi być niepusta, gdy insufficient_context=false, "
            "i pusta, gdy insufficient_context=true."
        ),
    )

    insufficient_context: bool = Field(
        description=(
            "Ustaw true wyłącznie wtedy, gdy źródła nie zawierają "
            "informacji potrzebnej do odpowiedzi."
        ),
    )

    @field_validator("answer")
    @classmethod
    def normalize_answer(
        cls,
        value: str,
    ) -> str:
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "answer cannot be empty."
            )

        return normalized_value

    @field_validator("used_source_ids")
    @classmethod
    def validate_source_ids(
        cls,
        source_ids: list[int],
    ) -> list[int]:
        unique_source_ids: list[int] = []
        seen_source_ids: set[int] = set()

        for source_id in source_ids:
            if source_id <= 0:
                raise ValueError(
                    "used_source_ids must contain "
                    "positive integers."
                )

            if source_id in seen_source_ids:
                continue

            unique_source_ids.append(source_id)
            seen_source_ids.add(source_id)

        return unique_source_ids

    @model_validator(mode="after")
    def validate_answer_contract(
        self,
    ) -> Self:
        if self.insufficient_context:
            if self.used_source_ids:
                raise ValueError(
                    "used_source_ids must be empty when "
                    "insufficient_context is true."
                )

            if self.answer != INSUFFICIENT_CONTEXT_ANSWER:
                raise ValueError(
                    "answer must contain the exact controlled refusal "
                    "when insufficient_context is true."
                )

            return self

        if not self.used_source_ids:
            raise ValueError(
                "used_source_ids must contain at least one source "
                "when insufficient_context is false."
            )

        if self.answer == INSUFFICIENT_CONTEXT_ANSWER:
            raise ValueError(
                "The controlled refusal cannot be returned when "
                "insufficient_context is false."
            )

        return self


@dataclass(frozen=True)
class AnswerCitation:
    source_id: int
    document_name: str
    page_number: int | None
    chunk_id: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    answer: str
    citations: tuple[AnswerCitation, ...]
    insufficient_context: bool