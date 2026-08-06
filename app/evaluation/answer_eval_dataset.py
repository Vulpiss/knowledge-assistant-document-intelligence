import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnswerEvalCase:
    case_id: str
    question: str
    accepted_answer_phrases: tuple[str, ...]
    expected_documents: tuple[str, ...]
    should_refuse: bool


def load_answer_eval_cases(
    dataset_path: Path,
) -> list[AnswerEvalCase]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Answer evaluation dataset not found: {dataset_path}"
        )

    try:
        raw_data = json.loads(
            dataset_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid answer evaluation JSON: {dataset_path}"
        ) from error

    if not isinstance(raw_data, list):
        raise ValueError(
            "Answer evaluation dataset must contain a JSON list."
        )

    cases: list[AnswerEvalCase] = []

    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Answer evaluation case #{index} must be an object."
            )

        eval_case = _build_case(
            item=item,
            index=index,
        )

        cases.append(eval_case)

    if not cases:
        raise ValueError(
            "Answer evaluation dataset cannot be empty."
        )

    return cases


def _build_case(
    item: dict[str, Any],
    index: int,
) -> AnswerEvalCase:
    required_fields = {
        "case_id",
        "question",
        "accepted_answer_phrases",
        "expected_documents",
        "should_refuse",
    }

    missing_fields = required_fields - item.keys()

    if missing_fields:
        raise ValueError(
            f"Answer evaluation case #{index} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    case_id = str(item["case_id"]).strip()
    question = str(item["question"]).strip()
    should_refuse = item["should_refuse"]

    if not case_id:
        raise ValueError(
            f"Answer evaluation case #{index} has empty case_id."
        )

    if not question:
        raise ValueError(
            f"Answer evaluation case #{index} has empty question."
        )

    if not isinstance(should_refuse, bool):
        raise ValueError(
            f"Answer evaluation case #{index}: "
            "should_refuse must be boolean."
        )

    accepted_answer_phrases = _read_string_list(
        value=item["accepted_answer_phrases"],
        field_name="accepted_answer_phrases",
        case_index=index,
    )

    expected_documents = _read_string_list(
        value=item["expected_documents"],
        field_name="expected_documents",
        case_index=index,
    )

    if should_refuse:
        if accepted_answer_phrases:
            raise ValueError(
                f"Refusal case #{index} cannot define "
                "accepted_answer_phrases."
            )

        if expected_documents:
            raise ValueError(
                f"Refusal case #{index} cannot define "
                "expected_documents."
            )

    else:
        if not accepted_answer_phrases:
            raise ValueError(
                f"Answerable case #{index} must define "
                "accepted_answer_phrases."
            )

        if not expected_documents:
            raise ValueError(
                f"Answerable case #{index} must define "
                "expected_documents."
            )

    return AnswerEvalCase(
        case_id=case_id,
        question=question,
        accepted_answer_phrases=accepted_answer_phrases,
        expected_documents=expected_documents,
        should_refuse=should_refuse,
    )


def _read_string_list(
    value: Any,
    field_name: str,
    case_index: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"Answer evaluation case #{case_index}: "
            f"{field_name} must be a list."
        )

    normalized_values = tuple(
        str(item).strip()
        for item in value
        if str(item).strip()
    )

    return normalized_values