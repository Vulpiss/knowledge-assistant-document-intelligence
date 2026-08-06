import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    question: str
    expected_document: str
    expected_phrase: str


def load_retrieval_eval_cases(
    dataset_path: Path,
) -> list[RetrievalEvalCase]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    try:
        raw_data = json.loads(
            dataset_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON evaluation dataset: {dataset_path}"
        ) from error

    if not isinstance(raw_data, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    cases: list[RetrievalEvalCase] = []

    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Evaluation case #{index} must be an object."
            )

        required_fields = {
            "case_id",
            "question",
            "expected_document",
            "expected_phrase",
        }

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Evaluation case #{index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        eval_case = RetrievalEvalCase(
            case_id=str(item["case_id"]).strip(),
            question=str(item["question"]).strip(),
            expected_document=str(
                item["expected_document"]
            ).strip(),
            expected_phrase=str(
                item["expected_phrase"]
            ).strip(),
        )

        if not all(
            [
                eval_case.case_id,
                eval_case.question,
                eval_case.expected_document,
                eval_case.expected_phrase,
            ]
        ):
            raise ValueError(
                f"Evaluation case #{index} contains empty values."
            )

        cases.append(eval_case)

    if not cases:
        raise ValueError(
            "Evaluation dataset cannot be empty."
        )

    return cases