import unittest

from pydantic import ValidationError

from app.generation.answer_models import (
    INSUFFICIENT_CONTEXT_ANSWER,
    LLMAnswerPayload,
)


class LLMAnswerPayloadTests(unittest.TestCase):
    def test_accepts_grounded_answer_and_deduplicates_sources(
        self,
    ) -> None:
        payload = LLMAnswerPayload(
            answer="  Hasło musi mieć 14 znaków.  ",
            used_source_ids=[1, 1, 2],
            insufficient_context=False,
        )

        self.assertEqual(
            payload.answer,
            "Hasło musi mieć 14 znaków.",
        )
        self.assertEqual(payload.used_source_ids, [1, 2])

    def test_accepts_exact_controlled_refusal(self) -> None:
        payload = LLMAnswerPayload(
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            used_source_ids=[],
            insufficient_context=True,
        )

        self.assertTrue(payload.insufficient_context)

    def test_rejects_refusal_with_sources(self) -> None:
        with self.assertRaises(ValidationError):
            LLMAnswerPayload(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                used_source_ids=[1],
                insufficient_context=True,
            )

    def test_rejects_grounded_answer_without_source(self) -> None:
        with self.assertRaises(ValidationError):
            LLMAnswerPayload(
                answer="Hasło musi mieć 14 znaków.",
                used_source_ids=[],
                insufficient_context=False,
            )

    def test_rejects_unknown_json_fields(self) -> None:
        with self.assertRaises(ValidationError):
            LLMAnswerPayload.model_validate(
                {
                    "answer": "Odpowiedź.",
                    "used_source_ids": [1],
                    "insufficient_context": False,
                    "unexpected": "value",
                }
            )


if __name__ == "__main__":
    unittest.main()
