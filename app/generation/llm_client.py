from typing import Protocol

import requests
from loguru import logger
from pydantic import ValidationError

from app.core.config import config
from app.generation.answer_models import LLMAnswerPayload


ANSWER_FORMAT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        },
        "used_source_ids": {
            "type": "array",
            "items": {
                "type": "integer",
            },
        },
        "insufficient_context": {
            "type": "boolean",
        },
    },
    "required": [
        "answer",
        "used_source_ids",
        "insufficient_context",
    ],
    "additionalProperties": False,
}


def nanoseconds_to_seconds(
    value: object,
) -> float:
    if not isinstance(value, (int, float)):
        return 0.0

    return float(value) / 1_000_000_000


class LLMContractError(RuntimeError):
    """Raised when the model cannot produce a valid answer contract."""


class LLMClient(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMAnswerPayload:
        ...


class OllamaLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        self.base_url = (
            base_url or config.ollama_base_url
        ).rstrip("/")

        self.model_name = (
            model_name or config.ollama_model
        )

        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.ollama_timeout_seconds
        )

        if max_repair_attempts < 0:
            raise ValueError(
                "max_repair_attempts cannot be negative."
            )

        self.max_repair_attempts = max_repair_attempts

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMAnswerPayload:
        endpoint = f"{self.base_url}/api/chat"

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        last_validation_error: ValidationError | None = None

        for attempt in range(
            self.max_repair_attempts + 1
        ):
            content = self._request_content(
                endpoint=endpoint,
                messages=messages,
                repair_attempt=attempt,
            )

            try:
                result = (
                    LLMAnswerPayload.model_validate_json(
                        content
                    )
                )

            except ValidationError as error:
                last_validation_error = error

                if attempt >= self.max_repair_attempts:
                    logger.error(
                        "Ollama failed answer contract validation "
                        "after repair attempts | attempts={}",
                        self.max_repair_attempts,
                    )

                    raise LLMContractError(
                        "Model nie zwrócił odpowiedzi zgodnej "
                        "z kontraktem po próbie naprawczej."
                    ) from error

                logger.warning(
                    "Ollama answer contract invalid. "
                    "Starting repair attempt {}/{}.",
                    attempt + 1,
                    self.max_repair_attempts,
                )

                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": content,
                        },
                        {
                            "role": "user",
                            "content": self._build_repair_prompt(
                                error
                            ),
                        },
                    ]
                )

                continue

            logger.info(
                "Ollama response received | "
                "insufficient_context={} | "
                "sources={} | "
                "repaired={}",
                result.insufficient_context,
                result.used_source_ids,
                attempt > 0,
            )

            return result

        raise LLMContractError(
            "Model nie zwrócił poprawnej odpowiedzi."
        ) from last_validation_error

    def _request_content(
        self,
        endpoint: str,
        messages: list[dict[str, str]],
        repair_attempt: int,
    ) -> str:
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": ANSWER_FORMAT_SCHEMA,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": 128,
                "num_ctx": 4096,
            },
        }

        logger.info(
            "Sending request to Ollama | "
            "model={} | "
            "endpoint={} | "
            "repair_attempt={}",
            self.model_name,
            endpoint,
            repair_attempt,
        )

        try:
            response = requests.post(
                endpoint,
                json=request_body,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()

        except requests.ConnectionError as error:
            logger.exception(
                "Cannot connect to Ollama."
            )

            raise RuntimeError(
                "Nie można połączyć się z Ollamą. "
                "Sprawdź, czy aplikacja Ollama jest uruchomiona."
            ) from error

        except requests.Timeout as error:
            logger.exception(
                "Ollama request timed out."
            )

            raise RuntimeError(
                "Model nie odpowiedział w wyznaczonym czasie."
            ) from error

        except requests.RequestException as error:
            logger.exception(
                "Ollama request failed."
            )

            response_text = (
                error.response.text[:500]
                if error.response is not None
                else str(error)
            )

            raise RuntimeError(
                f"Błąd Ollama API: {response_text}"
            ) from error

        try:
            response_data = response.json()

        except ValueError as error:
            logger.exception(
                "Ollama returned invalid JSON response."
            )

            raise RuntimeError(
                "Ollama zwróciła nieprawidłową odpowiedź."
            ) from error

        if not isinstance(response_data, dict):
            raise RuntimeError(
                "Ollama zwróciła odpowiedź "
                "w nieoczekiwanym formacie."
            )

        logger.info(
            "Ollama generation metrics | "
            "done_reason={} | "
            "prompt_tokens={} | "
            "output_tokens={} | "
            "total_duration={:.2f}s | "
            "prompt_eval_duration={:.2f}s | "
            "eval_duration={:.2f}s",
            response_data.get("done_reason"),
            response_data.get("prompt_eval_count"),
            response_data.get("eval_count"),
            nanoseconds_to_seconds(
                response_data.get("total_duration")
            ),
            nanoseconds_to_seconds(
                response_data.get(
                    "prompt_eval_duration"
                )
            ),
            nanoseconds_to_seconds(
                response_data.get("eval_duration")
            ),
        )

        message = response_data.get(
            "message"
        )

        if not isinstance(message, dict):
            raise RuntimeError(
                "Ollama nie zwróciła poprawnego "
                "obiektu message."
            )

        content = message.get(
            "content"
        )

        if not isinstance(content, str):
            raise RuntimeError(
                "Ollama zwróciła treść odpowiedzi "
                "w nieoczekiwanym formacie."
            )

        normalized_content = content.strip()

        if not normalized_content:
            raise RuntimeError(
                "Ollama zwróciła pustą odpowiedź."
            )

        return normalized_content

    @staticmethod
    def _build_repair_prompt(
        error: ValidationError,
    ) -> str:
        validation_details = str(error)[:1500]

        return f"""
Poprzednia odpowiedź narusza wymagany kontrakt JSON.

Napraw ją, korzystając z tego samego pytania i tych samych źródeł.

ZASADY KONTRAKTU:

Błąd walidacji formatu NIE oznacza, że brakuje informacji w źródłach.
Nie zmieniaj odpowiedzi na odmowę tylko po to, aby przejść walidację.
Przeczytaj ponownie pierwotne pytanie i źródła.

Jeżeli poprzednia odpowiedź zawierała fakt wsparty źródłem, zachowaj ten
fakt i napraw pola JSON. Gdy brakowało used_source_ids, dodaj właściwy,
istniejący numer SOURCE zamiast ustawiać insufficient_context na true.

Jeżeli dokument obowiązujący jest sprzeczny z archiwalnym, wybierz dokument
obowiązujący. Nie traktuj takiej sprzeczności jako braku kontekstu.

1. Jeżeli źródła zawierają odpowiedź:
   - insufficient_context musi być false,
   - used_source_ids musi zawierać co najmniej jeden istniejący
     numer SOURCE, który bezpośrednio wspiera odpowiedź,
   - answer musi odpowiadać na pytanie.

2. Jeżeli źródła nie zawierają odpowiedzi:
   - insufficient_context musi być true,
   - used_source_ids musi być pustą listą,
   - answer musi brzmieć dokładnie:
     "Nie znalazłem wystarczających informacji w dostarczonych dokumentach."

3. Nie wymyślaj numerów źródeł.
4. Nie dodawaj komentarza ani wyjaśnienia.
5. Zwróć wyłącznie kompletny, poprawiony obiekt JSON.

BŁĘDY WALIDACJI:

{validation_details}
""".strip()
