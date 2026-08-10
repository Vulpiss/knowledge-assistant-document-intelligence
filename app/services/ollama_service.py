import json
from collections.abc import Iterator
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    model_available: bool
    message: str


@dataclass(frozen=True)
class OllamaModelPullProgress:
    status: str
    completed: int | None = None
    total: int | None = None

    @property
    def fraction(self) -> float | None:
        if not self.total or self.completed is None:
            return None

        return min(max(self.completed / self.total, 0.0), 1.0)


class OllamaService:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def get_status(
        self,
        timeout_seconds: float = 3.0,
    ) -> OllamaStatus:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return OllamaStatus(
                available=False,
                model_available=False,
                message="Ollama nie odpowiada.",
            )

        raw_models = (
            payload.get("models", [])
            if isinstance(payload, dict)
            else []
        )
        models = raw_models if isinstance(raw_models, list) else []
        model_names = {
            name
            for model in models
            if isinstance(model, dict)
            for name in (model.get("name"), model.get("model"))
            if isinstance(name, str)
        }
        model_available = self.model_name in model_names

        if model_available:
            message = (
                f"Ollama i model {self.model_name} są gotowe."
            )
        else:
            message = (
                "Ollama działa, ale brakuje modelu "
                f"{self.model_name}."
            )

        return OllamaStatus(
            available=True,
            model_available=model_available,
            message=message,
        )

    def pull_model(self) -> Iterator[OllamaModelPullProgress]:
        try:
            with requests.post(
                f"{self.base_url}/api/pull",
                json={
                    "model": self.model_name,
                    "stream": True,
                },
                stream=True,
                timeout=(10, 3600),
            ) as response:
                response.raise_for_status()

                for raw_line in response.iter_lines(
                    decode_unicode=True
                ):
                    if not raw_line:
                        continue

                    line = (
                        raw_line.decode("utf-8")
                        if isinstance(raw_line, bytes)
                        else raw_line
                    )
                    payload = json.loads(line)

                    if error_message := payload.get("error"):
                        raise RuntimeError(str(error_message))

                    status = str(
                        payload.get("status", "Pobieranie modelu…")
                    )
                    completed = payload.get("completed")
                    total = payload.get("total")

                    yield OllamaModelPullProgress(
                        status=status,
                        completed=(
                            completed
                            if isinstance(completed, int)
                            else None
                        ),
                        total=(
                            total
                            if isinstance(total, int)
                            else None
                        ),
                    )
        except requests.RequestException as error:
            raise RuntimeError(
                "Nie udało się pobrać modelu Ollama."
            ) from error
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Ollama zwróciła nieprawidłowy status pobierania."
            ) from error
