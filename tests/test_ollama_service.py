import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services.ollama_service import OllamaService


class OllamaServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = OllamaService(
            base_url="http://localhost:11434",
            model_name="gemma3:4b",
        )

    @patch("app.services.ollama_service.requests.get")
    def test_status_detects_available_model(self, get: MagicMock) -> None:
        response = get.return_value
        response.json.return_value = {
            "models": [{"name": "gemma3:4b"}]
        }

        status = self.service.get_status()

        self.assertTrue(status.available)
        self.assertTrue(status.model_available)

    @patch("app.services.ollama_service.requests.get")
    def test_status_handles_connection_error(self, get: MagicMock) -> None:
        get.side_effect = requests.ConnectionError()

        status = self.service.get_status()

        self.assertFalse(status.available)
        self.assertFalse(status.model_available)

    @patch("app.services.ollama_service.requests.post")
    def test_pull_model_reports_progress(self, post: MagicMock) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            json.dumps(
                {
                    "status": "downloading",
                    "completed": 50,
                    "total": 100,
                }
            ),
            json.dumps({"status": "success"}),
        ]
        post.return_value = response

        updates = list(self.service.pull_model())

        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0].fraction, 0.5)
        self.assertEqual(updates[-1].status, "success")

    @patch("app.services.ollama_service.requests.post")
    def test_pull_model_surfaces_api_error(self, post: MagicMock) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            json.dumps({"error": "download failed"})
        ]
        post.return_value = response

        with self.assertRaisesRegex(
            RuntimeError,
            "download failed",
        ):
            list(self.service.pull_model())


if __name__ == "__main__":
    unittest.main()
