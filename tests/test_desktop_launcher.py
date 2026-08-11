import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import desktop_launcher


class DesktopLauncherTests(unittest.TestCase):
    def test_configures_bundled_ollama_on_a_separate_port(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource_root = root / "resources"
            offline_root = resource_root / "offline"
            executable = offline_root / "ollama" / "ollama.exe"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"test")
            (offline_root / "models").mkdir()
            (offline_root / "FULL_OFFLINE").write_text(
                "1",
                encoding="ascii",
            )

            with patch.dict(
                os.environ,
                {"LOCALAPPDATA": str(root / "local")},
                clear=True,
            ), patch(
                "desktop_launcher._find_available_port",
                return_value=11555,
            ):
                desktop_launcher._configure_environment(resource_root)

                self.assertEqual(
                    os.environ["KNOWLEDGE_ASSISTANT_FULL_OFFLINE"],
                    "1",
                )
                self.assertEqual(
                    os.environ["OLLAMA_EXECUTABLE"],
                    str(executable),
                )
                self.assertEqual(
                    os.environ["OLLAMA_MODELS"],
                    str(offline_root / "models"),
                )
                self.assertEqual(
                    os.environ["OLLAMA_BASE_URL"],
                    "http://127.0.0.1:11555",
                )

    def test_standard_package_does_not_enable_full_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            with patch.dict(
                os.environ,
                {"LOCALAPPDATA": str(root / "local")},
                clear=True,
            ):
                desktop_launcher._configure_environment(root / "resources")

                self.assertNotIn(
                    "KNOWLEDGE_ASSISTANT_FULL_OFFLINE",
                    os.environ,
                )
                self.assertNotIn("OLLAMA_EXECUTABLE", os.environ)

    def test_ollama_base_url_removes_trailing_slash(self) -> None:
        with patch.dict(
            os.environ,
            {"OLLAMA_BASE_URL": "http://127.0.0.1:12345/"},
        ):
            self.assertEqual(
                desktop_launcher._ollama_base_url(),
                "http://127.0.0.1:12345",
            )


if __name__ == "__main__":
    unittest.main()
