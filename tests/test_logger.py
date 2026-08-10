import unittest
from unittest.mock import patch

from app.core.logger import setup_logger


class LoggerSetupTests(unittest.TestCase):
    @patch("app.core.config.AppConfig.ensure_directories")
    @patch("app.core.logger.logger")
    def test_windowed_app_without_stdout_uses_file_sink(
        self,
        mocked_logger,
        _ensure_directories,
    ) -> None:
        with patch("app.core.logger.sys.stdout", None):
            setup_logger()

        mocked_logger.remove.assert_called_once_with()
        mocked_logger.add.assert_called_once()


if __name__ == "__main__":
    unittest.main()
