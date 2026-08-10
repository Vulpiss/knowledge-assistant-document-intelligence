from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


APP_NAME = "Knowledge Assistant"
APP_DATA_DIRECTORY_NAME = "KnowledgeAssistant"
MUTEX_NAME = "Local\\KnowledgeAssistantDesktopApp"
OLLAMA_HEALTH_URL = "http://127.0.0.1:11434/api/tags"

_mutex_handle: int | None = None


def main() -> int:
    multiprocessing.freeze_support()
    resource_root = Path(__file__).resolve().parent
    app_data_root = _configure_environment(resource_root)
    _configure_logging(app_data_root)

    if not _acquire_single_instance():
        _open_existing_instance(app_data_root)
        return 0

    _start_ollama_if_available()
    port = _find_available_port()
    _write_port_file(app_data_root, port)
    application_url = f"http://127.0.0.1:{port}"

    if os.getenv("KNOWLEDGE_ASSISTANT_NO_BROWSER") != "1":
        threading.Thread(
            target=_open_browser_when_ready,
            args=(application_url,),
            daemon=True,
        ).start()

    streamlit_script = (
        resource_root
        / "app"
        / "interfaces"
        / "streamlit_app.py"
    )

    if not streamlit_script.exists():
        raise FileNotFoundError(
            f"Nie znaleziono interfejsu aplikacji: {streamlit_script}"
        )

    os.chdir(resource_root)

    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(streamlit_script),
        "--global.developmentMode=false",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--browser.gatherUsageStats=false",
    ]

    try:
        return int(streamlit_cli.main() or 0)
    finally:
        _remove_port_file(app_data_root)


def _configure_environment(resource_root: Path) -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")

    if local_app_data:
        app_data_root = Path(local_app_data) / APP_DATA_DIRECTORY_NAME
    else:
        app_data_root = (
            Path.home()
            / "AppData"
            / "Local"
            / APP_DATA_DIRECTORY_NAME
        )

    data_root = app_data_root / "data"
    documents_directory = data_root / "documents"
    qdrant_directory = data_root / "qdrant"
    logs_directory = app_data_root / "logs"
    processed_directory = data_root / "processed"
    eval_directory = data_root / "eval"
    model_directory = (
        resource_root
        / "models"
        / "paraphrase-multilingual-MiniLM-L12-v2"
    )

    for directory in (
        documents_directory,
        logs_directory,
        processed_directory,
        eval_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    environment = {
        "APP_ENV": "production",
        "RAW_DOCUMENTS_DIR": str(documents_directory),
        "CORPUS_PRODUCTION_DOCUMENTS_DIR": str(
            documents_directory
        ),
        "CORPUS_PRODUCTION_COLLECTION": (
            "knowledge_chunks_production"
        ),
        "QDRANT_PATH": str(qdrant_directory),
        "PROCESSED_DIR": str(processed_directory),
        "EVAL_DIR": str(eval_directory),
        "LOG_DIR": str(logs_directory),
        "HF_HOME": str(app_data_root / "models" / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }

    if model_directory.exists():
        environment["EMBEDDING_MODEL_NAME"] = str(model_directory)

    os.environ.update(environment)
    return app_data_root


def _configure_logging(app_data_root: Path) -> None:
    log_file = app_data_root / "logs" / "launcher.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8",
    )
    logging.info("Starting %s", APP_NAME)


def _acquire_single_instance() -> bool:
    global _mutex_handle

    if os.name != "nt":
        return True

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)

    if not handle:
        return True

    _mutex_handle = int(handle)
    error_already_exists = 183
    return ctypes.get_last_error() != error_already_exists


def _open_existing_instance(app_data_root: Path) -> None:
    port_file = app_data_root / "runtime.port"

    try:
        port = int(port_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        port = 8501

    webbrowser.open(f"http://127.0.0.1:{port}")


def _start_ollama_if_available() -> None:
    if _ollama_is_ready():
        return

    executable = _find_ollama_executable()

    if executable is None:
        logging.warning("Ollama executable was not found.")
        return

    try:
        subprocess.Popen(
            [str(executable), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            ),
        )
    except OSError:
        logging.exception("Could not start Ollama.")
        return

    for _ in range(40):
        if _ollama_is_ready():
            logging.info("Ollama started successfully.")
            return

        time.sleep(0.5)

    logging.warning("Ollama did not become ready in time.")


def _find_ollama_executable() -> Path | None:
    configured = os.getenv("OLLAMA_EXECUTABLE")

    candidates = [
        Path(configured) if configured else None,
        Path(found) if (found := shutil.which("ollama")) else None,
        (
            Path(os.getenv("LOCALAPPDATA", ""))
            / "Programs"
            / "Ollama"
            / "ollama.exe"
        ),
    ]

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate

    return None


def _ollama_is_ready() -> bool:
    try:
        with urllib.request.urlopen(
            OLLAMA_HEALTH_URL,
            timeout=1,
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _open_browser_when_ready(application_url: str) -> None:
    health_url = f"{application_url}/_stcore/health"

    for _ in range(240):
        try:
            with urllib.request.urlopen(
                health_url,
                timeout=1,
            ) as response:
                if response.status == 200:
                    webbrowser.open(application_url)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)

    logging.error("Streamlit did not become ready in time.")


def _write_port_file(app_data_root: Path, port: int) -> None:
    (app_data_root / "runtime.port").write_text(
        str(port),
        encoding="utf-8",
    )


def _remove_port_file(app_data_root: Path) -> None:
    (app_data_root / "runtime.port").unlink(missing_ok=True)


def _show_error(message: str) -> None:
    logging.exception(message)

    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(
            None,
            message,
            APP_NAME,
            0x10,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        _show_error(
            "Nie udało się uruchomić Knowledge Assistant.\n\n"
            f"Szczegóły: {error}"
        )
        raise SystemExit(1) from error
