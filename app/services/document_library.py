from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".txt",
    }
)
MAX_DOCUMENT_SIZE_BYTES = 20 * 1024 * 1024

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"/\\|?*')


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    extension: str
    size_bytes: int


class DocumentLibrary:
    def __init__(
        self,
        directory: Path,
        *,
        max_document_size_bytes: int = MAX_DOCUMENT_SIZE_BYTES,
    ) -> None:
        if max_document_size_bytes <= 0:
            raise ValueError(
                "Maksymalny rozmiar dokumentu musi być większy od zera."
            )

        self.directory = Path(directory)
        self.max_document_size_bytes = max_document_size_bytes

    def list_documents(self) -> list[DocumentInfo]:
        if not self.directory.exists():
            return []

        documents = [
            self._to_info(path)
            for path in self.directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ]

        return sorted(
            documents,
            key=lambda document: document.name.casefold(),
        )

    def save_document(
        self,
        file_name: str,
        content: bytes,
    ) -> DocumentInfo:
        normalized_name = self._validate_file_name(file_name)
        self._validate_content(normalized_name, content)
        self.directory.mkdir(parents=True, exist_ok=True)

        target = self.directory / normalized_name

        try:
            with target.open("xb") as stream:
                stream.write(content)
        except FileExistsError as error:
            raise FileExistsError(
                f"Dokument już istnieje: {normalized_name}"
            ) from error
        except Exception:
            target.unlink(missing_ok=True)
            raise

        return self._to_info(target)

    def delete_document(self, file_name: str) -> bool:
        normalized_name = self._validate_file_name(file_name)
        target = self.directory / normalized_name

        if not target.exists():
            return False

        if not target.is_file():
            raise ValueError(
                f"Ścieżka nie wskazuje dokumentu: {normalized_name}"
            )

        target.unlink()
        return True

    def _validate_file_name(self, file_name: str) -> str:
        if not isinstance(file_name, str):
            raise TypeError("Nazwa dokumentu musi być tekstem.")

        normalized_name = file_name.strip()

        if not normalized_name or normalized_name != file_name:
            raise ValueError("Nazwa dokumentu jest nieprawidłowa.")

        if normalized_name in {".", ".."}:
            raise ValueError("Nazwa dokumentu jest nieprawidłowa.")

        if any(
            character in _WINDOWS_INVALID_CHARACTERS
            or ord(character) < 32
            for character in normalized_name
        ):
            raise ValueError(
                "Nazwa dokumentu zawiera niedozwolone znaki."
            )

        if Path(normalized_name).name != normalized_name:
            raise ValueError(
                "Dokument musi mieć prostą nazwę bez katalogów."
            )

        path = Path(normalized_name)
        extension = path.suffix.lower()

        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            supported = ", ".join(
                sorted(SUPPORTED_DOCUMENT_EXTENSIONS)
            )
            raise ValueError(
                f"Nieobsługiwany typ dokumentu. Dozwolone: {supported}."
            )

        if path.stem.upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(
                "Ta nazwa dokumentu jest zarezerwowana przez Windows."
            )

        return normalized_name

    def _validate_content(
        self,
        file_name: str,
        content: bytes,
    ) -> None:
        if not isinstance(content, bytes):
            raise TypeError("Zawartość dokumentu musi mieć format bytes.")

        if not content:
            raise ValueError("Dokument jest pusty.")

        if len(content) > self.max_document_size_bytes:
            maximum_megabytes = (
                self.max_document_size_bytes / 1024 / 1024
            )
            raise ValueError(
                "Dokument przekracza limit "
                f"{maximum_megabytes:g} MB."
            )

        extension = Path(file_name).suffix.lower()

        if extension == ".pdf" and b"%PDF-" not in content[:1024]:
            raise ValueError("Plik nie ma poprawnego nagłówka PDF.")

        if extension == ".docx" and not content.startswith(b"PK"):
            raise ValueError("Plik nie ma poprawnego formatu DOCX.")

        if extension == ".txt":
            text = self._decode_text(content)

            if not text.strip() or "\x00" in text:
                raise ValueError(
                    "Plik TXT nie zawiera prawidłowej treści tekstowej."
                )

    @staticmethod
    def _decode_text(content: bytes) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("cp1250")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "Plik TXT musi używać kodowania UTF-8 lub CP1250."
                ) from error

    @staticmethod
    def _to_info(path: Path) -> DocumentInfo:
        return DocumentInfo(
            name=path.name,
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
        )
