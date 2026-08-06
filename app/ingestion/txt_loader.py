from pathlib import Path

from loguru import logger

from app.ingestion.document import DocumentPage, LoadedDocument


class TxtLoader:
    supported_extensions = {".txt"}

    def load(self, file_path: Path) -> LoadedDocument:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported TXT file extension: {file_path.suffix}")

        logger.info("Loading TXT document: {}", file_path)

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed for {}. Trying cp1250.", file_path)
            text = file_path.read_text(encoding="cp1250")

        page = DocumentPage(
            document_name=file_path.name,
            source_path=file_path,
            file_type="txt",
            text=text,
            page_number=None,
            unit_number=1,
            metadata={
                "loader": "TxtLoader",
            },
        )

        return LoadedDocument(
            document_name=file_path.name,
            source_path=file_path,
            file_type="txt",
            pages=[page],
        )