from pathlib import Path

import fitz
from loguru import logger

from app.ingestion.document import DocumentPage, LoadedDocument


class PdfLoader:
    supported_extensions = {".pdf"}

    def load(self, file_path: Path) -> LoadedDocument:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported PDF file extension: {file_path.suffix}")

        logger.info("Loading PDF document: {}", file_path)

        pages: list[DocumentPage] = []

        try:
            with fitz.open(file_path) as pdf_document:
                for page_index, page in enumerate(pdf_document, start=1):
                    text = page.get_text("text")

                    pages.append(
                        DocumentPage(
                            document_name=file_path.name,
                            source_path=file_path,
                            file_type="pdf",
                            text=text,
                            page_number=page_index,
                            unit_number=page_index,
                            metadata={
                                "loader": "PdfLoader",
                                "pdf_page_count": pdf_document.page_count,
                            },
                        )
                    )

        except Exception as error:
            logger.exception("Failed to load PDF document: {}", file_path)
            raise RuntimeError(f"Failed to load PDF document: {file_path}") from error

        return LoadedDocument(
            document_name=file_path.name,
            source_path=file_path,
            file_type="pdf",
            pages=pages,
        )