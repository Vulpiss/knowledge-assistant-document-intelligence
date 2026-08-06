from pathlib import Path

from app.ingestion.docx_loader import DocxLoader
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.txt_loader import TxtLoader
from app.ingestion.document import LoadedDocument


class DocumentLoaderFactory:
    def __init__(self) -> None:
        self.loaders = {
            ".pdf": PdfLoader(),
            ".docx": DocxLoader(),
            ".txt": TxtLoader(),
        }

    def load(self, file_path: Path) -> LoadedDocument:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        loader = self.loaders.get(extension)

        if loader is None:
            supported = ", ".join(self.loaders.keys())
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported: {supported}"
            )

        return loader.load(file_path)