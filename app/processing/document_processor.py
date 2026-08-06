from loguru import logger

from app.ingestion.document import LoadedDocument
from app.processing.cleaner import TextCleaner
from app.processing.metadata_builder import MetadataBuilder
from app.processing.processed_document import ProcessedDocumentUnit


class DocumentProcessor:
    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.metadata_builder = metadata_builder or MetadataBuilder()

    def process(self, loaded_document: LoadedDocument) -> list[ProcessedDocumentUnit]:
        document_id = self.metadata_builder.build_document_id(
            loaded_document.source_path
        )

        processed_units: list[ProcessedDocumentUnit] = []

        for page in loaded_document.pages:
            clean_text = self.cleaner.clean(page.text)

            if not clean_text:
                logger.warning(
                    "Skipping empty text unit | document={} | page={} | unit={}",
                    page.document_name,
                    page.page_number,
                    page.unit_number,
                )
                continue

            metadata = self.metadata_builder.build_unit_metadata(
                loaded_document=loaded_document,
                page=page,
                clean_text=clean_text,
            )

            processed_units.append(
                ProcessedDocumentUnit(
                    document_id=document_id,
                    document_name=loaded_document.document_name,
                    source_path=loaded_document.source_path,
                    file_type=loaded_document.file_type,
                    text=clean_text,
                    page_number=page.page_number,
                    unit_number=page.unit_number,
                    raw_characters=len(page.text),
                    clean_characters=len(clean_text),
                    metadata=metadata,
                )
            )

        logger.info(
            "Processed document: {} | units={} | characters={}",
            loaded_document.document_name,
            len(processed_units),
            sum(unit.clean_characters for unit in processed_units),
        )

        return processed_units