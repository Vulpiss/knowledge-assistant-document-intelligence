import hashlib
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from app.ingestion.document import DocumentPage, LoadedDocument


class MetadataBuilder:
    def build_document_id(self, source_path: Path) -> str:
        normalized_path = str(source_path).lower().encode("utf-8")
        return hashlib.sha256(normalized_path).hexdigest()[:16]

    def build_content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def build_unit_metadata(
        self,
        loaded_document: LoadedDocument,
        page: DocumentPage,
        clean_text: str,
    ) -> dict[str, Any]:
        source_path = loaded_document.source_path
        document_context = self._extract_document_context(
            clean_text
        )

        return {
            "document_name": loaded_document.document_name,
            "source_path": str(source_path),
            "file_type": loaded_document.file_type,
            "page_number": page.page_number,
            "unit_number": page.unit_number,
            "raw_characters": len(page.text),
            "clean_characters": len(clean_text),
            "content_hash": self.build_content_hash(clean_text),
            "ingested_at_utc": datetime.now(UTC).isoformat(),
            "loader_metadata": page.metadata,
            **document_context,
        }

    @staticmethod
    def _extract_document_context(
        clean_text: str,
    ) -> dict[str, str]:
        lines = [
            line.strip()
            for line in clean_text.splitlines()
            if line.strip()
        ]

        if not lines:
            return {}

        context = {
            "document_title": lines[0],
        }

        field_prefixes = {
            "status:": "document_status",
            "wersja:": "document_version",
            "obowiązuje od:": "document_valid_from",
            "obowiązywała do:": "document_valid_until",
        }

        for line in lines[:12]:
            normalized_line = line.casefold()

            for prefix, field_name in field_prefixes.items():
                if not normalized_line.startswith(prefix):
                    continue

                value = line.split(":", maxsplit=1)[-1].strip()

                if value:
                    context[field_name] = value

                break

        return context
