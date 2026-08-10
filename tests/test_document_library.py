import tempfile
import unittest
from pathlib import Path

from app.services.document_library import DocumentLibrary


class DocumentLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.library = DocumentLibrary(
            self.directory,
            max_document_size_bytes=32,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_saves_and_lists_supported_document(self) -> None:
        saved = self.library.save_document(
            "polityka.txt",
            "Treść dokumentu".encode("utf-8"),
        )

        documents = self.library.list_documents()

        self.assertEqual(saved.name, "polityka.txt")
        self.assertEqual(
            [item.name for item in documents],
            ["polityka.txt"],
        )

    def test_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            self.library.save_document(
                "../sekret.txt",
                b"content",
            )

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(ValueError):
            self.library.save_document(
                "skrypt.exe",
                b"content",
            )

    def test_rejects_empty_document(self) -> None:
        with self.assertRaises(ValueError):
            self.library.save_document("pusty.txt", b"")

    def test_rejects_document_above_size_limit(self) -> None:
        with self.assertRaises(ValueError):
            self.library.save_document(
                "duzy.txt",
                b"a" * 33,
            )

    def test_rejects_duplicate_name(self) -> None:
        self.library.save_document("duplikat.txt", b"pierwszy")

        with self.assertRaises(FileExistsError):
            self.library.save_document(
                "duplikat.txt",
                b"drugi",
            )

    def test_deletes_existing_document(self) -> None:
        self.library.save_document("usun.txt", b"tresc")

        deleted = self.library.delete_document("usun.txt")

        self.assertTrue(deleted)
        self.assertEqual(self.library.list_documents(), [])

    def test_rejects_invalid_pdf_header(self) -> None:
        with self.assertRaises(ValueError):
            self.library.save_document(
                "falszywy.pdf",
                b"not a pdf",
            )


if __name__ == "__main__":
    unittest.main()
