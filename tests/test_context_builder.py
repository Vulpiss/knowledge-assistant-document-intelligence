import unittest

from app.generation.context_builder import ContextBuilder
from tests.chunk_factory import make_chunk


class ContextBuilderTests(unittest.TestCase):
    def test_includes_document_version_metadata(self) -> None:
        chunk = make_chunk(
            document_name="bezpieczenstwo_2026.txt",
            text="Hasło musi mieć co najmniej 14 znaków.",
            chunk_id="security-current",
            metadata={
                "document_title": "Polityka bezpieczeństwa",
                "document_status": "OBOWIĄZUJĄCY",
                "document_version": "3.2",
                "document_valid_from": "2026-07-01",
            },
        )

        context = ContextBuilder(max_characters=2000).build(
            [chunk]
        )

        self.assertIn("[SOURCE 1]", context.text)
        self.assertIn(
            "document: bezpieczenstwo_2026.txt",
            context.text,
        )
        self.assertIn("status: OBOWIĄZUJĄCY", context.text)
        self.assertIn("version: 3.2", context.text)
        self.assertIn("[/SOURCE 1]", context.text)
        self.assertEqual(len(context.sources), 1)

    def test_respects_maximum_context_size(self) -> None:
        chunk = make_chunk(
            document_name="long.txt",
            text="wartość " * 300,
        )

        context = ContextBuilder(max_characters=240).build([chunk])

        self.assertLessEqual(len(context.text), 240)
        self.assertEqual(len(context.sources), 1)
        self.assertTrue(context.sources[0].text)

    def test_empty_input_builds_empty_context(self) -> None:
        context = ContextBuilder(max_characters=100).build([])

        self.assertEqual(context.text, "")
        self.assertEqual(context.sources, ())


if __name__ == "__main__":
    unittest.main()
