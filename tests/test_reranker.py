import unittest

from app.retrieval.reranker import HybridReranker
from tests.chunk_factory import make_chunk


class HybridRerankerTests(unittest.TestCase):
    def test_current_query_excludes_archived_document(self) -> None:
        current = make_chunk(
            document_name="bezpieczenstwo_2026.txt",
            text=(
                "Status: OBOWIĄZUJĄCY. Aktualne hasło musi mieć "
                "co najmniej 14 znaków."
            ),
            score=0.2,
            chunk_id="current",
            metadata={"document_status": "OBOWIĄZUJĄCY"},
        )
        archived = make_chunk(
            document_name="bezpieczenstwo_2024_archiwum.txt",
            text=(
                "Status: ARCHIWALNY — NIE STOSOWAĆ. "
                "Hasło miało 12 znaków."
            ),
            score=0.99,
            chunk_id="archived",
            metadata={"document_status": "ARCHIWALNY"},
        )

        results = HybridReranker().rerank(
            "Jaki jest aktualny wymóg hasła?",
            [archived, current],
            limit=2,
        )

        self.assertEqual(
            [result.document_name for result in results],
            ["bezpieczenstwo_2026.txt"],
        )

    def test_historical_query_can_return_archive(self) -> None:
        current = make_chunk(
            document_name="sprzet_2026.txt",
            text="Obowiązujący okres laptopa wynosi 4 lata.",
            score=0.4,
            chunk_id="current",
            metadata={"document_status": "OBOWIĄZUJĄCY"},
        )
        archived = make_chunk(
            document_name="sprzet_2023_archiwum.txt",
            text="Archiwalny okres laptopa wynosił 3 lata.",
            score=0.9,
            chunk_id="archived",
            metadata={"document_status": "ARCHIWALNY"},
        )

        results = HybridReranker().rerank(
            "Jaki był stary okres laptopa w 2023 roku?",
            [current, archived],
            limit=2,
        )

        self.assertIn(
            "sprzet_2023_archiwum.txt",
            [result.document_name for result in results],
        )

    def test_limits_chunks_per_document(self) -> None:
        reranker = HybridReranker(
            max_chunks_per_document=1
        )
        chunks = [
            make_chunk(
                document_name="a.txt",
                text="Zasady urlopu i wniosku.",
                score=0.9,
                chunk_id="a-1",
            ),
            make_chunk(
                document_name="a.txt",
                text="Dodatkowe zasady urlopu.",
                score=0.8,
                chunk_id="a-2",
            ),
            make_chunk(
                document_name="b.txt",
                text="Termin wniosku urlopowego.",
                score=0.7,
                chunk_id="b-1",
            ),
        ]

        results = reranker.rerank(
            "Jaki jest termin wniosku urlopowego?",
            chunks,
            limit=3,
        )

        document_names = [
            result.document_name
            for result in results
        ]
        self.assertEqual(document_names.count("a.txt"), 1)
        self.assertIn("b.txt", document_names)

    def test_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            HybridReranker().rerank(
                "pytanie",
                [],
                limit=0,
            )


if __name__ == "__main__":
    unittest.main()
