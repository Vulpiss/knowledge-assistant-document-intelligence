import unittest
from pathlib import Path

from app.core.config import AppConfig, CORPUS_PROFILES


class CorpusConfigTests(unittest.TestCase):
    def test_profiles_use_separate_collections_and_datasets(
        self,
    ) -> None:
        v1 = CORPUS_PROFILES["v1"]
        v2 = CORPUS_PROFILES["v2"]

        self.assertNotEqual(
            v1.qdrant_collection,
            v2.qdrant_collection,
        )
        self.assertEqual(
            v1.retrieval_eval_file.name,
            "retrieval_eval_v1.json",
        )
        self.assertEqual(
            v2.retrieval_eval_file.name,
            "retrieval_eval_v2.json",
        )
        self.assertEqual(
            v2.answer_eval_file.name,
            "answer_eval_v2.json",
        )

    def test_activate_corpus_switches_coherent_profile(self) -> None:
        test_config = AppConfig()

        profile = test_config.activate_corpus("v2")

        self.assertEqual(test_config.active_corpus, "v2")
        self.assertEqual(
            test_config.raw_documents_dir,
            profile.raw_documents_dir,
        )
        self.assertEqual(
            test_config.qdrant_collection,
            profile.qdrant_collection,
        )
        self.assertEqual(
            test_config.answer_eval_file,
            profile.answer_eval_file,
        )

    def test_activate_corpus_accepts_safe_overrides(self) -> None:
        test_config = AppConfig()

        test_config.activate_corpus(
            "v2",
            raw_documents_dir=Path("custom/documents"),
            qdrant_collection="custom_collection",
        )

        self.assertEqual(
            test_config.raw_documents_dir,
            Path("custom/documents"),
        )
        self.assertEqual(
            test_config.qdrant_collection,
            "custom_collection",
        )
        self.assertEqual(
            test_config.retrieval_eval_file.name,
            "retrieval_eval_v2.json",
        )

    def test_activate_corpus_rejects_blank_collection(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig().activate_corpus(
                "v1",
                qdrant_collection="   ",
            )


if __name__ == "__main__":
    unittest.main()
