import contextlib
import io
import unittest

from main import build_parser


class CliCorpusChoiceTests(unittest.TestCase):
    def test_index_accepts_production_corpus(self) -> None:
        args = build_parser().parse_args(
            ["index", "--corpus", "production", "--rebuild"]
        )

        self.assertEqual(args.corpus, "production")
        self.assertTrue(args.rebuild)

    def test_answer_evaluation_rejects_production_corpus(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "evaluate-answers",
                        "--corpus",
                        "production",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
