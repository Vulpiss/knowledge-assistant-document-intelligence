import json
import unittest
from pathlib import Path


BASELINE_PATH = Path("data/eval/release_baseline_v1.json")


class ReleaseBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = json.loads(
            BASELINE_PATH.read_text(encoding="utf-8")
        )

    def test_baseline_covers_both_corpora(self) -> None:
        self.assertEqual(
            set(self.baseline["profiles"]),
            {"v1", "v2"},
        )

    def test_case_counts_are_frozen(self) -> None:
        profiles = self.baseline["profiles"]

        self.assertEqual(
            profiles["v1"]["retrieval"]["total_cases"],
            9,
        )
        self.assertEqual(
            profiles["v1"]["answers"]["total_cases"],
            6,
        )
        self.assertEqual(
            profiles["v2"]["retrieval"]["total_cases"],
            30,
        )
        self.assertEqual(
            profiles["v2"]["answers"]["total_cases"],
            35,
        )

    def test_answer_quality_stays_strict(self) -> None:
        for profile in self.baseline["profiles"].values():
            answers = profile["answers"]

            self.assertEqual(
                answers["minimum"]["overall_pass_rate"],
                1.0,
            )
            self.assertEqual(
                answers["minimum"]["grounded_answer_rate"],
                1.0,
            )
            self.assertEqual(
                answers["maximum"]["hallucination_rate"],
                0.0,
            )
            self.assertEqual(
                answers["maximum"]["execution_error_rate"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
