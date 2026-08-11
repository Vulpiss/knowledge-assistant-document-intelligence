import runpy
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "packaging"
    / "split_release.py"
)
SCRIPT = runpy.run_path(str(SCRIPT_PATH))
split_file = SCRIPT["split_file"]


class SplitReleaseTests(unittest.TestCase):
    def test_parts_reassemble_to_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "release.zip"
            source_bytes = bytes(range(256)) * 50
            source.write_bytes(source_bytes)

            parts = split_file(
                source=source,
                output_directory=root / "parts",
                part_size_bytes=4096,
            )

            self.assertEqual(len(parts), 4)
            rebuilt = b"".join(part.read_bytes() for part in parts)
            self.assertEqual(rebuilt, source_bytes)

    def test_rejects_non_positive_part_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "release.zip"
            source.write_bytes(b"test")

            with self.assertRaises(ValueError):
                split_file(
                    source=source,
                    output_directory=root / "parts",
                    part_size_bytes=0,
                )


if __name__ == "__main__":
    unittest.main()
