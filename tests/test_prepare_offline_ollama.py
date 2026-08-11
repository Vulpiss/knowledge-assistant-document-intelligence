import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "packaging"
    / "prepare_offline_ollama.py"
)
SCRIPT = runpy.run_path(str(SCRIPT_PATH))
export_model = SCRIPT["export_model"]


class PrepareOfflineOllamaTests(unittest.TestCase):
    def test_exports_only_manifest_referenced_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            blobs = source / "blobs"
            manifest_path = (
                source
                / "manifests"
                / "registry.ollama.ai"
                / "library"
                / "gemma3"
                / "4b"
            )
            blobs.mkdir(parents=True)
            manifest_path.parent.mkdir(parents=True)
            payloads = (b"config", b"weights")
            digests = []

            for payload in payloads:
                digest = hashlib.sha256(payload).hexdigest()
                digests.append(f"sha256:{digest}")
                (blobs / f"sha256-{digest}").write_bytes(payload)

            (blobs / "sha256-unused").write_bytes(b"unused")
            manifest_path.write_text(
                json.dumps(
                    {
                        "config": {"digest": digests[0]},
                        "layers": [{"digest": digests[1]}],
                    }
                ),
                encoding="utf-8",
            )

            summary = export_model(
                source_models=source,
                destination_models=destination,
                model_name="gemma3:4b",
            )

            self.assertEqual(summary.files, 3)
            exported_blobs = list((destination / "blobs").iterdir())
            self.assertEqual(len(exported_blobs), 2)
            self.assertFalse(
                (destination / "blobs" / "sha256-unused").exists()
            )

    def test_rejects_blob_with_wrong_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            manifest_path = (
                source
                / "manifests"
                / "registry.ollama.ai"
                / "library"
                / "gemma3"
                / "4b"
            )
            manifest_path.parent.mkdir(parents=True)
            (source / "blobs").mkdir()
            digest = "0" * 64
            (source / "blobs" / f"sha256-{digest}").write_bytes(
                b"wrong"
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "config": {"digest": f"sha256:{digest}"},
                        "layers": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                export_model(
                    source_models=source,
                    destination_models=root / "destination",
                    model_name="gemma3:4b",
                )


if __name__ == "__main__":
    unittest.main()
