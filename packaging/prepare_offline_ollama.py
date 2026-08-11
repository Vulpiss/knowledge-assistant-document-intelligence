from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportSummary:
    model_name: str
    files: int
    bytes_copied: int


def export_model(
    *,
    source_models: Path,
    destination_models: Path,
    model_name: str,
) -> ExportSummary:
    manifest_relative = _manifest_relative_path(model_name)
    source_manifest = source_models / "manifests" / manifest_relative

    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"Ollama manifest not found: {source_manifest}"
        )

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    digests = _manifest_digests(manifest)
    destination_manifest = (
        destination_models / "manifests" / manifest_relative
    )
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, destination_manifest)

    bytes_copied = destination_manifest.stat().st_size

    for digest in sorted(digests):
        blob_name = digest.replace(":", "-")
        source_blob = source_models / "blobs" / blob_name
        destination_blob = destination_models / "blobs" / blob_name

        if not source_blob.is_file():
            raise FileNotFoundError(
                f"Ollama blob not found: {source_blob}"
            )

        destination_blob.parent.mkdir(parents=True, exist_ok=True)
        bytes_copied += _copy_verified_blob(
            source=source_blob,
            destination=destination_blob,
            digest=digest,
        )

    return ExportSummary(
        model_name=model_name,
        files=len(digests) + 1,
        bytes_copied=bytes_copied,
    )


def _manifest_relative_path(model_name: str) -> Path:
    name, separator, tag = model_name.partition(":")

    if not separator:
        tag = "latest"

    if not name or not tag:
        raise ValueError("Model name must use the name:tag format.")

    name_parts = [part for part in name.split("/") if part]

    if not name_parts or any(part in {".", ".."} for part in name_parts):
        raise ValueError("Invalid Ollama model name.")

    return Path("registry.ollama.ai", "library", *name_parts, tag)


def _manifest_digests(manifest: object) -> set[str]:
    if not isinstance(manifest, dict):
        raise ValueError("Ollama manifest must be a JSON object.")

    objects = [manifest.get("config")]
    layers = manifest.get("layers", [])

    if not isinstance(layers, list):
        raise ValueError("Ollama manifest layers must be a list.")

    objects.extend(layers)
    digests: set[str] = set()

    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("Ollama manifest contains an invalid item.")

        digest = item.get("digest")

        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("Ollama manifest contains an invalid digest.")

        digests.add(digest)

    return digests


def _copy_verified_blob(
    *,
    source: Path,
    destination: Path,
    digest: str,
) -> int:
    expected_hash = digest.removeprefix("sha256:")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    hasher = hashlib.sha256()
    bytes_copied = 0

    try:
        with source.open("rb") as source_stream, temporary.open(
            "wb"
        ) as destination_stream:
            while chunk := source_stream.read(8 * 1024 * 1024):
                destination_stream.write(chunk)
                hasher.update(chunk)
                bytes_copied += len(chunk)

        actual_hash = hasher.hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"Ollama blob checksum mismatch: {source.name}"
            )

        temporary.replace(destination)
        shutil.copystat(source, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return bytes_copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--destination-models", type=Path, required=True)
    parser.add_argument("--model", default="gemma3:4b")
    arguments = parser.parse_args()
    summary = export_model(
        source_models=arguments.source_models,
        destination_models=arguments.destination_models,
        model_name=arguments.model,
    )
    print(
        json.dumps(
            {
                "model": summary.model_name,
                "files": summary.files,
                "bytes": summary.bytes_copied,
            }
        )
    )


if __name__ == "__main__":
    main()
