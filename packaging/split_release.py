from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_file(
    *,
    source: Path,
    output_directory: Path,
    part_size_bytes: int,
) -> list[Path]:
    if part_size_bytes <= 0:
        raise ValueError("Part size must be positive.")

    output_directory.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []

    with source.open("rb") as source_stream:
        part_number = 1

        while True:
            part_path = output_directory / (
                f"{source.name}.part{part_number:03d}"
            )
            bytes_written = 0

            with part_path.open("wb") as part_stream:
                while bytes_written < part_size_bytes:
                    chunk = source_stream.read(
                        min(
                            8 * 1024 * 1024,
                            part_size_bytes - bytes_written,
                        )
                    )

                    if not chunk:
                        break

                    part_stream.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written == 0:
                part_path.unlink(missing_ok=True)
                break

            parts.append(part_path)
            part_number += 1

    return parts


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--part-size-mib", type=int, default=1900)
    arguments = parser.parse_args()
    parts = split_file(
        source=arguments.source,
        output_directory=arguments.output_directory,
        part_size_bytes=arguments.part_size_mib * 1024 * 1024,
    )
    archive_hash = sha256(arguments.source)
    (arguments.output_directory / "FULL_ARCHIVE_NAME.txt").write_text(
        arguments.source.name,
        encoding="utf-8",
    )
    (arguments.output_directory / "FULL_ARCHIVE_SHA256.txt").write_text(
        archive_hash,
        encoding="ascii",
    )
    part_hashes = {
        part.name: sha256(part)
        for part in parts
    }
    (arguments.output_directory / "PARTS_SHA256.json").write_text(
        json.dumps(part_hashes, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive": arguments.source.name,
                "sha256": archive_hash,
                "parts": len(parts),
            }
        )
    )


if __name__ == "__main__":
    main()
