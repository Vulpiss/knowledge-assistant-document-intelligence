from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "unigram.json",
)


def prepare_embedding_model(*, local_files_only: bool) -> Path:
    return Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=list(MODEL_FILES),
            local_files_only=local_files_only,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access the network; use the local Hugging Face cache.",
    )
    arguments = parser.parse_args()
    print(
        prepare_embedding_model(
            local_files_only=arguments.local_files_only
        )
    )


if __name__ == "__main__":
    main()
