import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


CorpusName = Literal["v1", "v2", "production"]
EVALUATION_CORPORA: tuple[CorpusName, ...] = ("v1", "v2")


@dataclass(frozen=True)
class CorpusProfile:
    name: CorpusName
    raw_documents_dir: Path
    qdrant_collection: str
    retrieval_eval_file: Path
    answer_eval_file: Path
    answer_eval_results_file: Path


DEFAULT_RAW_DOCUMENTS_DIR = Path(
    os.getenv("RAW_DOCUMENTS_DIR", "data/raw_documents")
)
DEFAULT_EVAL_DIR = Path(
    os.getenv("EVAL_DIR", "data/eval")
)
DEFAULT_QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "knowledge_chunks",
)


CORPUS_PROFILES: dict[CorpusName, CorpusProfile] = {
    "v1": CorpusProfile(
        name="v1",
        raw_documents_dir=Path(
            os.getenv(
                "CORPUS_V1_DOCUMENTS_DIR",
                str(DEFAULT_RAW_DOCUMENTS_DIR),
            )
        ),
        qdrant_collection=os.getenv(
            "CORPUS_V1_COLLECTION",
            DEFAULT_QDRANT_COLLECTION,
        ),
        retrieval_eval_file=Path(
            os.getenv(
                "CORPUS_V1_RETRIEVAL_EVAL_FILE",
                str(DEFAULT_EVAL_DIR / "retrieval_eval_v1.json"),
            )
        ),
        answer_eval_file=Path(
            os.getenv(
                "CORPUS_V1_ANSWER_EVAL_FILE",
                str(DEFAULT_EVAL_DIR / "answer_eval_v1.json"),
            )
        ),
        answer_eval_results_file=Path(
            os.getenv(
                "CORPUS_V1_ANSWER_RESULTS_FILE",
                str(DEFAULT_EVAL_DIR / "answer_eval_v1_results.json"),
            )
        ),
    ),
    "v2": CorpusProfile(
        name="v2",
        raw_documents_dir=Path(
            os.getenv(
                "CORPUS_V2_DOCUMENTS_DIR",
                "data/raw_documents_v2",
            )
        ),
        qdrant_collection=os.getenv(
            "CORPUS_V2_COLLECTION",
            "knowledge_chunks_eval_v2",
        ),
        retrieval_eval_file=Path(
            os.getenv(
                "CORPUS_V2_RETRIEVAL_EVAL_FILE",
                str(DEFAULT_EVAL_DIR / "retrieval_eval_v2.json"),
            )
        ),
        answer_eval_file=Path(
            os.getenv(
                "CORPUS_V2_ANSWER_EVAL_FILE",
                str(DEFAULT_EVAL_DIR / "answer_eval_v2.json"),
            )
        ),
        answer_eval_results_file=Path(
            os.getenv(
                "CORPUS_V2_ANSWER_RESULTS_FILE",
                str(DEFAULT_EVAL_DIR / "answer_eval_v2_results.json"),
            )
        ),
    ),
    "production": CorpusProfile(
        name="production",
        raw_documents_dir=Path(
            os.getenv(
                "CORPUS_PRODUCTION_DOCUMENTS_DIR",
                "data/user_documents",
            )
        ),
        qdrant_collection=os.getenv(
            "CORPUS_PRODUCTION_COLLECTION",
            "knowledge_chunks_production",
        ),
        retrieval_eval_file=Path(
            os.getenv(
                "CORPUS_PRODUCTION_RETRIEVAL_EVAL_FILE",
                str(DEFAULT_EVAL_DIR / "retrieval_eval_production.json"),
            )
        ),
        answer_eval_file=Path(
            os.getenv(
                "CORPUS_PRODUCTION_ANSWER_EVAL_FILE",
                str(DEFAULT_EVAL_DIR / "answer_eval_production.json"),
            )
        ),
        answer_eval_results_file=Path(
            os.getenv(
                "CORPUS_PRODUCTION_ANSWER_RESULTS_FILE",
                str(DEFAULT_EVAL_DIR / "answer_eval_production_results.json"),
            )
        ),
    ),
}


class AppConfig(BaseModel):
    app_name: str = os.getenv(
        "APP_NAME",
        "AI Knowledge Assistant",
    )
    app_env: str = os.getenv(
        "APP_ENV",
        "development",
    )
    log_level: str = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )

    active_corpus: CorpusName = "v1"
    raw_documents_dir: Path = (
        CORPUS_PROFILES["v1"].raw_documents_dir
    )
    processed_dir: Path = Path(
        os.getenv("PROCESSED_DIR", "data/processed")
    )
    eval_dir: Path = DEFAULT_EVAL_DIR
    log_dir: Path = Path(
        os.getenv("LOG_DIR", "logs")
    )

    retrieval_eval_file: Path = (
        CORPUS_PROFILES["v1"].retrieval_eval_file
    )
    answer_eval_file: Path = (
        CORPUS_PROFILES["v1"].answer_eval_file
    )
    answer_eval_results_file: Path = (
        CORPUS_PROFILES["v1"].answer_eval_results_file
    )

    chunk_size: int = int(
        os.getenv("CHUNK_SIZE", "900")
    )
    chunk_overlap: int = int(
        os.getenv("CHUNK_OVERLAP", "150")
    )

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_batch_size: int = int(
        os.getenv("EMBEDDING_BATCH_SIZE", "16")
    )

    qdrant_path: Path = Path(
        os.getenv("QDRANT_PATH", "data/qdrant")
    )
    qdrant_collection: str = (
        CORPUS_PROFILES["v1"].qdrant_collection
    )

    retrieval_top_k: int = int(
        os.getenv("RETRIEVAL_TOP_K", "5")
    )
    retrieval_candidate_pool: int = int(
        os.getenv("RETRIEVAL_CANDIDATE_POOL", "20")
    )
    retrieval_dense_weight: float = float(
        os.getenv("RETRIEVAL_DENSE_WEIGHT", "0.55")
    )
    retrieval_lexical_weight: float = float(
        os.getenv("RETRIEVAL_LEXICAL_WEIGHT", "0.35")
    )
    retrieval_document_weight: float = float(
        os.getenv("RETRIEVAL_DOCUMENT_WEIGHT", "0.10")
    )
    retrieval_archive_penalty: float = float(
        os.getenv("RETRIEVAL_ARCHIVE_PENALTY", "0.25")
    )
    retrieval_max_chunks_per_document: int = int(
        os.getenv("RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", "2")
    )

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "gemma3:4b",
    )
    ollama_timeout_seconds: int = int(
        os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")
    )

    answer_top_k: int = int(
        os.getenv("ANSWER_TOP_K", "3")
    )
    max_context_characters: int = int(
        os.getenv("MAX_CONTEXT_CHARACTERS", "6000")
    )

    def activate_corpus(
        self,
        corpus_name: CorpusName,
        *,
        raw_documents_dir: Path | None = None,
        qdrant_collection: str | None = None,
    ) -> CorpusProfile:
        profile = CORPUS_PROFILES[corpus_name]

        selected_collection = (
            qdrant_collection or profile.qdrant_collection
        ).strip()

        if not selected_collection:
            raise ValueError(
                "Nazwa kolekcji Qdrant nie może być pusta."
            )

        self.active_corpus = corpus_name
        self.raw_documents_dir = Path(
            raw_documents_dir or profile.raw_documents_dir
        )
        self.qdrant_collection = selected_collection
        self.retrieval_eval_file = (
            profile.retrieval_eval_file
        )
        self.answer_eval_file = profile.answer_eval_file
        self.answer_eval_results_file = (
            profile.answer_eval_results_file
        )

        return profile

    def ensure_directories(self) -> None:
        self.raw_documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.eval_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Qdrant sam utworzy swój katalog.
        # Tutaj zapewniamy istnienie katalogu nadrzędnego.
        self.qdrant_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


config = AppConfig()
