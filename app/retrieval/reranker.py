import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace

from app.retrieval.retrieved_chunk import RetrievedChunk


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOP_WORDS = {
    "a",
    "aby",
    "albo",
    "ale",
    "bo",
    "by",
    "byc",
    "czy",
    "dla",
    "do",
    "gdzie",
    "i",
    "ile",
    "jak",
    "jaka",
    "jaki",
    "jakie",
    "jest",
    "kiedy",
    "ktory",
    "ktora",
    "ktore",
    "ma",
    "mam",
    "mozna",
    "na",
    "nalezy",
    "nie",
    "o",
    "od",
    "oraz",
    "po",
    "pod",
    "przed",
    "przez",
    "sie",
    "to",
    "w",
    "wedlug",
    "z",
    "za",
}

_SYNONYM_GROUPS = (
    (
        ("mfa", "uwier"),
        (
            "mfa",
            "uwierzytelnianie",
            "wieloskladnikowe",
            "bezpieczenstwo",
        ),
    ),
    (
        ("narus", "incyd"),
        ("naruszenie", "incydent", "bezpieczenstwo"),
    ),
    (
        ("laptop", "kompu", "urzad", "sprze"),
        ("laptop", "komputer", "urzadzenie", "sprzet"),
    ),
    (
        ("uster", "awari", "serwi"),
        ("usterka", "awaria", "serwis", "zgloszenie"),
    ),
    (
        ("urlop", "nieob"),
        ("urlop", "nieobecnosc", "wniosek"),
    ),
    (
        ("kradz", "utrat", "zgubi", "zagin"),
        ("kradziez", "utrata", "sprzet", "urzadzenie"),
    ),
    (
        ("wymia",),
        ("wymiana", "sprzet", "komputer", "laptop"),
    ),
    (
        ("reten",),
        ("retencja", "przechowywanie", "usuniecie"),
    ),
)

_HISTORICAL_TERMS = {
    "archi",
    "histo",
    "poprz",
    "stary",
    "2023",
    "2024",
}

_CURRENT_TERMS = {
    "aktua",
    "biezac",
    "obecn",
    "obowi",
}


class HybridReranker:
    def __init__(
        self,
        *,
        dense_weight: float = 0.55,
        lexical_weight: float = 0.35,
        document_weight: float = 0.10,
        archive_penalty: float = 0.25,
        max_chunks_per_document: int = 2,
    ) -> None:
        weights = (
            dense_weight,
            lexical_weight,
            document_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Reranking weights cannot be negative."
            )

        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError(
                "Reranking weights must sum to 1.0."
            )

        if archive_penalty < 0:
            raise ValueError(
                "archive_penalty cannot be negative."
            )

        if max_chunks_per_document <= 0:
            raise ValueError(
                "max_chunks_per_document must be greater than 0."
            )

        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.document_weight = document_weight
        self.archive_penalty = archive_penalty
        self.max_chunks_per_document = (
            max_chunks_per_document
        )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            raise ValueError(
                "Reranking limit must be greater than 0."
            )

        if not chunks:
            return []

        query_terms = self._query_terms(query)
        search_terms = [
            self._terms(self._searchable_text(chunk))
            for chunk in chunks
        ]
        bm25_scores = self._bm25_scores(
            query_terms=query_terms,
            documents=search_terms,
        )
        normalized_bm25 = self._normalize_scores(
            bm25_scores
        )

        query_term_set = set(query_terms)
        scored_chunks: list[RetrievedChunk] = []

        for chunk, lexical_score in zip(
            chunks,
            normalized_bm25,
            strict=True,
        ):
            dense_score = min(
                1.0,
                max(0.0, chunk.score),
            )
            document_score = self._document_score(
                query_terms=query_term_set,
                chunk=chunk,
            )
            status_adjustment = self._status_adjustment(
                query_terms=query_term_set,
                chunk=chunk,
            )

            final_score = (
                self.dense_weight * dense_score
                + self.lexical_weight * lexical_score
                + self.document_weight * document_score
                + status_adjustment
            )
            final_score = min(1.0, max(0.0, final_score))

            rerank_metadata = {
                **chunk.metadata,
                "retrieval_dense_score": dense_score,
                "retrieval_lexical_score": lexical_score,
                "retrieval_document_score": document_score,
                "retrieval_status_adjustment": status_adjustment,
                "retrieval_final_score": final_score,
            }

            scored_chunks.append(
                replace(
                    chunk,
                    score=final_score,
                    metadata=rerank_metadata,
                )
            )

        ranked_chunks = sorted(
            scored_chunks,
            key=lambda item: item.score,
            reverse=True,
        )

        if not self._asks_for_history(query_term_set):
            current_chunks = [
                chunk
                for chunk in ranked_chunks
                if not self._is_archived(chunk)
            ]

            if current_chunks:
                ranked_chunks = current_chunks

        return self._select_diverse_results(
            ranked_chunks,
            limit=limit,
        )

    def _select_diverse_results(
        self,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        document_counts: defaultdict[str, int] = defaultdict(int)

        for chunk in chunks:
            if (
                document_counts[chunk.document_name]
                >= self.max_chunks_per_document
            ):
                continue

            selected.append(chunk)
            document_counts[chunk.document_name] += 1

            if len(selected) >= limit:
                break

        return selected

    def _query_terms(self, query: str) -> list[str]:
        normalized_tokens = self._normalized_tokens(query)
        expanded_tokens = list(normalized_tokens)

        for token in normalized_tokens:
            for triggers, synonyms in _SYNONYM_GROUPS:
                if any(
                    token.startswith(trigger)
                    for trigger in triggers
                ):
                    expanded_tokens.extend(synonyms)

        return [
            self._stem(token)
            for token in expanded_tokens
            if token not in _STOP_WORDS
        ]

    def _terms(self, value: str) -> list[str]:
        return [
            self._stem(token)
            for token in self._normalized_tokens(value)
            if token not in _STOP_WORDS
        ]

    @staticmethod
    def _normalized_tokens(value: str) -> list[str]:
        normalized = unicodedata.normalize(
            "NFKD",
            value.casefold(),
        )
        ascii_value = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )

        return _TOKEN_PATTERN.findall(ascii_value)

    @staticmethod
    def _stem(token: str) -> str:
        if token.isdigit() or len(token) <= 5:
            return token

        return token[:5]

    def _searchable_text(
        self,
        chunk: RetrievedChunk,
    ) -> str:
        metadata = chunk.metadata

        return "\n".join(
            [
                chunk.document_name.replace("_", " "),
                str(metadata.get("document_title", "")),
                str(metadata.get("document_status", "")),
                str(metadata.get("document_version", "")),
                chunk.text,
            ]
        )

    def _document_score(
        self,
        *,
        query_terms: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        filename_terms = set(
            self._terms(
                chunk.document_name.replace("_", " ")
            )
        )
        filename_terms = {
            term
            for term in filename_terms
            if not term.isdigit()
            and term != "archi"
        }
        title_terms = set(
            self._terms(
                str(
                    chunk.metadata.get(
                        "document_title",
                        "",
                    )
                )
            )
        )

        if not query_terms:
            return 0.0

        filename_overlap = query_terms & filename_terms

        if filename_overlap:
            return 1.0

        if not title_terms:
            return 0.0

        title_overlap = query_terms & title_terms

        return min(1.0, len(title_overlap) / 3)

    def _status_adjustment(
        self,
        *,
        query_terms: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        status_text = self._status_text(chunk)
        status_terms = set(self._terms(status_text))
        asks_for_history = self._asks_for_history(
            query_terms
        )
        asks_for_current = bool(
            query_terms & _CURRENT_TERMS
        )
        is_archived = self._is_archived(chunk)
        is_current = (
            "obowi" in status_terms
            and not is_archived
        )

        if is_archived and not asks_for_history:
            return -self.archive_penalty

        if is_current and asks_for_current:
            return 0.10

        if is_current:
            return 0.03

        return 0.0

    @staticmethod
    def _asks_for_history(
        query_terms: set[str],
    ) -> bool:
        return bool(query_terms & _HISTORICAL_TERMS)

    def _is_archived(
        self,
        chunk: RetrievedChunk,
    ) -> bool:
        status_text = self._status_text(chunk)
        status_terms = set(self._terms(status_text))

        return (
            "archi" in status_terms
            or "nie stosować" in status_text.casefold()
        )

    @staticmethod
    def _status_text(
        chunk: RetrievedChunk,
    ) -> str:
        return " ".join(
            [
                chunk.document_name,
                str(
                    chunk.metadata.get(
                        "document_status",
                        "",
                    )
                ),
                chunk.text[:300],
            ]
        )

    @staticmethod
    def _bm25_scores(
        *,
        query_terms: list[str],
        documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[float]:
        if not documents:
            return []

        document_count = len(documents)
        average_length = sum(
            len(document)
            for document in documents
        ) / document_count
        average_length = max(1.0, average_length)

        document_frequency: Counter[str] = Counter()

        for document in documents:
            document_frequency.update(set(document))

        query_frequency = Counter(query_terms)
        scores: list[float] = []

        for document in documents:
            term_frequency = Counter(document)
            document_length = len(document)
            score = 0.0

            for term, query_count in query_frequency.items():
                frequency = term_frequency.get(term, 0)

                if frequency == 0:
                    continue

                containing_documents = document_frequency[term]
                inverse_document_frequency = math.log(
                    1
                    + (
                        document_count
                        - containing_documents
                        + 0.5
                    )
                    / (containing_documents + 0.5)
                )
                denominator = (
                    frequency
                    + k1
                    * (
                        1
                        - b
                        + b
                        * document_length
                        / average_length
                    )
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (k1 + 1)
                    / denominator
                    * query_count
                )

            scores.append(score)

        return scores

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:
        if not scores:
            return []

        maximum = max(scores)

        if maximum <= 0:
            return [0.0 for _ in scores]

        return [score / maximum for score in scores]
