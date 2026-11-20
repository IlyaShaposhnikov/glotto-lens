"""Vector search over feature documents and its fusion with BM25 (hybrid search).

BM25 matches words, so a rewritten query with extra words ("number of genders noun
classes") can rank the wrong features first. Embeddings compare meanings instead. Each
feature document gets a vector from Yandex text-search-doc, each query one from
text-search-query; both have 256 dimensions. The vectors live in the same SQLite database,
and a search compares the query with every feature vector: there are a few thousand, which
numpy handles in milliseconds, so no separate vector database is needed.

Two ways to fuse the rankings of features (FUSION):
- "slots" (default): the first features of BM25 keep their places, and the last
  VECTOR_SLOTS places go to the best vector features BM25 did not bring. A feature only
  the vectors find (WALS 30A for "number of genders noun classes") gets in, and BM25's own
  first choice (WALS 2A for "number of vowel phonemes") cannot be pushed out.
- "rrf", reciprocal rank fusion: a feature scores 1 / (RRF_K + rank) in each ranking it
  appears in. It needs no score calibration, but it rewards agreement: features both
  channels rank fairly high beat the first feature of either one. On the tuning set that
  lost two features and won none (38/41 against 40/41 for BM25 alone).

    glotto-lens embed                          embed the feature documents (once, cached)
    glotto-lens search QUERY --hybrid          compare with plain BM25 (--per-feature)
    glotto-lens search QUERY --vectors-only    the vector channel alone, for diagnosis
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from glotto_lens.llm import BASE_URL, ModelError, load_env, timeout_seconds
from glotto_lens.search import Hit, document_hit, search
from glotto_lens.sources import DATA_DIR, DB_PATH

DIMENSIONS = 256
# "latest" is the only version AI Studio serves; reports record it, and the vector count
DOC_MODEL = "text-search-doc/latest"
QUERY_MODEL = "text-search-query/latest"
MAX_CHARS = 3000  # the beginning of a long document says what it is about
CANDIDATES = 30  # features taken from each ranking before fusion
RRF_K = 60  # the usual constant; larger values flatten the difference between ranks
FUSION = "slots"
VECTOR_SLOTS = 2  # of the pool's places, for features that only the vectors find
CACHE_DIR = DATA_DIR / "cache" / "embeddings"
CACHE_VERSION = 1

VECTORS_TABLE = """
CREATE TABLE IF NOT EXISTS document_vectors (
    document_id INTEGER PRIMARY KEY REFERENCES documents (id),
    model TEXT NOT NULL,
    vector BLOB NOT NULL
)"""
MISSING = """
    SELECT d.id, d.title, d.section, d.text FROM documents d
    LEFT JOIN document_vectors v ON v.document_id = d.id AND v.model = ?
    WHERE d.kind = 'feature' AND v.document_id IS NULL ORDER BY d.id"""
STORED = """
    SELECT v.document_id, d.feature_id, v.vector FROM document_vectors v
    JOIN documents d ON d.id = v.document_id WHERE v.model = ? ORDER BY v.document_id"""


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """Vectors for texts; kind is "doc" for documents, "query" for search queries."""
        ...


class YandexEmbedder:
    """Yandex AI Studio embeddings through the OpenAI-compatible API."""

    name = "text-search"

    def __init__(self) -> None:
        load_env()
        self.timeout = timeout_seconds()
        self.api_key = os.environ.get("YANDEX_API_KEY")
        self.folder_id = os.environ.get("YANDEX_FOLDER_ID")
        if not self.api_key or not self.folder_id:
            raise RuntimeError("Set YANDEX_API_KEY and YANDEX_FOLDER_ID in the .env file")
        self.tokens = 0
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                project=self.folder_id,
                base_url=BASE_URL,
                timeout=self.timeout,
            )
        return self._client

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        model = f"emb://{self.folder_id}/{DOC_MODEL if kind == 'doc' else QUERY_MODEL}"
        try:
            response = self.client.embeddings.create(
                model=model, input=texts, encoding_format="float"
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status is None and type(error).__name__ not in (
                "APIConnectionError",
                "APITimeoutError",
            ):
                raise
            raise ModelError(
                f"embedding model text-search-{kind}: {type(error).__name__}"
                + (f" (HTTP {status})" if status else "")
                + f". {getattr(error, 'message', '') or ''}".rstrip()
            ) from error
        if response.usage:
            self.tokens += response.usage.prompt_tokens or 0
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class CachedEmbedder:
    """Disk cache in front of an embedder: rebuilding the database or rerunning an
    evaluation does not pay for the same vectors twice."""

    def __init__(self, embedder: Embedder, cache_dir: Path = CACHE_DIR) -> None:
        self.embedder = embedder
        self.name = embedder.name
        self.cache_dir = cache_dir
        self.hits = 0
        self.calls = 0

    def _path(self, text: str, kind: str) -> Path:
        key = json.dumps([CACHE_VERSION, self.name, kind, text], ensure_ascii=False)
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        for text in texts:
            path = self._path(text, kind)
            try:
                vector = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                vector = None
            if isinstance(vector, list) and len(vector) == DIMENSIONS:
                vectors.append(vector)
                self.hits += 1
            else:  # missing, damaged or of another size: a miss
                vectors.append(None)
        missing = [i for i, vector in enumerate(vectors) if vector is None]
        if missing:
            self.calls += 1
            fresh = self.embedder.embed([texts[i] for i in missing], kind)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for i, vector in zip(missing, fresh, strict=True):
                if len(vector) != DIMENSIONS:
                    raise ModelError(
                        f"{self.name} returned {len(vector)} numbers, not {DIMENSIONS}"
                    )
                path = self._path(texts[i], kind)
                part = path.with_suffix(".part")
                part.write_text(json.dumps(vector), encoding="utf-8")
                part.replace(path)
                vectors[i] = vector
        return vectors  # type: ignore[return-value]


def document_text(title: str, section: str, text: str) -> str:
    return f"{title}. {section}\n{text}"[:MAX_CHARS]


def normalized(vectors) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def embed_documents(
    conn: sqlite3.Connection, embedder: Embedder, batch: int = 1, progress=None
) -> int:
    """Store vectors for feature documents that have none yet; return how many were added.
    Each batch is committed, so an interrupted run continues where it stopped. AI Studio
    takes one text per call ("Array input must contain exactly one string"), so the
    command uses batch 1; tests use larger batches."""
    conn.execute(VECTORS_TABLE)
    rows = conn.execute(MISSING, (embedder.name,)).fetchall()
    for start in range(0, len(rows), max(1, batch)):
        chunk = rows[start : start + max(1, batch)]
        texts = [document_text(title, section, text) for _, title, section, text in chunk]
        vectors = normalized(embedder.embed(texts, "doc"))
        conn.executemany(
            "INSERT OR REPLACE INTO document_vectors VALUES (?, ?, ?)",
            [
                (row[0], embedder.name, vector.astype("<f4").tobytes())
                for row, vector in zip(chunk, vectors, strict=True)
            ],
        )
        conn.commit()
        if progress:
            progress(start + len(chunk), len(rows))
    return len(rows)


def stored_vectors(conn: sqlite3.Connection, model: str) -> tuple[list, list, np.ndarray]:
    try:
        rows = conn.execute(STORED, (model,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        raise RuntimeError("No document vectors in the database: run 'glotto-lens embed' first")
    size = DIMENSIONS * 4
    damaged = [row[0] for row in rows if len(row[2]) != size]
    if damaged:
        raise RuntimeError(
            f"{len(damaged)} stored vectors are not {DIMENSIONS} floats (document"
            f" {damaged[0]}, ...): delete document_vectors and run 'glotto-lens embed'"
        )
    ids = [row[0] for row in rows]
    features = [row[1] for row in rows]
    matrix = np.frombuffer(b"".join(row[2] for row in rows), dtype="<f4")
    return ids, features, matrix.reshape(len(rows), DIMENSIONS)


def vector_ranking(
    conn: sqlite3.Connection, query_vector, model: str, limit: int
) -> list[tuple[int, str, float]]:
    """Features ranked by the cosine similarity of their best document to the query:
    (document id, feature id, similarity), one entry per feature."""
    if limit <= 0:
        return []
    ids, features, matrix = stored_vectors(conn, model)
    similarity = matrix @ normalized(query_vector)
    ranking: list[tuple[int, str, float]] = []
    seen: set[str] = set()
    for index in np.argsort(-similarity, kind="stable"):
        feature = features[index]
        if feature in seen:
            continue
        seen.add(feature)
        ranking.append((ids[index], feature, float(similarity[index])))
        if len(ranking) == limit:
            break
    return ranking


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Keys ordered by the sum of 1 / (k + rank) over the rankings; ties keep the order
    in which keys first appear (the first ranking wins)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    order = {key: i for i, key in enumerate(dict.fromkeys(k for r in rankings for k in r))}
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


def reserved_slots(keyword: list[str], semantic: list[str], limit: int, slots: int) -> list[str]:
    """The first limit - slots keyword features, then the best semantic features not among
    them; places still free are filled from the rest of both rankings."""
    slots = min(slots, limit)
    chosen = list(dict.fromkeys(keyword[: limit - slots]))
    chosen += [feature for feature in semantic if feature not in chosen][:slots]
    for feature in [*keyword, *semantic]:
        if len(chosen) >= limit:
            break
        if feature not in chosen:
            chosen.append(feature)
    return chosen[:limit]


def hybrid_features(
    conn: sqlite3.Connection, question: str, limit: int, embedder: Embedder
) -> list[Hit]:
    """The best document of each feature, from BM25 and vectors fused as FUSION says.
    A drop-in replacement for search(..., per_feature=True); scores are ranks turned into
    numbers (higher is better), not comparable with BM25 scores."""
    if limit <= 0:
        return []  # no paid query call for nothing
    candidates = max(CANDIDATES, limit)  # never fewer than asked for
    keyword = search(conn, question, candidates, per_feature=True)
    query_vector = embedder.embed([question], "query")[0]
    semantic = vector_ranking(conn, query_vector, embedder.name, candidates)
    keyword_hits = {hit.feature_id: hit for hit in keyword}
    semantic_documents = {feature: document_id for document_id, feature, _ in semantic}
    keyword_order = [hit.feature_id for hit in keyword]
    semantic_order = [feature for _, feature, _ in semantic]
    if FUSION == "rrf":
        fused = reciprocal_rank_fusion([keyword_order, semantic_order])
    else:
        order = reserved_slots(keyword_order, semantic_order, limit, VECTOR_SLOTS)
        fused = [(feature, 1.0 / rank) for rank, feature in enumerate(order, start=1)]
    hits = []
    for feature, score in fused[:limit]:
        hit = keyword_hits.get(feature)
        if hit is None:
            hit = document_hit(conn, semantic_documents[feature], score)
        hits.append(replace(hit, score=score))
    return hits


def vector_features(
    conn: sqlite3.Connection, question: str, limit: int, embedder: Embedder
) -> list[Hit]:
    """The vector channel alone, scored by cosine similarity: to see what it contributes."""
    if limit <= 0:
        return []
    query_vector = embedder.embed([question], "query")[0]
    return [
        document_hit(conn, document_id, similarity)
        for document_id, _, similarity in vector_ranking(conn, query_vector, embedder.name, limit)
    ]


def settings(conn: sqlite3.Connection | None = None) -> dict:
    """What decides hybrid results, for reports."""
    from glotto_lens import context

    result = {
        "embedder": context.EMBEDDER.name if context.EMBEDDER else None,
        "doc_model": DOC_MODEL,
        "query_model": QUERY_MODEL,
        "dimensions": DIMENSIONS,
        "max_chars": MAX_CHARS,
        "cache_version": CACHE_VERSION,
        "candidates": CANDIDATES,
        "fusion": FUSION,
        "vector_slots": VECTOR_SLOTS if FUSION == "slots" else None,
        "rrf_k": RRF_K if FUSION == "rrf" else None,
    }
    if conn is not None and context.EMBEDDER is not None:
        result.update(coverage(conn, context.EMBEDDER.name))
    return result


def coverage(conn: sqlite3.Connection, model: str) -> dict:
    """How many feature documents have a vector: a partly embedded database shows here."""
    documents = conn.execute("SELECT COUNT(*) FROM documents WHERE kind = 'feature'")
    try:
        # only vectors of current feature documents: stale ones must not hide missing ones
        vectors = conn.execute(
            "SELECT COUNT(*) FROM documents d JOIN document_vectors v"
            " ON v.document_id = d.id AND v.model = ? WHERE d.kind = 'feature'",
            (model,),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        vectors = 0
    return {"feature_documents": documents.fetchone()[0], "vectors": vectors}


def check_ready(conn: sqlite3.Connection, model: str) -> None:
    """Fail before any paid call when the feature documents are not all embedded."""
    counts = coverage(conn, model)
    if counts["vectors"] < counts["feature_documents"]:
        raise RuntimeError(
            f"{counts['vectors']} of {counts['feature_documents']} feature documents have"
            " vectors: run 'glotto-lens embed' first"
        )


def default_embedder() -> CachedEmbedder:
    return CachedEmbedder(YandexEmbedder())


def main(argv: list[str] | None = None, embedder: Embedder | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Embed the feature documents for hybrid search (Yandex text-search-doc)."
    )
    parser.parse_args(argv)
    try:
        embedder = embedder or default_embedder()
    except (RuntimeError, ValueError) as error:  # no key, or a bad YANDEX_TIMEOUT
        sys.exit(str(error))

    def progress(done: int, total: int) -> None:
        if done == total or done % 100 == 0:
            print(f"{done}/{total}", flush=True)

    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            added = embed_documents(conn, embedder, 1, progress)
            counts = coverage(conn, embedder.name)
    except RuntimeError as error:  # no API key, or a failed call (ModelError)
        sys.exit(f"Embedding stopped: {error}. Run again to continue where it stopped.")
    tokens = getattr(getattr(embedder, "embedder", embedder), "tokens", 0)
    print(
        f"Added {added} vectors; {counts['vectors']} of {counts['feature_documents']}"
        f" feature documents embedded; {tokens} tokens"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
