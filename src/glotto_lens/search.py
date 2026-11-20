"""Full-text search over the indexed documents."""

import argparse
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass

from glotto_lens.cliargs import non_negative
from glotto_lens.sources import DB_PATH

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the to what "
    "which with".split()
)

MATCHES = """
    SELECT d.id AS id, -bm25(documents_fts, 3.0, 2.0, 1.0) AS score, d.kind AS kind,
           d.title AS title, d.section AS section, d.text AS text, d.url AS url,
           snippet(documents_fts, 2, '[', ']', ' ... ', 32) AS snippet, d.feature_id AS feature_id
    FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid
    WHERE {conditions}"""
SEARCH = MATCHES + "\n    ORDER BY score DESC, id LIMIT :limit"
# The best document of each feature: a Grambank feature has several section documents,
# which would otherwise fill the results with one feature.
PER_FEATURE = f"""
    SELECT id, score, kind, title, section, text, url, snippet, feature_id FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY feature_id ORDER BY score DESC, id) AS rank
        FROM ({MATCHES}))
    WHERE rank = 1
    ORDER BY score DESC, id LIMIT :limit"""
# Only chapter texts that mention the language: feature cards are about all languages.
MENTIONS_LANGUAGE = (
    "d.id IN (SELECT document_id FROM document_languages WHERE glottocode = :glottocode)"
)

URLS = """
    SELECT c.url FROM document_chapters dc JOIN chapters c ON c.id = dc.chapter_id
    WHERE dc.document_id = ? ORDER BY c.rowid"""
DOCUMENT = "SELECT kind, title, section, text, url, feature_id FROM documents WHERE id = ?"
SNIPPET_CHARS = 200


@dataclass(frozen=True)
class Hit:
    document_id: int
    score: float
    kind: str
    title: str
    section: str
    text: str
    urls: tuple[str, ...]
    snippet: str
    feature_id: str | None = None


def to_match_query(question: str) -> str:
    """Turn free text into a safe FTS5 query: quoted words joined with OR."""
    words = [word for word in re.findall(r"\w+", question.lower()) if word not in STOPWORDS]
    return " OR ".join(f'"{word}"' for word in dict.fromkeys(words))


def search(
    conn: sqlite3.Connection,
    question: str,
    limit: int = 5,
    glottocode: str | None = None,
    kind: str | None = None,
    per_feature: bool = False,
) -> list[Hit]:
    """Return the best-matching documents for the question, ranked by BM25.

    With a glottocode, only chapter texts that mention that language are searched; with a
    kind, only documents of that kind ("chapter" or "feature"). BM25 uses statistics of the
    whole index, so scores of differently filtered searches can be compared.
    With per_feature, only feature documents are searched and each feature brings only its
    best document, so the limit counts features.
    """
    match = to_match_query(question)
    if not match:
        return []
    conditions = ["documents_fts MATCH :match"]
    if glottocode:
        conditions.append(MENTIONS_LANGUAGE)
    if per_feature:
        kind = "feature"
    if kind:
        conditions.append("d.kind = :kind")
    query = (PER_FEATURE if per_feature else SEARCH).format(conditions=" AND ".join(conditions))
    params = {"match": match, "limit": limit, "glottocode": glottocode, "kind": kind}
    hits = []
    for row in conn.execute(query, params).fetchall():
        document_id, score, kind_, title, section, text, url, snippet, feature_id = row
        urls = _urls(conn, document_id, url)
        hits.append(Hit(document_id, score, kind_, title, section, text, urls, snippet, feature_id))
    return hits


def _urls(conn: sqlite3.Connection, document_id: int, url: str | None) -> tuple[str, ...]:
    urls = [url] if url else []
    urls += [url_row[0] for url_row in conn.execute(URLS, (document_id,))]
    return tuple(dict.fromkeys(urls))


def document_hit(conn: sqlite3.Connection, document_id: int, score: float) -> Hit:
    """A hit for a document found without full-text search (by its vector)."""
    row = conn.execute(DOCUMENT, (document_id,)).fetchone()
    if row is None:
        raise RuntimeError(
            f"document {document_id} has a vector but no text: rebuild, then 'glotto-lens embed'"
        )
    kind, title, section, text, url, feature_id = row
    snippet = " ".join(text.split())[:SNIPPET_CHARS]
    urls = _urls(conn, document_id, url)
    return Hit(document_id, score, kind, title, section, text, urls, snippet, feature_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full-text search over the documents.")
    parser.add_argument("query", nargs="+", help="words to search for, in English")
    parser.add_argument("--limit", type=non_negative, default=5, help="number of results")
    parser.add_argument("--kind", choices=("chapter", "feature"), help="only this kind")
    parser.add_argument(
        "--per-feature",
        action="store_true",
        help="best document of each feature, as the evidence is chosen",
    )
    channels = parser.add_mutually_exclusive_group()
    channels.add_argument(
        "--hybrid",
        action="store_true",
        help="features ranked by BM25 and embeddings together (needs 'glotto-lens embed')",
    )
    channels.add_argument(
        "--vectors-only", action="store_true", help="features ranked by embeddings alone"
    )
    args = parser.parse_args(argv)
    vector = args.hybrid or args.vectors_only
    if (args.per_feature or vector) and args.kind == "chapter":
        parser.error("--per-feature, --hybrid and --vectors-only rank feature documents")
    question = " ".join(args.query)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if vector:
            from glotto_lens import vectors

            try:
                embedder = vectors.default_embedder()
                vectors.check_ready(conn, embedder.name)  # no silent half-embedded search
                rank = vectors.hybrid_features if args.hybrid else vectors.vector_features
                hits = rank(conn, question, args.limit, embedder)
            except RuntimeError as error:
                sys.exit(str(error))
        else:
            hits = search(conn, question, args.limit, kind=args.kind, per_feature=args.per_feature)
    if args.vectors_only:
        print(f"Vector query: {question}")
    else:
        print(f"Query: {to_match_query(question)}")
    for rank, hit in enumerate(hits, start=1):
        feature = f" ({hit.feature_id})" if hit.feature_id else ""
        print(f"\n{rank}. [{hit.score:.3g}] {hit.kind}{feature}: {hit.title} / {hit.section}")
        print("   " + ", ".join(hit.urls))
        print("   " + hit.snippet.replace("\n", " "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
