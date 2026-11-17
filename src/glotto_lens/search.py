"""Full-text search over the indexed documents."""

import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from glotto_lens.sources import DB_PATH

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the to what "
    "which with".split()
)

SEARCH = """
    SELECT d.id, -bm25(documents_fts, 3.0, 2.0, 1.0) AS score, d.kind, d.title, d.section,
           d.text, d.url, snippet(documents_fts, 2, '[', ']', ' ... ', 32), d.feature_id
    FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid
    WHERE documents_fts MATCH :match {language_filter}
    ORDER BY score DESC, d.id LIMIT :limit"""
ANY_DOCUMENT = SEARCH.format(language_filter="")
# Only chapter texts that mention the language: feature cards are about all languages.
MENTIONING = SEARCH.format(
    language_filter="AND d.id IN (SELECT document_id FROM document_languages "
    "WHERE glottocode = :glottocode)"
)

URLS = """
    SELECT c.url FROM document_chapters dc JOIN chapters c ON c.id = dc.chapter_id
    WHERE dc.document_id = ? ORDER BY c.rowid"""


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
    conn: sqlite3.Connection, question: str, limit: int = 5, glottocode: str | None = None
) -> list[Hit]:
    """Return the best-matching documents for the question, ranked by BM25.

    With a glottocode, only chapter texts that mention that language are searched.
    """
    match = to_match_query(question)
    if not match:
        return []
    query = MENTIONING if glottocode else ANY_DOCUMENT
    params = {"match": match, "limit": limit, "glottocode": glottocode}
    hits = []
    for row in conn.execute(query, params).fetchall():
        document_id, score, kind, title, section, text, url, snippet, feature_id = row
        urls = [url] if url else []
        urls += [url_row[0] for url_row in conn.execute(URLS, (document_id,))]
        urls = tuple(dict.fromkeys(urls))
        hits.append(Hit(document_id, score, kind, title, section, text, urls, snippet, feature_id))
    return hits


def main(question: str, db_path: Path = DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        hits = search(conn, question)
    print(f"Query: {to_match_query(question)}")
    for rank, hit in enumerate(hits, start=1):
        print(f"\n{rank}. [{hit.score:.1f}] {hit.kind}: {hit.title} / {hit.section}")
        print("   " + ", ".join(hit.urls))
        print("   " + hit.snippet.replace("\n", " "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(" ".join(sys.argv[1:]))
