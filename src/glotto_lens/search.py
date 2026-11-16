"""Full-text search over the indexed documents."""

import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from glotto_lens.database import DB_PATH

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the to what "
    "which with".split()
)

SEARCH = """
    SELECT d.id, -bm25(documents_fts, 3.0, 2.0, 1.0) AS score, d.title, d.section, d.text
    FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid
    WHERE documents_fts MATCH ? ORDER BY score DESC LIMIT ?"""

URLS = """
    SELECT c.url FROM document_chapters dc JOIN chapters c ON c.id = dc.chapter_id
    WHERE dc.document_id = ? ORDER BY c.rowid"""


@dataclass(frozen=True)
class Hit:
    document_id: int
    score: float
    title: str
    section: str
    text: str
    urls: tuple[str, ...]


def to_match_query(question: str) -> str:
    """Turn free text into a safe FTS5 query: quoted words joined with OR."""
    words = [word for word in re.findall(r"\w+", question.lower()) if word not in STOPWORDS]
    return " OR ".join(f'"{word}"' for word in dict.fromkeys(words))


def search(conn: sqlite3.Connection, question: str, limit: int = 5) -> list[Hit]:
    """Return the best-matching documents for the question, ranked by BM25."""
    match = to_match_query(question)
    if not match:
        return []
    hits = []
    for document_id, score, title, section, text in conn.execute(SEARCH, (match, limit)):
        urls = tuple(row[0] for row in conn.execute(URLS, (document_id,)))
        hits.append(Hit(document_id, score, title, section, text, urls))
    return hits


def main(question: str, db_path: Path = DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        hits = search(conn, question)
    print(f"Query: {to_match_query(question)}")
    for rank, hit in enumerate(hits, start=1):
        print(f"\n{rank}. [{hit.score:.1f}] {hit.title} / {hit.section}")
        print("   " + ", ".join(hit.urls))
        print("   " + hit.text[:400].replace("\n", " "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(" ".join(sys.argv[1:]))
