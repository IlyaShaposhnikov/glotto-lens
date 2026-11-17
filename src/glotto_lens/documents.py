"""Build retrieval documents: chapter chunks and feature cards, plus the FTS index."""

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field

from glotto_lens.sources import SOURCES

MAX_CHARS = 1500
DISPLAY_NAMES = {"wals": "WALS", "grambank": "Grambank"}
RELEASES = {
    source.name: f"{DISPLAY_NAMES[source.name]} {source.tag}"
    for source in SOURCES
    if source.name in DISPLAY_NAMES
}
FEATURE_URLS = {
    "wals": "https://wals.info/feature/{}",
    "grambank": "https://grambank.clld.org/parameters/{}",
}
SUMMARY = "summary"
# Bibliography, coder names and cross-links: identical boilerplate that only adds noise.
SKIPPED_SECTIONS = frozenset(
    {SUMMARY, "further reading", "references", "patron", "patrons", "related features"}
)

VALUE_COUNTS = """
    SELECT COALESCE(c.description, c.name),
           (SELECT COUNT(*) FROM feature_values v
            WHERE v.feature_id = c.feature_id AND v.code_id = c.id)
    FROM codes c WHERE c.feature_id = ? ORDER BY c.rowid"""

UNKNOWN_COUNT = "SELECT COUNT(*) FROM feature_values WHERE feature_id = ? AND code_id IS NULL"

FEATURES = """
    SELECT f.id, f.source, f.local_id, f.name, f.description, c.id, c.name, c.contributors
    FROM features f LEFT JOIN chapters c ON c.id = f.chapter_id AND f.source = 'wals'
    WHERE f.source IN ('wals', 'grambank') ORDER BY f.rowid"""

INSERT_DOCUMENT = """
    INSERT INTO documents (kind, title, section, text, position, feature_id, url)
    VALUES (?, ?, ?, ?, ?, ?, ?)"""


@dataclass
class Chunk:
    section: str
    block_ids: list[int] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)


def chunk_blocks(
    blocks: Iterable[tuple[int, str, str, str]], max_chars: int = MAX_CHARS
) -> list[Chunk]:
    """Group (id, section, kind, text) rows of one chapter into chunks.

    A chunk never crosses a section boundary. Only a paragraph can open a new chunk,
    so examples, tables and figure captions stay with the text that introduces them.
    """
    chunks: list[Chunk] = []
    size = 0
    for block_id, section, kind, text in blocks:
        too_long = kind == "paragraph" and size + len(text) > max_chars
        if not chunks or section != chunks[-1].section or too_long:
            chunks.append(Chunk(section))
            size = 0
        chunks[-1].block_ids.append(block_id)
        chunks[-1].parts.append(text)
        size += len(text)
    return chunks


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown text on "## " headings into (heading, body) pairs with a body."""
    sections = []
    heading, lines = "", []
    for line in markdown.splitlines():
        if line.startswith("## "):
            sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines).strip()))
    return [(heading, body) for heading, body in sections if body]


def pack(parts: Iterable[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Join consecutive parts into texts of at most max_chars; a longer part stays whole."""
    groups: list[list[str]] = []
    size = 0
    for part in parts:
        if not groups or size + len(part) > max_chars:
            groups.append([])
            size = 0
        groups[-1].append(part)
        size += len(part)
    return ["\n\n".join(group) for group in groups]


def _languages(count: int) -> str:
    return f"{count} language" if count == 1 else f"{count} languages"


def value_distribution(conn: sqlite3.Connection, feature_id: str, release: str) -> str:
    """Render the value counts of a feature, labelled as computed from the data."""
    counts = conn.execute(VALUE_COUNTS, (feature_id,)).fetchall()
    unknown = conn.execute(UNKNOWN_COUNT, (feature_id,)).fetchone()[0]
    lines = [f"Distribution in {release} data:"]
    lines += [f"- {label}: {_languages(count)}" for label, count in counts]
    if unknown:
        lines.append(f"- not known (?): {_languages(unknown)}")
    lines.append(f"Total: {_languages(unknown + sum(count for _, count in counts))}")
    return "\n".join(lines)


def load_chapter_documents(conn: sqlite3.Connection) -> None:
    """Chunk chapter blocks; identical chunks of several chapters are stored once.

    Value tables are left out: every feature gets its own card with the distribution.
    """
    seen: dict[tuple[str, str], int] = {}
    chapters = conn.execute("SELECT id, name FROM chapters ORDER BY rowid").fetchall()
    for chapter_id, name in chapters:
        blocks = conn.execute(
            "SELECT id, section, kind, text FROM blocks "
            "WHERE chapter_id = ? AND kind != 'values' ORDER BY position",
            (chapter_id,),
        ).fetchall()
        for position, chunk in enumerate(chunk_blocks(blocks)):
            key = (chunk.section, chunk.text)
            if key not in seen:
                cursor = conn.execute(
                    INSERT_DOCUMENT,
                    ("chapter", name, chunk.section, chunk.text, position, None, None),
                )
                seen[key] = cursor.lastrowid
                marks = ", ".join("?" * len(chunk.block_ids))
                conn.execute(
                    "INSERT OR IGNORE INTO document_languages "
                    "SELECT ?, glottocode FROM block_languages "
                    f"WHERE block_id IN ({marks})",
                    (seen[key], *chunk.block_ids),
                )
            conn.execute(
                "INSERT OR IGNORE INTO document_chapters VALUES (?, ?)", (seen[key], chapter_id)
            )


def load_feature_documents(conn: sqlite3.Connection) -> None:
    """Create a card per WALS and Grambank feature and documents for its other sections."""
    for row in conn.execute(FEATURES).fetchall():
        feature_id, source, local_id, name, description, chapter_id, chapter, contributors = row
        sections = split_sections(description or "")
        summary = "\n\n".join(body for heading, body in sections if heading.lower() == SUMMARY)
        header = f"{DISPLAY_NAMES[source]} feature {local_id}: {name}"
        if chapter_id:
            author = f" by {contributors}" if contributors else ""
            header += f'\nDescribed in the WALS chapter "{chapter}"{author}.'
        distribution = value_distribution(conn, feature_id, RELEASES[source])
        card = "\n\n".join(part for part in (header, summary, distribution) if part)
        url = FEATURE_URLS[source].format(local_id)
        cursor = conn.execute(
            INSERT_DOCUMENT, ("feature", name, "Summary", card, 0, feature_id, url)
        )
        if chapter_id:
            conn.execute(
                "INSERT INTO document_chapters VALUES (?, ?)", (cursor.lastrowid, chapter_id)
            )
        position = 1
        for heading, body in sections:
            if heading.lower() in SKIPPED_SECTIONS:
                continue
            for text in pack(re.split(r"\n\s*\n", body)):
                conn.execute(
                    INSERT_DOCUMENT,
                    ("feature", name, heading or "Description", text, position, feature_id, url),
                )
                position += 1


def load_documents(conn: sqlite3.Connection) -> None:
    """Fill the documents table from chapters and features and build the FTS index."""
    load_chapter_documents(conn)
    load_feature_documents(conn)
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
