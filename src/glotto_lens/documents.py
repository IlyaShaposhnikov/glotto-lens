"""Group chapter blocks into retrieval documents and index them for full-text search."""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field

MAX_CHARS = 1500


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


def load_documents(conn: sqlite3.Connection) -> None:
    """Turn chapter blocks into documents, merge identical ones and build the FTS index."""
    seen: dict[tuple[str, str], int] = {}
    chapters = conn.execute("SELECT id, name FROM chapters ORDER BY rowid").fetchall()
    for chapter_id, name in chapters:
        blocks = conn.execute(
            "SELECT id, section, kind, text FROM blocks WHERE chapter_id = ? ORDER BY position",
            (chapter_id,),
        ).fetchall()
        for chunk in chunk_blocks(blocks):
            key = (chunk.section, chunk.text)
            if key not in seen:
                cursor = conn.execute(
                    "INSERT INTO documents (kind, title, section, text) VALUES (?, ?, ?, ?)",
                    ("chapter", name, chunk.section, chunk.text),
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
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
