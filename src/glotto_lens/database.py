"""Build one SQLite database from the three CLDF archives."""

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from glotto_lens.chapters import parse_chapter
from glotto_lens.documents import load_documents
from glotto_lens.languages import load_languoid_facts, load_name_index
from glotto_lens.sources import (
    DB_PATH,
    GLOTTOLOG,
    SOURCES,
    WALS,
    Source,
    cldf_members,
    download,
    read_table,
    read_text,
)

SCHEMA = """
CREATE TABLE languoids (
    glottocode TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    level TEXT,
    macroarea TEXT,
    latitude REAL,
    longitude REAL,
    iso639_3 TEXT,
    family_id TEXT,
    parent_language_id TEXT,
    countries TEXT,
    is_isolate TEXT
);
CREATE TABLE names (
    glottocode TEXT NOT NULL,
    name TEXT NOT NULL,
    provider TEXT,
    lang TEXT
);
CREATE TABLE source_languages (
    source TEXT NOT NULL,
    source_language_id TEXT NOT NULL,
    name TEXT,
    glottocode TEXT,
    PRIMARY KEY (source, source_language_id)
);
CREATE TABLE features (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    local_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    chapter_id TEXT
);
CREATE TABLE codes (
    id TEXT PRIMARY KEY,
    feature_id TEXT NOT NULL REFERENCES features (id),
    name TEXT,
    description TEXT
);
CREATE TABLE feature_values (
    source TEXT NOT NULL,
    source_language_id TEXT NOT NULL,
    glottocode TEXT,
    feature_id TEXT NOT NULL REFERENCES features (id),
    value TEXT,
    code_id TEXT REFERENCES codes (id),
    comment TEXT
);
CREATE TABLE chapters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    contributors TEXT,
    citation TEXT,
    url TEXT NOT NULL
);
CREATE TABLE blocks (
    id INTEGER PRIMARY KEY,
    chapter_id TEXT NOT NULL REFERENCES chapters (id),
    position INTEGER NOT NULL,
    kind TEXT NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    feature_id TEXT REFERENCES features (id)
);
CREATE TABLE block_languages (
    block_id INTEGER NOT NULL REFERENCES blocks (id),
    glottocode TEXT NOT NULL
);
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    position INTEGER NOT NULL,
    feature_id TEXT REFERENCES features (id),
    url TEXT
);
CREATE TABLE document_chapters (
    document_id INTEGER NOT NULL REFERENCES documents (id),
    chapter_id TEXT NOT NULL REFERENCES chapters (id),
    PRIMARY KEY (document_id, chapter_id)
);
CREATE TABLE document_languages (
    document_id INTEGER NOT NULL REFERENCES documents (id),
    glottocode TEXT NOT NULL,
    PRIMARY KEY (document_id, glottocode)
);
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, section, text, content='documents', content_rowid='id', tokenize='porter unicode61'
);
CREATE TABLE name_index (
    key TEXT NOT NULL,
    glottocode TEXT NOT NULL REFERENCES languoids (glottocode),
    name TEXT NOT NULL,
    origin TEXT NOT NULL,
    PRIMARY KEY (key, glottocode, origin)
) WITHOUT ROWID;
CREATE TABLE languoid_facts (
    glottocode TEXT NOT NULL REFERENCES languoids (glottocode),
    source TEXT NOT NULL,
    own INTEGER NOT NULL,
    dialects INTEGER NOT NULL,
    PRIMARY KEY (glottocode, source)
) WITHOUT ROWID;
CREATE INDEX idx_values_glottocode ON feature_values (glottocode);
CREATE INDEX idx_values_feature_code ON feature_values (feature_id, code_id);
CREATE INDEX idx_languoids_parent ON languoids (parent_language_id);
CREATE INDEX idx_blocks_chapter ON blocks (chapter_id);
CREATE INDEX idx_blocks_feature ON blocks (feature_id);
CREATE INDEX idx_block_languages_glottocode ON block_languages (glottocode);
CREATE INDEX idx_block_languages_block ON block_languages (block_id);
CREATE INDEX idx_documents_feature ON documents (feature_id);
CREATE INDEX idx_document_chapters_chapter ON document_chapters (chapter_id);
CREATE INDEX idx_name_index_glottocode ON name_index (glottocode);
"""

REPORT = {
    "Languoids by level": """
        SELECT level, COUNT(*) FROM languoids GROUP BY level ORDER BY 2 DESC""",
    "Source languages: total / without glottocode / unknown to Glottolog": """
        SELECT s.source, COUNT(*), SUM(s.glottocode IS NULL),
               SUM(s.glottocode IS NOT NULL AND l.glottocode IS NULL)
        FROM source_languages s LEFT JOIN languoids l ON l.glottocode = s.glottocode
        GROUP BY s.source""",
    "Source languages by Glottolog level": """
        SELECT s.source, l.level, COUNT(*)
        FROM source_languages s JOIN languoids l ON l.glottocode = s.glottocode
        GROUP BY s.source, l.level""",
    "Glottocodes shared by WALS and Grambank": """
        SELECT COUNT(*) FROM (
            SELECT glottocode FROM source_languages
            WHERE source = 'wals' AND glottocode IS NOT NULL
            INTERSECT
            SELECT glottocode FROM source_languages WHERE source = 'grambank')""",
    "Values: total / unknown ('?') / language without glottocode": """
        SELECT source, COUNT(*), SUM(value = '?'), SUM(glottocode IS NULL)
        FROM feature_values GROUP BY source""",
    "Glottolog features": """
        SELECT local_id, name FROM features WHERE source = 'glottolog'""",
    "Names by language tag (top 10)": """
        SELECT lang, COUNT(*) FROM names GROUP BY lang ORDER BY 2 DESC LIMIT 10""",
    "Chapters / blocks / value tables matched to a feature": """
        SELECT (SELECT COUNT(*) FROM chapters), (SELECT COUNT(*) FROM blocks),
               (SELECT COUNT(*) FROM blocks WHERE feature_id IS NOT NULL)""",
    "Blocks by kind: count / characters": """
        SELECT kind, COUNT(*), SUM(LENGTH(text)) FROM blocks GROUP BY kind ORDER BY 2 DESC""",
    "WALS features: total / with chapter text / matched to a value table block": """
        SELECT COUNT(*), SUM(chapter_id IN (SELECT id FROM chapters)),
               SUM(id IN (SELECT feature_id FROM blocks WHERE feature_id IS NOT NULL))
        FROM features WHERE source = 'wals'""",
    "Languages linked from chapter text": """
        SELECT COUNT(DISTINCT glottocode) FROM block_languages""",
    "Documents by kind: count / shortest / average / longest (characters)": """
        SELECT kind, COUNT(*), MIN(LENGTH(text)), CAST(AVG(LENGTH(text)) AS INTEGER),
               MAX(LENGTH(text))
        FROM documents GROUP BY kind""",
    "Documents longer than 3000 characters, by kind": """
        SELECT kind, COUNT(*) FROM documents WHERE LENGTH(text) > 3000 GROUP BY kind""",
    "Documents shared by several chapters": """
        SELECT COUNT(*) FROM (
            SELECT document_id FROM document_chapters GROUP BY document_id HAVING COUNT(*) > 1)""",
    "Values without a code that are not '?'": """
        SELECT source, COUNT(*) FROM feature_values
        WHERE code_id IS NULL AND COALESCE(value, '') != '?' GROUP BY source""",
    "Longest feature documents": """
        SELECT feature_id, section, LENGTH(text) FROM documents
        WHERE kind = 'feature' ORDER BY 3 DESC LIMIT 5""",
    "Foreign key violations": """
        SELECT * FROM pragma_foreign_key_check""",
    "Grambank feature sections (top 15)": """
        SELECT section, COUNT(*) FROM documents WHERE feature_id LIKE 'grambank:%'
        GROUP BY section ORDER BY 2 DESC LIMIT 15""",
    "Name index: keys / languoids reachable / languages reachable": """
        SELECT COUNT(DISTINCT key), COUNT(DISTINCT glottocode),
               (SELECT COUNT(DISTINCT n.glottocode) FROM name_index n
                JOIN languoids l USING (glottocode) WHERE l.level = 'language')
        FROM name_index""",
    "Name index by origin": """
        SELECT origin, COUNT(*) FROM name_index GROUP BY origin ORDER BY 2 DESC""",
    "Languoids whose profile borrows a source from dialects": """
        SELECT source, COUNT(*) FROM languoid_facts WHERE own = 0 AND dialects > 0
        GROUP BY source""",
    "Glottocodes with several WALS entries": """
        SELECT COUNT(*) FROM (
            SELECT glottocode FROM source_languages WHERE source = 'wals'
            AND glottocode IS NOT NULL GROUP BY glottocode HAVING COUNT(*) > 1)""",
    "Glottolog status codes (aes, med)": """
        SELECT c.feature_id, c.name, c.description FROM codes c
        WHERE c.feature_id IN ('glottolog:aes', 'glottolog:med') ORDER BY c.rowid""",
}


def _text(value: str | None) -> str | None:
    return value or None


def _number(value: str | None) -> float | None:
    return float(value) if value else None


def load_languoids(conn: sqlite3.Connection, archive: Path) -> None:
    """Load the Glottolog languoid list and alternative names."""
    conn.executemany(
        "INSERT INTO languoids VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            (
                row["ID"],
                row["Name"],
                _text(row["Level"]),
                _text(row["Macroarea"]),
                _number(row["Latitude"]),
                _number(row["Longitude"]),
                _text(row["ISO639P3code"]),
                _text(row["Family_ID"]),
                _text(row["Language_ID"]),
                _text(row["Countries"]),
                _text(row["Is_Isolate"]),
            )
            for row in read_table(archive, "languages.csv")
        ),
    )
    conn.executemany(
        "INSERT INTO names VALUES (?, ?, ?, ?)",
        (
            (row["Language_ID"], row["Name"], _text(row["Provider"]), _text(row["lang"]))
            for row in read_table(archive, "names.csv")
        ),
    )


def load_structure(conn: sqlite3.Connection, source: Source, archive: Path) -> None:
    """Load languages, features, codes and values of one CLDF structure dataset."""
    name = source.name
    glottocodes = {}
    for row in read_table(archive, "languages.csv"):
        glottocodes[row["ID"]] = _text(row["Glottocode"])
        conn.execute(
            "INSERT INTO source_languages VALUES (?, ?, ?, ?)",
            (name, row["ID"], row["Name"], glottocodes[row["ID"]]),
        )
    conn.executemany(
        "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                f"{name}:{row['ID']}",
                name,
                row["ID"],
                row["Name"],
                _text(row["Description"]),
                _text(row.get("Chapter_ID")),
            )
            for row in read_table(archive, "parameters.csv")
        ),
    )
    conn.executemany(
        "INSERT INTO codes VALUES (?, ?, ?, ?)",
        (
            (
                f"{name}:{row['ID']}",
                f"{name}:{row['Parameter_ID']}",
                _text(row["Name"]),
                _text(row["Description"]),
            )
            for row in read_table(archive, "codes.csv")
        ),
    )
    conn.executemany(
        "INSERT INTO feature_values VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            (
                name,
                row["Language_ID"],
                glottocodes[row["Language_ID"]],
                f"{name}:{row['Parameter_ID']}",
                _text(row["Value"]),
                f"{name}:{row['Code_ID']}" if row["Code_ID"] else None,
                _text(row["Comment"]),
            )
            for row in read_table(archive, "values.csv")
        ),
    )


def load_chapters(conn: sqlite3.Connection, archive: Path) -> None:
    """Load WALS chapter texts as blocks linked to features and languages.

    Blocks keep the source text unchanged; computed numbers live only in documents.
    """
    members = cldf_members(archive)
    features = {row[0] for row in conn.execute("SELECT id FROM features")}
    glottocodes = dict(
        conn.execute(
            "SELECT source_language_id, glottocode FROM source_languages "
            "WHERE source = 'wals' AND glottocode IS NOT NULL"
        )
    )
    for row in read_table(archive, "chapters.csv"):
        path = f"docs/chapter_{row['ID']}.html"
        if path not in members:
            continue
        page = "supplement" if row["ID"].startswith("s") else "chapter"
        url = f"https://wals.info/{page}/{row['ID'].removeprefix('s')}"
        conn.execute(
            "INSERT INTO chapters VALUES (?, ?, ?, ?, ?)",
            (row["ID"], row["Name"], _text(row["Contributor"]), _text(row["Citation"]), url),
        )
        for position, block in enumerate(parse_chapter(read_text(archive, path))):
            feature_id = f"wals:{block.feature}" if block.feature else None
            if feature_id not in features:
                feature_id = None
            cursor = conn.execute(
                "INSERT INTO blocks (chapter_id, position, kind, section, text, feature_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (row["ID"], position, block.kind, block.section, block.text, feature_id),
            )
            conn.executemany(
                "INSERT INTO block_languages VALUES (?, ?)",
                (
                    (cursor.lastrowid, glottocodes[code])
                    for code in block.languages
                    if code in glottocodes
                ),
            )


def build(db_path: Path = DB_PATH, archives: dict[str, Path] | None = None) -> Path:
    """Create the database from scratch; archives are downloaded if not given."""
    archives = archives or {source.name: download(source) for source in SOURCES}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    partial = db_path.with_suffix(".part")
    partial.unlink(missing_ok=True)
    with closing(sqlite3.connect(partial)) as conn, conn:
        # The partial file is discarded on failure, so durability can be traded for speed.
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        load_languoids(conn, archives[GLOTTOLOG.name])
        for source in SOURCES:
            load_structure(conn, source, archives[source.name])
        load_chapters(conn, archives[WALS.name])
        load_documents(conn)
        load_name_index(conn)
        load_languoid_facts(conn)
        conn.execute("ANALYZE")
    partial.replace(db_path)
    return db_path


def report(db_path: Path = DB_PATH) -> dict[str, list[tuple]]:
    """Run the sanity-check queries and return their rows by section title."""
    with closing(sqlite3.connect(db_path)) as conn:
        return {title: conn.execute(query).fetchall() for title, query in REPORT.items()}


def main(argv: list[str] | None = None) -> None:
    """Download the pinned releases if needed, build the database and print the report."""
    parser = argparse.ArgumentParser(
        description="Download the pinned releases (once) and build the SQLite database."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH, help="database path")
    args = parser.parse_args(argv)
    path = build(args.db)
    print(f"Built {path} ({path.stat().st_size / 1024 / 1024:.0f} MB)")
    for title, rows in report(path).items():
        print(f"\n{title}")
        for row in rows:
            print("  " + " | ".join(str(cell) for cell in row))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
