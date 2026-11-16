"""Build one SQLite database from the three CLDF archives."""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from glotto_lens.sources import GLOTTOLOG, SOURCES, Source, download, read_table

DB_PATH = Path("data/glottolens.sqlite")

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
    glottocode TEXT NOT NULL,
    feature_id TEXT NOT NULL REFERENCES features (id),
    value TEXT,
    code_id TEXT REFERENCES codes (id),
    comment TEXT
);
CREATE INDEX idx_values_glottocode ON feature_values (glottocode);
CREATE INDEX idx_values_feature ON feature_values (feature_id);
CREATE INDEX idx_names_name ON names (name COLLATE NOCASE);
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
    "Values: total / unknown ('?')": """
        SELECT source, COUNT(*), SUM(value = '?') FROM feature_values GROUP BY source""",
    "Glottolog features": """
        SELECT local_id, name FROM features WHERE source = 'glottolog'""",
    "Names by language tag (top 10)": """
        SELECT lang, COUNT(*) FROM names GROUP BY lang ORDER BY 2 DESC LIMIT 10""",
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
            if glottocodes.get(row["Language_ID"])
        ),
    )


def build(db_path: Path = DB_PATH, archives: dict[str, Path] | None = None) -> Path:
    """Create the database from scratch; archives are downloaded if not given."""
    archives = archives or {source.name: download(source) for source in SOURCES}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    partial = db_path.with_suffix(".part")
    partial.unlink(missing_ok=True)
    with closing(sqlite3.connect(partial)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        load_languoids(conn, archives[GLOTTOLOG.name])
        for source in SOURCES:
            load_structure(conn, source, archives[source.name])
    partial.replace(db_path)
    return db_path


def report(db_path: Path = DB_PATH) -> dict[str, list[tuple]]:
    """Run the sanity-check queries and return their rows by section title."""
    with closing(sqlite3.connect(db_path)) as conn:
        return {title: conn.execute(query).fetchall() for title, query in REPORT.items()}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path = build()
    print(f"Built {path} ({path.stat().st_size / 1024 / 1024:.0f} MB)")
    for title, rows in report(path).items():
        print(f"\n{title}")
        for row in rows:
            print("  " + " | ".join(str(cell) for cell in row))
