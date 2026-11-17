"""Find languages by name or code and collect what the databases say about them."""

import sqlite3
import sys
import unicodedata
from contextlib import closing
from dataclasses import dataclass, field

STATUS_FEATURES = ("aes", "med")

NAME_SOURCES = """
    SELECT glottocode, name, 'glottolog' FROM languoids
    UNION ALL
    SELECT n.glottocode, n.name, 'alternative' FROM names n JOIN languoids l USING (glottocode)
    UNION ALL
    SELECT s.glottocode, s.name, s.source
    FROM source_languages s JOIN languoids l USING (glottocode)
    WHERE s.source IN ('wals', 'grambank')
    UNION ALL
    SELECT glottocode, iso639_3, 'iso' FROM languoids WHERE iso639_3 IS NOT NULL
    UNION ALL
    SELECT glottocode, glottocode, 'glottocode' FROM languoids"""

# Exact codes first, then primary names, then alternative names. Within a tier, languages
# come before dialects and families, and better documented languages come first.
CANDIDATES = """
    SELECT l.glottocode, l.name, l.level, l.macroarea, f.name, l.iso639_3,
           (SELECT COUNT(*) FROM feature_values v
            WHERE v.glottocode = l.glottocode AND v.source IN ('wals', 'grambank')) AS facts,
           MIN(CASE n.origin WHEN 'glottocode' THEN 0 WHEN 'iso' THEN 0
                             WHEN 'alternative' THEN 2 ELSE 1 END) AS tier
    FROM name_index n
    JOIN languoids l USING (glottocode)
    LEFT JOIN languoids f ON f.glottocode = l.family_id
    WHERE n.key = ?
    GROUP BY l.glottocode
    ORDER BY tier,
             CASE l.level WHEN 'language' THEN 0 WHEN 'dialect' THEN 1 ELSE 2 END,
             facts DESC, l.glottocode
    LIMIT ?"""

LANGUOID = """
    SELECT l.glottocode, l.name, l.level, l.macroarea, f.name, l.iso639_3,
           (SELECT COUNT(*) FROM feature_values v
            WHERE v.glottocode = l.glottocode AND v.source IN ('wals', 'grambank'))
    FROM languoids l LEFT JOIN languoids f ON f.glottocode = l.family_id
    WHERE l.glottocode = ?"""

VALUE_LABEL = """COALESCE(c.description, c.name,
                 CASE v.value WHEN '?' THEN 'not known (?)' ELSE v.value END)"""

FACTS = f"""
    SELECT v.source, f.local_id, f.name, {VALUE_LABEL}, v.comment
    FROM feature_values v
    JOIN features f ON f.id = v.feature_id
    LEFT JOIN codes c ON c.id = v.code_id
    WHERE v.glottocode = ? AND v.source IN ('wals', 'grambank')
    ORDER BY v.source = 'grambank', f.rowid, v.source_language_id"""

STATUS = f"""
    SELECT f.local_id, {VALUE_LABEL}
    FROM feature_values v
    JOIN features f ON f.id = v.feature_id
    LEFT JOIN codes c ON c.id = v.code_id
    WHERE v.glottocode = ? AND v.source = 'glottolog'
      AND f.local_id IN ({", ".join("?" * len(STATUS_FEATURES))})"""


@dataclass(frozen=True)
class Languoid:
    glottocode: str
    name: str
    level: str | None
    macroarea: str | None
    family: str | None
    iso639_3: str | None
    facts: int


@dataclass(frozen=True)
class Fact:
    source: str
    feature: str
    name: str
    value: str | None
    comment: str | None


@dataclass
class Profile:
    languoid: Languoid
    status: dict[str, str] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)


def normalize_name(name: str) -> str:
    """Make a lookup key: no diacritics, case folded, punctuation turned into spaces."""
    decomposed = unicodedata.normalize("NFKD", name)
    letters = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = "".join(char if char.isalnum() else " " for char in letters.casefold())
    return " ".join(cleaned.split())


def load_name_index(conn: sqlite3.Connection) -> None:
    """Index every known name and code of every languoid under its normalized key."""
    conn.executemany(
        "INSERT OR IGNORE INTO name_index VALUES (?, ?, ?, ?)",
        (
            (key, glottocode, name, origin)
            for glottocode, name, origin in conn.execute(NAME_SOURCES).fetchall()
            if (key := normalize_name(name))
        ),
    )


def find_languages(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[Languoid]:
    """Return languoids whose name or code matches the query, best candidate first."""
    key = normalize_name(query)
    if not key:
        return []
    rows = conn.execute(CANDIDATES, (key, limit)).fetchall()
    return [Languoid(*row[:7]) for row in rows]


def language_profile(conn: sqlite3.Connection, glottocode: str) -> Profile | None:
    """Collect status and all WALS and Grambank values recorded for one languoid."""
    row = conn.execute(LANGUOID, (glottocode,)).fetchone()
    if row is None:
        return None
    status = dict(conn.execute(STATUS, (glottocode, *STATUS_FEATURES)).fetchall())
    facts = [Fact(*fact) for fact in conn.execute(FACTS, (glottocode,))]
    return Profile(Languoid(*row), status, facts)


def main(queries: list[str]) -> None:
    from glotto_lens.database import DB_PATH

    with closing(sqlite3.connect(DB_PATH)) as conn:
        for query in queries:
            candidates = find_languages(conn, query)
            print(f"\n=== {query!r} -> key {normalize_name(query)!r}")
            for languoid in candidates:
                print(
                    f"  {languoid.glottocode} | {languoid.name} | {languoid.level} | "
                    f"{languoid.family} | {languoid.macroarea} | {languoid.facts} facts"
                )
            if not candidates:
                continue
            profile = language_profile(conn, candidates[0].glottocode)
            print(f"  status: {profile.status}")
            for source in ("wals", "grambank"):
                facts = [fact for fact in profile.facts if fact.source == source]
                print(f"  {source}: {len(facts)} facts")
                for fact in facts[:5]:
                    print(f"    {fact.feature} {fact.name}: {fact.value}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
