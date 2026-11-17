"""Find languages by name or code and collect what the databases say about them."""

import sqlite3
import sys
import unicodedata
from contextlib import closing
from dataclasses import dataclass, field

from glotto_lens.sources import DB_PATH

STATUS_FEATURES = ("aes", "med")
# Words that name the category rather than the language: "Russian language" -> "russian".
FILLER_WORDS = frozenset({"language", "languages", "lang", "язык", "языка", "языке", "языки"})

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

# A value counts as known only when it has a code: '?' says nothing about the language.
KNOWN_FACTS = """(SELECT COUNT(*) FROM feature_values v
                  WHERE v.glottocode = l.glottocode AND v.source IN ('wals', 'grambank')
                    AND v.code_id IS NOT NULL)"""

# Tiers: 0 exact code, 1 primary name, 2 alternative name, 3 name starting with the query.
# Codes win outright; otherwise languages come before dialects and families, then the
# better tier, then the better documented languoid.
CANDIDATES = f"""
    SELECT l.glottocode, l.name, l.level, l.macroarea, f.name, l.iso639_3,
           {KNOWN_FACTS} AS facts,
           MIN(CASE WHEN n.key != :key THEN 3
                    WHEN n.origin IN ('glottocode', 'iso') THEN 0
                    WHEN n.origin = 'alternative' THEN 2 ELSE 1 END) AS tier
    FROM name_index n
    JOIN languoids l USING (glottocode)
    LEFT JOIN languoids f ON f.glottocode = l.family_id
    WHERE {{condition}}
    GROUP BY l.glottocode
    ORDER BY tier > 0,
             CASE l.level WHEN 'language' THEN 0 WHEN 'dialect' THEN 1 ELSE 2 END,
             tier, facts DESC, l.glottocode
    LIMIT :limit"""
EXACT = CANDIDATES.format(condition="n.key = :key")
# "russian" also finds "russian sign"; the range uses the primary key index.
PREFIX = CANDIDATES.format(condition="n.key >= :key || ' ' AND n.key < :key || '!'")

LANGUOID = f"""
    SELECT l.glottocode, l.name, l.level, l.macroarea, f.name, l.iso639_3, {KNOWN_FACTS}
    FROM languoids l LEFT JOIN languoids f ON f.glottocode = l.family_id
    WHERE l.glottocode = ?"""

FACTS = """
    SELECT v.source, f.local_id, f.name,
           COALESCE(c.description, c.name,
                    CASE v.value WHEN '?' THEN 'not known (?)' ELSE v.value END),
           v.comment, s.name
    FROM feature_values v
    JOIN features f ON f.id = v.feature_id
    JOIN source_languages s
      ON s.source = v.source AND s.source_language_id = v.source_language_id
    LEFT JOIN codes c ON c.id = v.code_id
    WHERE v.glottocode IN ({glottocodes}) AND v.source IN ('wals', 'grambank')
    ORDER BY CASE v.source WHEN 'wals' THEN 0 ELSE 1 END, f.rowid, s.name"""
OWN_FACTS = FACTS.format(glottocodes="?")
DIALECT_FACTS = FACTS.format(
    glottocodes="SELECT glottocode FROM languoids WHERE parent_language_id = ? "
    "AND level = 'dialect'"
)

# Glottolog codes have a short name ("threatened") and a longer description (scales).
STATUS = f"""
    SELECT f.local_id, COALESCE(c.name || COALESCE(' (' || c.description || ')', ''), v.value)
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
    entry: str


@dataclass
class Profile:
    languoid: Languoid
    status: dict[str, str | None] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    from_dialects: bool = False


def normalize_name(name: str) -> str:
    """Make a lookup key: no diacritics, case folded, punctuation and filler words dropped."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    letters = "".join(char for char in decomposed if not unicodedata.combining(char))
    words = "".join(char if char.isalnum() else " " for char in letters).split()
    return " ".join(word for word in words if word not in FILLER_WORDS)


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
    """Return languoids whose name or code matches the query, best candidate first.

    Names that only start with the query are tried when nothing matches exactly.
    """
    key = normalize_name(query)
    if not key:
        return []
    params = {"key": key, "limit": limit}
    rows = conn.execute(EXACT, params).fetchall() or conn.execute(PREFIX, params).fetchall()
    return [Languoid(*row[:7]) for row in rows]


def language_profile(conn: sqlite3.Connection, glottocode: str) -> Profile | None:
    """Collect status and all WALS and Grambank values recorded for one languoid.

    A language without values of its own gets the values of its dialects; every fact
    names the dataset entry it comes from, so conflicting entries stay visible.
    """
    row = conn.execute(LANGUOID, (glottocode,)).fetchone()
    if row is None:
        return None
    status = dict(conn.execute(STATUS, (glottocode, *STATUS_FEATURES)).fetchall())
    facts = [Fact(*fact) for fact in conn.execute(OWN_FACTS, (glottocode,))]
    from_dialects = not facts
    if from_dialects:
        facts = [Fact(*fact) for fact in conn.execute(DIALECT_FACTS, (glottocode,))]
    return Profile(Languoid(*row), status, facts, from_dialects and bool(facts))


def main(queries: list[str]) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        for query in queries:
            candidates = find_languages(conn, query)
            print(f"\n=== {query!r} -> key {normalize_name(query)!r}")
            for languoid in candidates:
                print(
                    f"  {languoid.glottocode} | {languoid.name} | {languoid.level} | "
                    f"{languoid.family} | {languoid.macroarea} | {languoid.facts} known facts"
                )
            profile = candidates and language_profile(conn, candidates[0].glottocode)
            if not profile:
                continue
            print(f"  status: {profile.status}")
            if profile.from_dialects:
                print("  values come from dialects")
            for source in ("wals", "grambank"):
                facts = [fact for fact in profile.facts if fact.source == source]
                entries = sorted({fact.entry for fact in facts})
                print(f"  {source}: {len(facts)} values from {entries}")
                for fact in facts[:3]:
                    print(f"    {fact.feature} {fact.name}: {fact.value}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
