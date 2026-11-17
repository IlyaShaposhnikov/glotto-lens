import sqlite3
from contextlib import closing

import pytest

from glotto_lens.languages import Fact, find_languages, language_profile, normalize_name


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


def codes(languoids):
    return [languoid.glottocode for languoid in languoids]


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Nadëb", "nadeb"),
        ("Русский язык", "русскии"),
        ("Russian Sign Language", "russian sign"),
        ("Bobo Madaré (Northern)", "bobo madare northern"),
        ("Ёлки-Палки", "елки палки"),
        ("  ", ""),
    ],
)
def test_normalize_name(name, key):
    assert normalize_name(name) == key


def test_alternative_name_in_cyrillic_is_found(conn):
    assert codes(find_languages(conn, "Русский")) == ["russ1263"]


@pytest.mark.parametrize("query", ["RUS", "russ1263", "russian", "Russian language"])
def test_codes_and_names_are_found(conn, query):
    assert codes(find_languages(conn, query)) == ["russ1263"]


def test_language_comes_before_dialect_of_the_same_name(conn):
    found = find_languages(conn, "Básque")
    assert [(languoid.glottocode, languoid.level) for languoid in found] == [
        ("basq1248", "language"),
        ("basq9999", "dialect"),
    ]


def test_candidate_counts_only_known_facts(conn):
    russian, basque = find_languages(conn, "Russian")[0], find_languages(conn, "Basque")[0]
    assert (russian.facts, basque.facts) == (2, 1)


def test_name_prefix_is_used_when_nothing_matches_exactly(conn):
    assert codes(find_languages(conn, "Indo")) == ["indo1319"]


def test_unknown_or_empty_query_finds_nothing(conn):
    assert find_languages(conn, "Klingon") == []
    assert find_languages(conn, "?!") == []


def test_profile_lists_wals_then_grambank_facts_with_readable_values(conn):
    profile = language_profile(conn, "basq1248")
    assert profile.facts == [
        Fact("wals", "1A", "Feature", "no", None, "Basque", "bsq"),
        Fact("grambank", "GB020", "Feature", "not known (?)", None, "Basque", "basq1248"),
    ]


def test_several_entries_of_one_language_are_all_kept(conn):
    conn.execute(
        "INSERT INTO source_languages VALUES ('wals', 'rus2', 'Russian (Old)', 'russ1263')"
    )
    conn.execute(
        "INSERT INTO feature_values VALUES ('wals', 'rus2', 'russ1263', 'wals:1A', '2', "
        "'wals:1A-2', NULL)"
    )
    facts = [fact for fact in language_profile(conn, "russ1263").facts if fact.source == "wals"]
    assert [(fact.value, fact.entry) for fact in facts] == [
        ("yes", "Russian"),
        ("no", "Russian (Old)"),
    ]


def test_language_without_values_borrows_them_from_its_dialects(conn):
    tula = find_languages(conn, "Tula")[0]
    assert (tula.glottocode, tula.facts) == ("tula1234", 1)
    facts = language_profile(conn, "tula1234").facts
    assert [(fact.feature, fact.value, fact.entry, fact.from_dialect) for fact in facts] == [
        ("GB020", "yes", "Tula North", True)
    ]


def test_dialects_fill_only_the_sources_without_own_values(conn):
    conn.execute("INSERT INTO source_languages VALUES ('wals', 'tul', 'Tula', 'tula1234')")
    conn.execute(
        "INSERT INTO feature_values VALUES ('wals', 'tul', 'tula1234', 'wals:1A', '1', "
        "'wals:1A-1', NULL)"
    )
    facts = language_profile(conn, "tula1234").facts
    assert [(fact.source, fact.from_dialect) for fact in facts] == [
        ("wals", False),
        ("grambank", True),
    ]


def test_wals_code_finds_the_language(conn):
    assert codes(find_languages(conn, "bsq")) == ["basq1248"]


def test_entry_with_more_values_comes_first(conn):
    conn.execute(
        "INSERT INTO source_languages VALUES ('wals', 'rus2', 'Russian (Old)', 'russ1263')"
    )
    conn.execute("INSERT INTO features VALUES ('wals:2A', 'wals', '2A', 'Other', NULL, NULL)")
    conn.executemany(
        "INSERT INTO feature_values VALUES ('wals', 'rus2', 'russ1263', ?, '1', NULL, NULL)",
        [("wals:1A",), ("wals:2A",)],
    )
    facts = [fact for fact in language_profile(conn, "russ1263").facts if fact.feature == "1A"]
    assert [fact.entry_id for fact in facts] == ["rus2", "rus"]
