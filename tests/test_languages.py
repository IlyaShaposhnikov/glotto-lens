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
        Fact("wals", "1A", "Feature", "no", None, "Basque"),
        Fact("grambank", "GB020", "Feature", "not known (?)", None, "Basque"),
    ]
    assert not profile.from_dialects


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
    conn.execute(
        "INSERT INTO languoids (glottocode, name, level) VALUES ('lang0001', 'Lang', 'language')"
    )
    conn.execute(
        "INSERT INTO languoids (glottocode, name, level, parent_language_id) "
        "VALUES ('dial0001', 'Lang North', 'dialect', 'lang0001')"
    )
    conn.execute(
        "INSERT INTO source_languages VALUES ('grambank', 'dial0001', 'North', 'dial0001')"
    )
    conn.execute(
        "INSERT INTO feature_values VALUES ('grambank', 'dial0001', 'dial0001', "
        "'grambank:GB020', '1', 'grambank:GB020-1', NULL)"
    )
    profile = language_profile(conn, "lang0001")
    assert profile.from_dialects
    assert [(fact.feature, fact.value, fact.entry) for fact in profile.facts] == [
        ("GB020", "yes", "North")
    ]


def test_profile_includes_glottolog_status(conn):
    assert language_profile(conn, "russ1263").status == {"aes": "yes"}


def test_profile_of_unknown_glottocode_is_none(conn):
    assert language_profile(conn, "xxxx0000") is None
