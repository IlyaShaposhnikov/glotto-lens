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
        ("Bobo Madaré (Northern)", "bobo madare northern"),
        ("Ёлки-Палки", "елки палки"),
        ("  ", ""),
    ],
)
def test_normalize_name(name, key):
    assert normalize_name(name) == key


def test_alternative_name_in_cyrillic_is_found(conn):
    assert codes(find_languages(conn, "Русский")) == ["russ1263"]


@pytest.mark.parametrize("query", ["RUS", "russ1263", "russian"])
def test_codes_and_names_are_found(conn, query):
    assert codes(find_languages(conn, query)) == ["russ1263"]


def test_language_comes_before_dialect_of_the_same_name(conn):
    found = find_languages(conn, "Básque")
    assert [(languoid.glottocode, languoid.level) for languoid in found] == [
        ("basq1248", "language"),
        ("basq9999", "dialect"),
    ]


def test_candidate_reports_family_and_number_of_facts(conn):
    russian = find_languages(conn, "Russian")[0]
    assert (russian.name, russian.facts) == ("Russian", 2)


def test_unknown_or_empty_query_finds_nothing(conn):
    assert find_languages(conn, "Klingon") == []
    assert find_languages(conn, "?!") == []


def test_profile_lists_wals_then_grambank_facts_with_readable_values(conn):
    profile = language_profile(conn, "basq1248")
    assert profile.facts == [
        Fact("wals", "1A", "Feature", "no", None),
        Fact("grambank", "GB020", "Feature", "not known (?)", None),
    ]


def test_profile_includes_glottolog_status(conn):
    assert language_profile(conn, "russ1263").status == {"aes": "yes"}


def test_profile_of_unknown_glottocode_is_none(conn):
    assert language_profile(conn, "xxxx0000") is None
