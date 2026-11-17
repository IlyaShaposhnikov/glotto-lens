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
    assert (russian.facts, basque.facts) == (2, 2)


def test_name_prefix_is_used_when_nothing_matches_exactly(conn):
    assert codes(find_languages(conn, "Indo")) == ["indo1319"]


def test_unknown_or_empty_query_finds_nothing(conn):
    assert find_languages(conn, "Klingon") == []
    assert find_languages(conn, "?!") == []


def test_wals_code_finds_the_language_and_says_so(conn):
    found = find_languages(conn, "bsq")
    assert [(f.glottocode, f.matched_name, f.matched_origin) for f in found] == [
        ("basq1248", "bsq", "wals_code")
    ]


def test_candidate_reports_the_best_matching_name(conn):
    russian = find_languages(conn, "Русский")[0]
    assert (russian.matched_name, russian.matched_origin) == ("русский", "alternative")
    assert find_languages(conn, "Russian")[0].matched_origin == "glottolog"


def test_profile_lists_wals_then_grambank_facts(conn):
    assert language_profile(conn, "russ1263").facts == [
        Fact("wals", "1A", "Feature", "yes", True, None, "Russian", "rus", "russ1263", main=True),
        Fact(
            "grambank",
            "GB020",
            "Feature",
            "yes",
            True,
            None,
            "Russian",
            "russ1263",
            "russ1263",
            main=True,
        ),
    ]


def test_unknown_own_values_give_way_to_known_dialect_values(conn):
    facts = [f for f in language_profile(conn, "basq1248").facts if f.source == "grambank"]
    assert [(f.value, f.glottocode, f.from_dialect) for f in facts] == [("yes", "basq9999", True)]


def test_unknown_values_without_dialect_data_are_kept(conn):
    conn.execute("DELETE FROM languoid_facts WHERE glottocode = 'basq1248'")
    facts = [f for f in language_profile(conn, "basq1248").facts if f.source == "grambank"]
    assert [(f.value, f.known, f.from_dialect) for f in facts] == [(None, False, False)]


def test_profile_includes_glottolog_status(conn):
    assert language_profile(conn, "russ1263").status == {"aes": "yes"}


def test_profile_of_unknown_glottocode_is_none(conn):
    assert language_profile(conn, "xxxx0000") is None


def test_several_entries_of_one_language_are_all_kept(conn):
    conn.execute(
        "INSERT INTO source_languages VALUES ('wals', 'rus2', 'Russian (Old)', 'russ1263')"
    )
    conn.execute(
        "INSERT INTO feature_values VALUES ('wals', 'rus2', 'russ1263', 'wals:1A', '2', "
        "'wals:1A-2', NULL)"
    )
    facts = [fact for fact in language_profile(conn, "russ1263").facts if fact.source == "wals"]
    assert sorted((fact.value, fact.entry) for fact in facts) == [
        ("no", "Russian (Old)"),
        ("yes", "Russian"),
    ]


def test_language_without_values_borrows_them_from_its_dialects(conn):
    tula = find_languages(conn, "Tula")[0]
    assert (tula.glottocode, tula.facts) == ("tula1234", 1)
    facts = language_profile(conn, "tula1234").facts
    assert [(f.feature, f.value, f.entry, f.glottocode, f.from_dialect) for f in facts] == [
        ("GB020", "yes", "Tula North", "tuln1234", True)
    ]


def test_dialects_fill_only_the_sources_without_own_values(conn):
    conn.execute("INSERT INTO source_languages VALUES ('wals', 'tul', 'Tula', 'tula1234')")
    conn.execute(
        "INSERT INTO feature_values VALUES ('wals', 'tul', 'tula1234', 'wals:1A', '1', "
        "'wals:1A-1', NULL)"
    )
    conn.execute("INSERT INTO languoid_facts VALUES ('tula1234', 'wals', 1, 0)")
    facts = language_profile(conn, "tula1234").facts
    assert [(fact.source, fact.from_dialect) for fact in facts] == [
        ("wals", False),
        ("grambank", True),
    ]


def test_entry_with_more_known_values_is_main(conn):
    conn.execute(
        "INSERT INTO source_languages VALUES ('wals', 'rus2', 'Russian (Old)', 'russ1263')"
    )
    conn.execute("INSERT INTO features VALUES ('wals:2A', 'wals', '2A', 'Other', NULL, NULL)")
    conn.execute("INSERT INTO codes VALUES ('wals:2A-1', 'wals:2A', 'some', NULL)")
    conn.executemany(
        "INSERT INTO feature_values VALUES ('wals', 'rus2', 'russ1263', ?, '1', ?, NULL)",
        [("wals:1A", "wals:1A-2"), ("wals:2A", "wals:2A-1")],
    )
    facts = [fact for fact in language_profile(conn, "russ1263").facts if fact.feature == "1A"]
    assert [(fact.entry_id, fact.main) for fact in facts] == [("rus2", True), ("rus", False)]


def test_unknown_values_do_not_make_an_entry_main(conn):
    conn.execute(
        "INSERT INTO source_languages VALUES ('wals', 'rus2', 'Russian (Old)', 'russ1263')"
    )
    conn.execute("INSERT INTO features VALUES ('wals:2A', 'wals', '2A', 'Other', NULL, NULL)")
    conn.executemany(
        "INSERT INTO feature_values VALUES ('wals', 'rus2', 'russ1263', ?, '?', NULL, NULL)",
        [("wals:1A",), ("wals:2A",)],
    )
    facts = [fact for fact in language_profile(conn, "russ1263").facts if fact.feature == "1A"]
    assert [(fact.entry_id, fact.main) for fact in facts] == [("rus", True), ("rus2", False)]
