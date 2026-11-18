import sqlite3
from contextlib import closing

import pytest

from glotto_lens.database import SCHEMA
from glotto_lens.documents import load_documents
from glotto_lens.search import search, to_match_query


@pytest.fixture
def conn():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO chapters VALUES (?, ?, NULL, NULL, ?)",
            [
                ("81", "Order of Subject, Object and Verb", "https://wals.info/chapter/81"),
                ("132", "Number of Basic Colour Categories", "https://wals.info/chapter/132"),
                ("133", "Number of Non-Derived Colour Categories", "https://wals.info/chapter/133"),
            ],
        )
        connection.executemany(
            "INSERT INTO blocks (chapter_id, position, kind, section, text) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    "81",
                    0,
                    "paragraph",
                    "1. Types",
                    "In verb-initial languages the verb comes first.",
                ),
                ("132", 0, "paragraph", "1. Colours", "Languages differ in basic colour terms."),
                ("133", 0, "paragraph", "1. Colours", "Languages differ in basic colour terms."),
            ],
        )
        connection.execute("INSERT INTO block_languages VALUES (1, 'iris1253')")
        load_documents(connection)
        yield connection


def test_match_query_quotes_words_and_drops_stopwords():
    assert to_match_query('Which languages put the "verb" first?') == (
        '"languages" OR "put" OR "verb" OR "first"'
    )


def test_search_ranks_the_relevant_document_first(conn):
    hits = search(conn, "where does the verb come first?")
    assert hits[0].title == "Order of Subject, Object and Verb"
    assert hits[0].urls == ("https://wals.info/chapter/81",)


def test_search_matches_word_forms(conn):
    assert search(conn, "colours")[0].section == "1. Colours"
    assert search(conn, "colour")[0].section == "1. Colours"


def test_identical_text_is_stored_once_with_all_its_chapters(conn):
    hits = search(conn, "basic colour terms")
    assert len(hits) == 1
    assert hits[0].urls == ("https://wals.info/chapter/132", "https://wals.info/chapter/133")


def test_document_inherits_languages_of_its_blocks(conn):
    assert conn.execute("SELECT * FROM document_languages").fetchall() == [(1, "iris1253")]


def test_feature_card_is_found_with_its_own_url(conn):
    conn.execute(
        "INSERT INTO documents (kind, title, section, text, position, url) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("feature", "Tone", "Summary", "Tone systems.", 0, "https://wals.info/feature/13A"),
    )
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
    hit = search(conn, "tone")[0]
    assert (hit.kind, hit.urls) == ("feature", ("https://wals.info/feature/13A",))


def test_question_without_searchable_words_returns_nothing(conn):
    assert search(conn, "what is the?") == []


def test_equal_scores_are_ordered_by_document_id(conn):
    conn.executemany(
        "INSERT INTO documents (kind, title, section, text, position) VALUES (?, ?, ?, ?, ?)",
        [("feature", "Same", "Summary", "Identical tone text.", 0)] * 3,
    )
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
    ids = [hit.document_id for hit in search(conn, "identical tone")]
    assert ids == sorted(ids)


def test_snippet_highlights_matched_words(conn):
    assert "[verb]" in search(conn, "verb")[0].snippet


def test_language_filter_keeps_only_texts_mentioning_it(conn):
    assert [hit.title for hit in search(conn, "verb first", glottocode="iris1253")] == [
        "Order of Subject, Object and Verb"
    ]
    assert search(conn, "colour", glottocode="iris1253") == []


def test_kind_filter_keeps_only_documents_of_that_kind(conn):
    conn.execute(
        "INSERT INTO documents (kind, title, section, text, position) VALUES (?, ?, ?, ?, ?)",
        ("feature", "Verb card", "Summary", "Verb position card.", 0),
    )
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
    assert [hit.kind for hit in search(conn, "verb", kind="feature")] == ["feature"]


def test_per_feature_search_keeps_the_best_document_of_each_feature(conn):
    rows = [
        ("grambank:GB1", "Summary", "verb verb verb"),
        ("grambank:GB1", "Procedure", "verb verb"),
        ("grambank:GB1", "Examples", "verb"),
        ("wals:1A", "Summary", "verb word"),
    ]
    conn.executemany(
        "INSERT INTO documents (kind, title, section, text, position, feature_id)"
        " VALUES ('feature', 'Card', ?, ?, 0, ?)",
        [(section, text, feature) for feature, section, text in rows],
    )
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
    by_document = search(conn, "verb", limit=3, kind="feature")
    assert [hit.feature_id for hit in by_document] == ["grambank:GB1"] * 3
    by_feature = search(conn, "verb", limit=3, per_feature=True)
    assert [(hit.feature_id, hit.section) for hit in by_feature] == [
        ("grambank:GB1", "Summary"),
        ("wals:1A", "Summary"),
    ]
