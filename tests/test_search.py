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


def test_question_without_searchable_words_returns_nothing(conn):
    assert search(conn, "what is the?") == []
