import sqlite3
from contextlib import closing

import pytest

from glotto_lens import context as context_module
from glotto_lens.context import build_context, main, related_features, render_context


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


def values(context):
    return {
        evidence.feature_id: [(fact.value, fact.from_dialect) for fact in evidence.facts]
        for evidence in context.evidence
    }


def add_chapter_text(conn, text, glottocode):
    conn.execute("INSERT INTO chapters VALUES ('2', 'Other', NULL, NULL, 'https://x/2')")
    cursor = conn.execute(
        "INSERT INTO documents (kind, title, section, text, position) "
        "VALUES ('chapter', 'Other', '1. Other', ?, 0)",
        (text,),
    )
    conn.execute("INSERT INTO document_chapters VALUES (?, '2')", (cursor.lastrowid,))
    conn.execute("INSERT INTO document_languages VALUES (?, ?)", (cursor.lastrowid, glottocode))
    conn.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")


def test_feature_documents_bring_the_language_values(conn):
    context = build_context(conn, "feature distribution", "Russian")
    assert context.languoid.glottocode == "russ1263"
    assert values(context) == {"grambank:GB020": [("yes", False)], "wals:1A": [("yes", False)]}


def test_without_feature_documents_the_best_chapter_decides(conn):
    hits = build_context(conn, "Russian").hits
    assert [hit.kind for hit in hits] == ["chapter"]
    assert related_features(conn, hits) == ["wals:1A"]


def test_language_texts_on_unrelated_chapters_are_dropped(conn, monkeypatch):
    add_chapter_text(conn, "Russian feature distribution aside.", "russ1263")
    assert "Other" in [hit.title for hit in build_context(conn, "feature distribution").hits]
    monkeypatch.setattr(context_module, "TEXT_LIMIT", 0)
    hits = build_context(conn, "feature distribution", "Russian").hits
    assert "Other" not in [hit.title for hit in hits]


def test_language_texts_on_related_chapters_are_kept(conn, monkeypatch):
    monkeypatch.setattr(context_module, "TEXT_LIMIT", 0)
    monkeypatch.setattr(context_module, "FEATURE_TEXT_LIMIT", 1)
    hits = build_context(conn, "text", "Russian").hits
    assert [hit.kind for hit in hits] == ["feature", "chapter"]


def test_missing_and_borrowed_values_are_explicit(conn):
    context = build_context(conn, "feature distribution", "Tula")
    assert values(context) == {"grambank:GB020": [("yes", True)], "wals:1A": []}
    rendered = render_context(context)
    assert "WALS 1A Feature: no value recorded https://wals.info/feature/1A" in rendered
    assert "recorded for dialect tuln1234" in rendered


def test_rendered_context_labels_links_and_attributes_evidence(conn):
    rendered = render_context(build_context(conn, "Russian feature distribution", "Russian"))
    assert "Language: Russian (glottocode russ1263;" in rendered
    assert "https://wals.info/datapoint/1A/wals_code_rus" in rendered
    assert "https://grambank.clld.org/valuesets/GB020-russ1263" in rendered
    assert "Glottolog endangerment status: yes (EGIDS: 6b)" in rendered
    assert "1. Intro (by A. Author) https://wals.info/chapter/1" in rendered
    assert "[V1] " in rendered and "[T1] " in rendered


def test_passages_are_ordered_by_score_and_not_repeated(conn):
    hits = build_context(conn, "feature distribution", "Russian").hits
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({(hit.title, hit.section, hit.text) for hit in hits}) == len(hits)


def test_unknown_language_is_reported_and_texts_still_found(conn):
    context = build_context(conn, "feature", "Klingon")
    assert context.languoid is None and context.hits
    assert "'Klingon' was not found in Glottolog" in render_context(context)


def test_other_candidates_are_listed(conn):
    rendered = render_context(build_context(conn, "feature", "Basque"))
    assert "Other languages matching 'Basque': Basque (basq9999)" in rendered


def test_question_without_searchable_words_is_rendered_completely(conn):
    rendered = render_context(build_context(conn, "what is the?", "Russian"))
    assert "No feature matched the question." in rendered
    assert "No text matched the question." in rendered


def test_command_line_prints_the_context(db_path, monkeypatch, capsys):
    monkeypatch.setattr(context_module, "DB_PATH", db_path)
    main(["feature distribution", "-l", "Russian"])
    assert "Values recorded for Russian:" in capsys.readouterr().out
