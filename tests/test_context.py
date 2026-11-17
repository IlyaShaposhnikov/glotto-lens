import sqlite3
from contextlib import closing

import pytest

from glotto_lens import context as context_module
from glotto_lens.context import build_context, render_context


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


def values(context):
    return {
        evidence.feature_id: [(fact.value, fact.from_dialect) for fact in evidence.facts]
        for evidence in context.evidence
    }


def test_feature_cards_bring_the_language_values(conn):
    context = build_context(conn, "feature distribution", "Russian")
    assert context.languoid.glottocode == "russ1263"
    assert values(context) == {"wals:1A": [("yes", False)], "grambank:GB020": [("yes", False)]}


def test_chapter_text_points_to_the_features_of_its_chapter(conn):
    context = build_context(conn, "Russian text")
    chapter = [hit for hit in context.hits if hit.kind == "chapter"]
    assert context_module.related_features(conn, chapter) == ["wals:1A"]


def test_missing_and_borrowed_values_are_explicit(conn):
    context = build_context(conn, "feature distribution", "Tula")
    assert values(context) == {"wals:1A": [], "grambank:GB020": [("yes", True)]}
    rendered = render_context(context)
    assert "WALS 1A Feature: no value recorded https://wals.info/feature/1A" in rendered
    assert "recorded for dialect tuln1234" in rendered


def test_language_texts_are_added_to_general_ones(conn, monkeypatch):
    monkeypatch.setattr(context_module, "TEXT_LIMIT", 1)
    assert [hit.kind for hit in build_context(conn, "text").hits] == ["feature"]
    hits = build_context(conn, "text", "Russian").hits
    assert [hit.kind for hit in hits] == ["feature", "chapter"]


def test_rendered_context_labels_and_links_evidence(conn):
    rendered = render_context(build_context(conn, "feature distribution", "Russian"))
    value = "WALS 1A Feature: yes (entry: Russian) https://wals.info/datapoint/1A/wals_code_rus"
    assert "Language: Russian (glottocode russ1263;" in rendered
    assert value in rendered
    assert "[V1] " in rendered and "[T1] " in rendered
    assert "Glottolog aes: yes (EGIDS: 6b)" in rendered


def test_unknown_language_is_reported_and_texts_still_found(conn):
    context = build_context(conn, "feature", "Klingon")
    assert context.languoid is None and context.hits
    assert "'Klingon' was not found in Glottolog" in render_context(context)


def test_other_candidates_are_listed(conn):
    rendered = render_context(build_context(conn, "feature", "Basque"))
    assert "Other languages matching 'Basque': Basque (basq9999)" in rendered
