import sqlite3
from contextlib import closing

import pytest

from glotto_lens import demo
from glotto_lens.answer import answer_question
from glotto_lens.context import build_context
from glotto_lens.llm import Completion


class ScriptedModel:
    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)

    def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
        return Completion(self.texts.pop(0), self.name)


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


def test_evidence_becomes_rows_and_passages(conn):
    context = build_context(conn, "feature distribution", "Russian", embedder=None)
    assert "russ1263" in demo.language_line(context)
    rows = demo.value_rows(context)
    assert rows[0]["label"] == "V1" and rows[0]["url"].startswith("https://")
    items = demo.text_items(context)
    assert items and items[0]["label"] == "T1" and items[0]["text"]


def test_without_a_language_there_is_no_language_line(conn):
    context = build_context(conn, "feature distribution", embedder=None)
    assert demo.language_line(context) is None and demo.value_rows(context) == []


def test_valid_labels_become_links_and_invented_ones_stay_marked(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1], но [V99].")
    answer = answer_question(conn, model, "q", embedder=None)
    text = demo.linked_answer(answer)
    assert "[\\[V1\\]](https://" in text
    assert "\\[V99?\\]" in text
    assert any("V99" in warning for warning in demo.warnings(answer))


def test_hybrid_is_not_offered_without_vectors(db_path, tmp_path):
    assert demo.hybrid_ready(db_path) is False
    assert demo.hybrid_ready(tmp_path / "missing.sqlite") is False


def test_the_demo_models_are_in_the_comparison():
    assert "aliceai-llm" in demo.MODELS and demo.NO_REASONING <= set(demo.MODELS)
