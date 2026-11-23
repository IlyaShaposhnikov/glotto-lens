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
    assert "\\[[V1](https://" in text
    assert "\\[V99?\\]" in text
    assert any("V99" in warning for warning in demo.warnings(answer))


def test_hybrid_is_not_offered_without_vectors(db_path, tmp_path):
    assert demo.hybrid_ready(db_path) is False
    assert demo.hybrid_ready(tmp_path / "missing.sqlite") is False


def test_the_demo_models_are_in_the_comparison():
    assert "aliceai-llm" in demo.MODELS and demo.NO_REASONING <= set(demo.MODELS)


def test_every_label_the_citation_check_accepts_is_linked(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1, T1], [v 1].")
    answer = answer_question(conn, model, "q", embedder=None)
    text = demo.linked_answer(answer)
    assert "[V1](https://" in text and "[T1](https://" in text and "[v 1](https://" in text


def test_links_and_images_written_by_the_model_stay_text(conn):
    reply = "Да [V1]. [x](http://evil.example) ![](http://evil.example/p.png) www.evil.example"
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', reply)
    text = demo.linked_answer(answer_question(conn, model, "q", embedder=None))
    assert "](http://evil" not in text and "http://evil" not in text
    assert "www.evil" not in text and "\\[x\\]" in text


def test_an_answer_with_only_passages_uncited_is_flagged(conn):
    model = ScriptedModel('{"language": null, "query": "feature distribution"}', "Нет ссылок.")
    answer = answer_question(conn, model, "q", embedder=None)
    assert answer.context.languoid is None and answer.context.hits
    assert "The answer cites no evidence." in demo.warnings(answer)


@pytest.mark.parametrize("folder", ["plain", "with space", "odd#name%20"])
def test_hybrid_check_opens_absolute_paths(db_path, tmp_path, folder):
    import shutil

    target = tmp_path / folder / "glottolens.sqlite"
    target.parent.mkdir()
    shutil.copy(db_path, target)
    with closing(sqlite3.connect(target)) as connection:  # the database itself opens
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] > 0
    assert demo.hybrid_ready(target.resolve()) is False  # no vectors, but no error either
    from glotto_lens.sources import read_only

    with closing(read_only(target.resolve())) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] > 0


def test_logging_can_be_turned_off(monkeypatch):
    monkeypatch.delenv("GLOTTO_LENS_LOG", raising=False)
    assert demo.log_enabled()
    monkeypatch.setenv("GLOTTO_LENS_LOG", "0")
    assert not demo.log_enabled()
