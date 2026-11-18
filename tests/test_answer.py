import sqlite3
from contextlib import closing

import pytest

from glotto_lens import answer as answer_module
from glotto_lens.answer import Rewrite, answer_question, main, parse_rewrite, render_answer
from glotto_lens.llm import Completion


class ScriptedModel:
    """Returns prepared texts in order and records the prompts it received."""

    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
        self.prompts.append((system, user))
        return Completion(self.texts.pop(0), self.name, input_tokens=100, output_tokens=20)


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"language": "Russian", "query": "feature"}', Rewrite("Russian", "feature")),
        ('Sure!\n```json\n{"language": null, "query": "articles"}\n```', Rewrite(None, "articles")),
        ('{"language": "", "query": ""}', Rewrite(None, "question")),
        ("not json at all", Rewrite(None, "question")),
        ('{"language": 5, "query": ["x"]}', Rewrite(None, "question")),
        ("[1, 2]", Rewrite(None, "question")),
    ],
)
def test_parse_rewrite_tolerates_model_output(text, expected):
    assert parse_rewrite(text, "question") == expected


def test_pipeline_searches_the_rewritten_query_for_the_detected_language(conn):
    model = ScriptedModel(
        '{"language": "Russian", "query": "feature distribution"}',
        "В русском значение «yes» [V1], см. описание [T1].",
    )
    result = answer_question(conn, model, "Есть ли у русского признак?")
    assert result.rewrite == Rewrite("Russian", "feature distribution")
    assert result.context.languoid.glottocode == "russ1263"
    assert [source.label for source in result.cited] == ["V1", "T1"]
    assert result.invalid_labels == []
    system, user = model.prompts[1]
    assert "<question>\nЕсть ли у русского признак?\n</question>" in user
    assert "[V1]" in user and "<evidence>" in user


def test_invented_labels_are_reported(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1] [V99] [T42].")
    result = answer_question(conn, model, "q")
    assert [source.label for source in result.cited] == ["V1"]
    assert result.invalid_labels == ["V99", "T42"]
    assert "Labels not found in the evidence: V99, T42" in render_answer(result)


def test_explicit_language_overrides_detection(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Нет данных.")
    result = answer_question(conn, model, "q", language="Basque")
    assert result.context.languoid.glottocode == "basq1248"
    assert "(the answer cites no sources)" in render_answer(result)


def test_token_usage_counts_only_fresh_calls(conn):
    model = ScriptedModel('{"language": null, "query": "feature"}', "Ответ [T1].")
    result = answer_question(conn, model, "q")
    assert result.tokens == (200, 40)
    assert "Tokens: 200 in, 40 out; 0 of 2 calls from cache" in render_answer(result)


def test_command_line_runs_with_a_given_model(db_path, monkeypatch, capsys):
    monkeypatch.setattr(answer_module, "DB_PATH", db_path)
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Ответ [V1].")
    main(["Вопрос?", "--show-context"], model=model)
    output = capsys.readouterr().out
    assert "Ответ [V1]." in output and "Sources:" in output and "Values recorded" in output
