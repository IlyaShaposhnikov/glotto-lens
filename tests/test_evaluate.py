import json
import sqlite3
from contextlib import closing

import pytest

from glotto_lens import evaluate as evaluate_module
from glotto_lens.evaluate import (
    Item,
    check_items,
    evaluate,
    gold,
    load_items,
    main,
    summarize,
)
from glotto_lens.llm import Completion

RUSSIAN = Item("rus-1A", "Есть ли признак в русском?", "Russian", "wals:1A", "value")
TULA = Item("tula-1A", "Есть ли признак в туле?", "Tula", "wals:1A", "no_data")


class ScriptedModel:
    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)

    def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
        return Completion(self.texts.pop(0), self.name, input_tokens=10, output_tokens=5)


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


def test_gold_value_is_read_from_the_data(conn):
    assert gold(conn, RUSSIAN) == evaluate_module.Gold("russ1263", "yes")
    assert gold(conn, TULA) == evaluate_module.Gold("tula1234", None)


def test_check_reports_items_that_contradict_the_data(conn):
    wrong = [
        Item("a", "q", "Russian", "wals:1A", "no_data"),
        Item("b", "q", "Tula", "wals:1A", "value"),
        Item("c", "q", "Klingon", "wals:1A", "value"),
        Item("d", "q", "Russian", "wals:99Z", "value"),
    ]
    assert check_items(conn, [RUSSIAN, TULA]) == []
    problems = check_items(conn, wrong)
    assert [problem.split(":")[0] for problem in problems] == ["a", "b", "c", "d"]


def test_correct_cited_value_scores_as_correct(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    (result,) = evaluate(conn, model, [RUSSIAN])
    assert (result.language, result.retrieved, result.correct) == (True, True, True)


def test_answer_citing_a_wrong_line_is_not_correct(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [T1].")
    (result,) = evaluate(conn, model, [RUSSIAN])
    assert result.correct is False


def test_missing_data_must_be_admitted(conn):
    honest = ScriptedModel('{"language": "Tula", "query": "feature"}', "В WALS нет данных [V2].")
    silent = ScriptedModel('{"language": "Tula", "query": "feature"}', "Признак есть [T1].")
    assert evaluate(conn, honest, [TULA])[0].honest is True
    assert evaluate(conn, silent, [TULA])[0].honest is False


def test_summary_counts_each_metric_on_its_own_items(conn):
    model = ScriptedModel(
        '{"language": "Russian", "query": "feature"}',
        "Да [V1] [V9].",
        '{"language": "Tula", "query": "feature"}',
        "Нет данных.",
    )
    summary = summarize(evaluate(conn, model, [RUSSIAN, TULA]))
    assert summary["correct value cited"] == "1/1 (100%)"
    assert summary["honest about missing data"] == "1/1 (100%)"
    assert summary["answers without invalid citations"] == "1/2 (50%)"


def test_load_items_validates_the_file(tmp_path):
    path = tmp_path / "q.jsonl"
    line = json.dumps(RUSSIAN.__dict__, ensure_ascii=False)
    path.write_text(line + "\n\n", encoding="utf-8")
    assert load_items(path) == [RUSSIAN]
    path.write_text(line + "\n" + line, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_items(path)
    path.write_text(line.replace('"value"', '"maybe"'), encoding="utf-8")
    with pytest.raises(ValueError, match="expect"):
        load_items(path)


def test_repository_question_set_is_well_formed():
    items = load_items()
    assert len(items) >= 20
    assert {item.expect for item in items} == {"value", "no_data"}


def test_command_line_run_writes_a_report(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    main(["run", "--questions", str(path)], model=model)
    assert "correct value cited: 1/1 (100%)" in capsys.readouterr().out
    (report,) = (tmp_path / "reports").glob("eval-scripted-*.json")
    assert json.loads(report.read_text(encoding="utf-8"))["items"][0]["correct"] is True


def test_command_line_check_needs_no_model(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(TULA.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    main(["check", "--questions", str(path)])
    assert "All 1 items are consistent with the data." in capsys.readouterr().out
