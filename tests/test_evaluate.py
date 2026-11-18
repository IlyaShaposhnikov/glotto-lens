import json
import sqlite3
from contextlib import closing

import pytest

from glotto_lens import evaluate as evaluate_module
from glotto_lens.documents import RELEASES
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

RUSSIAN = Item("rus-1A", "Есть ли признак в русском?", "Russian", "wals:1A", "value", "russ1263")
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
        Item("e", "q", "Basque", "wals:1A", "value", "basq9999"),
    ]
    assert check_items(conn, [RUSSIAN, TULA]) == []
    problems = check_items(conn, wrong)
    assert [problem.split(":")[0] for problem in problems] == ["a", "b", "c", "d", "e"]
    assert "resolves to basq1248, not basq9999" in problems[-1]


def test_correct_cited_value_scores_as_correct(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    (result,) = evaluate(conn, model, [RUSSIAN])
    assert (result.language, result.retrieved, result.correct) == (True, True, True)


def test_right_value_for_the_wrong_language_is_not_correct(conn):
    item = Item("x", "q", "Russian", "wals:1A", "value", "russ1263")
    model = ScriptedModel('{"language": "Basque", "query": "feature"}', "Нет [V1].")
    (result,) = evaluate(conn, model, [item])
    assert (result.language, result.correct) == (False, False)


def test_answer_citing_a_wrong_line_is_not_correct(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [T1].")
    (result,) = evaluate(conn, model, [RUSSIAN])
    assert result.correct is False


def test_refusal_without_retrieval_is_not_honesty(conn):
    model = ScriptedModel('{"language": "Tula", "query": "zzz"}', "Нет данных.")
    (result,) = evaluate(conn, model, [TULA])
    assert (result.retrieved, result.honest) == (False, False)


@pytest.mark.parametrize(
    "text",
    [
        "Нет данных.",
        "В базе нет значения.",
        "Данные отсутствуют.",
        "Нет записей о признаке.",
        "Значение не указано.",
        "Grambank не содержит данных о немецком.",
        "Не зафиксировано значений.",
        "Не удалось найти сведения.",
        "No value is recorded.",
        "Значение не известно.",
        "Неизвестно, есть ли различие.",
    ],
)
def test_missing_data_phrases(text):
    assert evaluate_module.MISSING_DATA.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "Артикль отсутствует.",
        "Падежей нет.",
        "Русский не содержит артиклей.",
        "В турецком не выражен род.",
        "Род не указывается.",
    ],
)
def test_claims_of_absence_are_not_missing_data(text):
    assert not evaluate_module.MISSING_DATA.search(text)


def test_failing_call_is_recorded_and_the_run_goes_on(conn):
    class Broken(ScriptedModel):
        def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
            if not self.texts:
                raise ConnectionError("network down")
            return super().complete(system, user)

    model = Broken('{"language": "Russian", "query": "feature"}', "Да [V1].")
    first, second = evaluate(conn, model, [RUSSIAN, TULA])
    assert first.correct and second.error == "ConnectionError: network down"
    summary = summarize([first, second])
    assert summary["errors (left out of the metrics below)"] == "1"
    assert summary["honest about missing data"] == "- (no items to measure)"
    assert summary["language resolved"] == "1/1 (100%)"


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
    assert summary["correct (language and cited value)"] == "1/1 (100%)"
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
    path.write_text(line.replace('"expect"', '"expected"'), encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_items(path)


def test_repository_question_set_is_well_formed():
    items = load_items()
    assert len(items) >= 20
    assert {item.expect for item in items} == {"value", "no_data"}
    assert all(item.glottocode for item in items)


def test_only_rejects_unknown_ids(db_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    with pytest.raises(SystemExit):
        main(["check", "--only", "no-such-item"])


def test_report_is_written_even_when_the_run_is_interrupted(db_path, tmp_path, monkeypatch):
    path = tmp_path / "q.jsonl"
    lines = [json.dumps(i.__dict__, ensure_ascii=False) for i in (RUSSIAN, TULA)]
    path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")

    class Interrupted(ScriptedModel):
        def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
            if not self.texts:
                raise KeyboardInterrupt
            return super().complete(system, user)

    model = Interrupted('{"language": "Russian", "query": "feature"}', "Да [V1].")
    with pytest.raises(KeyboardInterrupt):
        main(["run", "--questions", str(path)], model=model)
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["finished"] is False and [item["id"] for item in data["items"]] == ["rus-1A"]


def test_uncoded_text_is_not_a_claimed_value(conn):
    conn.execute("INSERT INTO source_languages VALUES ('wals', 'tul', 'Tula', 'tula1234')")
    conn.execute(
        "INSERT INTO feature_values VALUES ('wals', 'tul', 'tula1234', 'wals:1A', "
        "'unclear', NULL, NULL)"
    )
    model = ScriptedModel('{"language": "Tula", "query": "feature"}', "Значение не указано [V1].")
    (result,) = evaluate(conn, model, [TULA])
    assert result.honest is True


def test_command_line_run_writes_a_report(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    main(["run", "--questions", str(path)], model=model)
    assert "correct (language and cited value): 1/1 (100%)" in capsys.readouterr().out
    (report,) = (tmp_path / "reports").glob("eval-scripted-*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["items"][0]["correct"] is True
    assert data["data"] == RELEASES and data["prompts_sha256"]
    assert data["retrieval"]["max_features"] == 8


def test_run_leaves_out_items_that_contradict_the_data(db_path, tmp_path, monkeypatch, capsys):
    broken = Item("broken", "q", "Tula", "wals:1A", "value")
    path = tmp_path / "q.jsonl"
    lines = [json.dumps(i.__dict__, ensure_ascii=False) for i in (broken, RUSSIAN)]
    path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    main(["run", "--questions", str(path)], model=model)
    output = capsys.readouterr().out
    assert "Left out, inconsistent with the data:\nbroken:" in output
    assert "items: 1" in output


def test_command_line_check_needs_no_model(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(TULA.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    main(["check", "--questions", str(path)])
    assert "All 1 items are consistent with the data." in capsys.readouterr().out


def test_gaps_skip_languages_whose_dialects_have_a_value(conn):
    codes = [row[0] for row in evaluate_module.gaps(conn, "grambank:GB020")]
    assert "tula1234" not in codes  # Tula borrows GB020 from its dialect


def test_only_selects_items_by_id(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    lines = [json.dumps(i.__dict__, ensure_ascii=False) for i in (RUSSIAN, TULA)]
    path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Tula", "query": "feature"}', "Нет данных [V1].")
    main(["run", "--questions", str(path), "--only", "tula-1A"], model=model)
    output = capsys.readouterr().out
    assert "items: 1" in output and "rus-1A" not in output


def test_retrieval_only_mode_makes_no_answer_call(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Russian", "query": "feature"}')  # one call only
    main(["run", "--retrieval-only", "--questions", str(path)], model=model)
    output = capsys.readouterr().out
    assert "feature retrieved: 1/1 (100%)" in output
    assert "correct (language and cited value): - (no items to measure)" in output
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["mode"] == "retrieval only" and data["metrics_sha256"]
    assert data["items"][0]["features"]


def test_failed_retrieval_is_printed_with_query_and_features(
    db_path, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Russian", "query": "zzz"}')
    main(["run", "--retrieval-only", "--questions", str(path)], model=model)
    assert "retrieved=False ok=None query='zzz' features=[]" in capsys.readouterr().out


def test_retrieval_limits_can_be_overridden(db_path, tmp_path, monkeypatch):
    from glotto_lens import context

    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    # main() assigns the constants directly; setting them to their own values here makes
    # monkeypatch restore them after the test.
    for name in ("FEATURE_POOL", "FEATURE_TEXT_LIMIT", "MAX_FEATURES"):
        monkeypatch.setattr(context, name, getattr(context, name))
    model = ScriptedModel('{"language": "Russian", "query": "feature"}')
    args = ["run", "--retrieval-only", "--feature-pool", "30", "--feature-text-limit", "6"]
    main([*args, "--max-features", "1", "--questions", str(path)], model=model)
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    retrieval = data["retrieval"]
    assert (retrieval["feature_pool"], retrieval["feature_text_limit"]) == (30, 6)
    assert retrieval["max_features"] == 1
    assert len(data["items"][0]["features"]) == 1


@pytest.mark.parametrize("flag", ["--feature-pool", "--feature-text-limit", "--max-features"])
def test_negative_limits_are_rejected(flag):
    with pytest.raises(SystemExit):
        main(["run", flag, "-1"])


def test_retrieval_only_summary_does_not_score_citations(db_path, tmp_path, monkeypatch):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel('{"language": "Russian", "query": "feature"}')
    main(["run", "--retrieval-only", "--questions", str(path)], model=model)
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
    assert summary["answers without invalid citations"] == "- (no items to measure)"


def test_experiment_identity_changes_with_the_prompts(tmp_path, monkeypatch):
    from glotto_lens import prompts

    path = tmp_path / "q.jsonl"
    path.write_text("{}", encoding="utf-8")
    before = evaluate_module.experiment_identity(path)
    monkeypatch.setattr(prompts, "REWRITE_SYSTEM", prompts.REWRITE_SYSTEM + " changed")
    assert evaluate_module.experiment_identity(path)["prompts_sha256"] != before["prompts_sha256"]


def test_experiment_identity_changes_with_the_metric(tmp_path, monkeypatch):
    import re

    path = tmp_path / "q.jsonl"
    path.write_text("{}", encoding="utf-8")
    before = evaluate_module.experiment_identity(path)
    monkeypatch.setattr(evaluate_module, "MISSING_DATA", re.compile("no data"))
    assert evaluate_module.experiment_identity(path)["metrics_sha256"] != before["metrics_sha256"]
