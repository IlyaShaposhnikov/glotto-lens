import json
import sqlite3
from contextlib import closing

import pytest

from glotto_lens import evaluate as evaluate_module
from glotto_lens.documents import RELEASES
from glotto_lens.evaluate import (
    Gold,
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
        "Значение по соответствующему признаку в WALS не зафиксировано.",
        "Значение для признака «Есть ли артикли» (GB020) для немецкого языка не записано.",
        "WALS does not record the number of cases for Hindi.",
        "Значение для признака «Are there definite or specific articles?» не записано.",
        "В Grambank значение по признаку «есть ли различие» отмечено как «не известно» [V1].",
    ],
)
def test_missing_data_phrases(text):
    assert evaluate_module.admits_missing_data(text)


@pytest.mark.parametrize(
    "text",
    [
        "Артикль отсутствует.",
        "Падежей нет.",
        "Русский не содержит артиклей.",
        "В турецком не выражен род.",
        "Род не указывается.",
        "Порядок слов SVO, но падеж не указывается предлогом.",
        "Значение порядка слов — SVO. Род не указывается.",
    ],
)
def test_claims_of_absence_are_not_missing_data(text):
    assert not evaluate_module.admits_missing_data(text)


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
    for name in ("FEATURE_TEXT_LIMIT", "MAX_FEATURES"):
        monkeypatch.setattr(context, name, getattr(context, name))
    model = ScriptedModel('{"language": "Russian", "query": "feature"}')
    args = ["run", "--retrieval-only", "--feature-text-limit", "6"]
    main([*args, "--max-features", "1", "--questions", str(path)], model=model)
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    retrieval = data["retrieval"]
    assert (retrieval["features_searched"], retrieval["feature_text_limit"]) == (6, 6)
    assert retrieval["max_features"] == 1
    assert len(data["items"][0]["features"]) == 1


@pytest.mark.parametrize("flag", ["--limit", "--feature-text-limit", "--max-features"])
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


def test_report_records_the_pool_really_searched(monkeypatch):
    from glotto_lens import context

    monkeypatch.setattr(context, "MAX_FEATURES", 2)
    monkeypatch.setattr(context, "FEATURE_TEXT_LIMIT", 5)
    assert evaluate_module.retrieval_settings()["features_searched"] == 5


def test_metric_fingerprint_covers_the_quote_patterns(monkeypatch):
    import re

    before = evaluate_module.metric_fingerprint()
    monkeypatch.setattr(evaluate_module, "QUOTED", re.compile("«[^»]*»"))
    assert evaluate_module.metric_fingerprint() != before


def test_metric_fingerprint_survives_missing_source(monkeypatch):
    def unavailable(_):
        raise OSError("no source")

    monkeypatch.setattr(evaluate_module.inspect, "getsource", unavailable)
    assert len(evaluate_module.metric_fingerprint()) == 64


def test_code_version_is_none_without_git(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(evaluate_module.subprocess, "run", missing)
    assert evaluate_module.code_version() is None


def test_code_version_marks_uncommitted_changes(monkeypatch):
    import subprocess

    outputs = iter(["abc1234\n", " M src/glotto_lens/context.py\n"])

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(evaluate_module.subprocess, "run", fake_run)
    assert evaluate_module.code_version() == "abc1234+dirty"


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ('{"verdict": true, "reason": "states SVO"}', True),
        ('Sure: {"verdict": false, "reason": "no value"}', False),
        ('{"verdict": "yes"}', None),
        ('{"verdict": "True"}', True),
        ('{"verdict": 1}', None),
        ("no json", None),
    ],
)
def test_judge_verdict_is_read_strictly(text, verdict):
    assert evaluate_module.parse_verdict(text)[0] is verdict


def test_judge_sees_the_recorded_value_and_scores_content(conn):
    calls = []

    class Recording(ScriptedModel):
        def complete(self, system, user, **kwargs):
            calls.append(user)
            return super().complete(system, user, **kwargs)

    model = Recording(
        '{"language": "Russian", "query": "feature"}',
        "Да, есть.",  # right content, no citation
        '{"verdict": true, "reason": "says yes"}',
    )
    (result,) = evaluate_module.evaluate_items(conn, model, [RUSSIAN], use_judge=True)
    assert result.correct is False and result.judged is True
    assert "<expected>\nFeature wals:1A" in calls[-1] and "yes" in calls[-1]
    assert (result.input_tokens, result.judge_input_tokens) == (20, 10)  # kept apart


def test_judge_cannot_approve_the_wrong_language(conn):
    model = ScriptedModel(
        '{"language": "Basque", "query": "feature"}', "Да [V1].", '{"verdict": true}'
    )
    (result,) = evaluate_module.evaluate_items(conn, model, [RUSSIAN], use_judge=True)
    assert result.language is False and result.judged is False


def test_failing_judge_keeps_the_answer(conn):
    class BrokenJudge(ScriptedModel):
        def complete(self, system, user, **kwargs):
            if "<expected>" in user:
                raise TimeoutError("judge")
            return super().complete(system, user, **kwargs)

    model = BrokenJudge('{"language": "Russian", "query": "feature"}', "Да [V1].")
    (result,) = evaluate_module.evaluate_items(conn, model, [RUSSIAN], use_judge=True)
    assert result.error is None and result.correct is True
    assert result.judged is None and "TimeoutError" in result.judge_reason


def test_summary_compares_judge_and_phrase_check(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    model = ScriptedModel(
        '{"language": "Russian", "query": "feature"}', "Да [V1].", '{"verdict": true}'
    )
    main(["run", "--judge", "--questions", str(path)], model=model)
    out = capsys.readouterr().out
    assert "judge: content correct: 1/1 (100%)" in out
    assert "judge no, citation check yes: 0" in out and "judge tokens (in / out): 10 / 5" in out
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    assert json.loads(report.read_text(encoding="utf-8"))["judge_sha256"]


def test_judge_needs_answers():
    with pytest.raises(SystemExit):
        main(["run", "--judge", "--retrieval-only"])


def test_judge_can_be_another_model(conn):
    answers = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    judge_model = ScriptedModel('{"verdict": true, "reason": "ok"}')
    (result,) = evaluate_module.evaluate(
        conn, answers, [RUSSIAN], use_judge=True, judge_model=judge_model
    )
    assert result.judged is True and not answers.texts and not judge_model.texts


def test_cut_off_judge_reply_is_reported():
    s = evaluate_module.Score("a", "q", "value", "russ1263", "x", language=True)
    call = Completion('{"verdict": tr', "j", complete=False, stop_reason="max_tokens")
    judged = evaluate_module.apply_judge(s, call)
    assert judged.judged is None and "cut off (max_tokens)" in judged.judge_reason


def test_judge_cannot_approve_a_gap_admitted_for_another_language():
    s = evaluate_module.Score("a", "q", "no_data", "stan1295", None, language=False)
    judged = evaluate_module.apply_judge(s, Completion('{"verdict": true}', "j"))
    assert judged.judged is False


def test_judge_is_told_which_kind_of_missing_value(conn, monkeypatch):
    from types import SimpleNamespace

    def fact(value, known):
        return SimpleNamespace(source="wals", feature="1A", value=value, known=known)

    item = Item("n", "q", "Russian", "wals:1A", "no_data", "russ1263")
    missing = Gold("russ1263", None)
    for facts, expected in [
        ([], "no value is recorded"),
        ([fact(None, False)], "coded as not known (?)"),
        ([fact("free text", False)], "recorded only as text: free text"),
    ]:
        profile = SimpleNamespace(facts=facts)
        monkeypatch.setattr(evaluate_module, "language_profile", lambda c, g, p=profile: p)
        assert expected in evaluate_module.expected_text(conn, item, missing)
    monkeypatch.undo()
    assert evaluate_module.expected_text(conn, RUSSIAN, gold(conn, RUSSIAN)).endswith(": yes")


def test_judge_model_is_recorded(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("{}", encoding="utf-8")
    identity = evaluate_module.experiment_identity(path, True, "deepseek-v32")
    assert identity["judge_model"] == "deepseek-v32"
    assert "judge_model" not in evaluate_module.experiment_identity(path)


def test_judge_model_needs_judge():
    with pytest.raises(SystemExit):
        main(["run", "--judge-model", "deepseek-v32"])


def test_judge_is_told_when_the_value_comes_from_a_dialect(conn, monkeypatch):
    from types import SimpleNamespace

    borrowed = SimpleNamespace(
        source="wals",
        feature="1A",
        value="yes",
        known=True,
        from_dialect=True,
        main=False,
        glottocode="tuln1234",
    )
    monkeypatch.setattr(
        evaluate_module, "language_profile", lambda c, g: SimpleNamespace(facts=[borrowed])
    )
    item = Item("t", "q", "Tula", "wals:1A", "value", "tula1234")
    text = evaluate_module.expected_text(conn, item, Gold("tula1234", "yes"))
    assert text.endswith("yes (recorded for its dialect tuln1234)")


def test_command_line_takes_a_judge_model(db_path, tmp_path, monkeypatch, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(RUSSIAN.__dict__, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "DB_PATH", db_path)
    monkeypatch.setattr(evaluate_module, "REPORTS", tmp_path / "reports")
    answers = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    judge_model = ScriptedModel('{"verdict": false, "reason": "strict"}')
    judge_model.name = "other"
    main(["run", "--judge", "--questions", str(path)], model=answers, judge_model=judge_model)
    assert "judge no, citation check yes: 1" in capsys.readouterr().out
    (report,) = (tmp_path / "reports").glob("eval-*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["judge_model"] == "other" and data["judge_max_tokens"] == 2000


def test_rewrite_model_gives_every_answer_model_the_same_evidence(conn):
    rewriter = ScriptedModel('{"language": "Russian", "query": "feature"}')
    answers = ScriptedModel("Да [V1].")  # only the answer: the rewrite is not asked of it
    (result,) = evaluate_module.evaluate(conn, answers, [RUSSIAN], rewrite_model=rewriter)
    assert result.correct is True and not rewriter.texts and not answers.texts
