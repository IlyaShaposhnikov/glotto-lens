"""Evaluate the question answering pipeline on a set of questions with known answers.

Each item names a question, the language it is about (an English name and, ideally, the
glottocode it must resolve to), the feature that answers it and whether the databases
record a known value ("value") or not ("no_data", which includes values coded "?").
The expected value is never written by hand: it is read from the databases, so the gold
answers are as verifiable as the data.

Metrics, per item and averaged:
- language: the pipeline resolved the item's glottocode;
- retrieved: the expected feature is in the evidence shown to the model;
- correct (value items): right language, and the answer cites a value line of the
  expected feature whose value equals the recorded main value;
- honest (no_data items): the feature was retrieved, the answer cites no value for it and
  says that the data is missing. A refusal without retrieval is a search failure, not
  honesty;
- invalid citations: the answer cites labels that are not in the evidence.
Items that contradict the data are reported and left out of every metric.

Usage:
    python -m glotto_lens.evaluate check                 # validate the items, free
    python -m glotto_lens.evaluate gaps grambank:GB020   # candidates for no_data items
    python -m glotto_lens.evaluate run --limit 5         # answer and score
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from glotto_lens import prompts
from glotto_lens.answer import Answer, answer_question
from glotto_lens.documents import RELEASES
from glotto_lens.languages import KNOWN_FACTS, find_languages, language_profile
from glotto_lens.llm import CachedModel, ChatModel, YandexModel
from glotto_lens.sources import DATA_DIR, DB_PATH

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions.jsonl"
REPORTS = DATA_DIR / "reports"
EXPECTATIONS = ("value", "no_data")
# How an answer says that the data is missing. This is part of the metric's definition,
# so every phrase is covered by a test.
MISSING_DATA = re.compile(
    r"нет (данных|значени|записанного|записей|информаци|сведени)|данных нет"
    r"|(данные|сведения|значения?) отсутству|отсутству\w* (данные|сведения|значени)"
    r"|не (содерж|указ|зафиксир|удалось найти)"
    r"|no (data|value|information)|not (recorded|known|available)",
    re.IGNORECASE,
)

# Well documented languoids without a known value of a feature: material for no_data items.
GAPS = f"""
    SELECT l.glottocode, l.name, {KNOWN_FACTS} AS facts,
           (SELECT v.value FROM feature_values v
            WHERE v.glottocode = l.glottocode AND v.feature_id = :feature LIMIT 1)
    FROM languoids l
    WHERE l.level = 'language'
      AND l.glottocode IN (SELECT glottocode FROM languoid_facts WHERE source = :source)
      AND NOT EXISTS (SELECT 1 FROM feature_values v
                      WHERE v.glottocode = l.glottocode AND v.feature_id = :feature
                        AND v.code_id IS NOT NULL)
    ORDER BY facts DESC, l.glottocode
    LIMIT :limit"""


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    language: str
    feature: str
    expect: str
    glottocode: str | None = None  # the languoid the language name must resolve to


@dataclass(frozen=True)
class Gold:
    glottocode: str | None
    value: str | None  # the main known value of the feature, None if there is none


@dataclass(frozen=True)
class Score:
    id: str
    question: str
    expect: str
    gold_glottocode: str | None
    gold_value: str | None
    glottocode: str | None = None
    language: bool = False
    retrieved: bool = False
    correct: bool | None = None  # value items only
    honest: bool | None = None  # no_data items only
    invalid_citations: int = 0
    rewrite_ok: bool = True
    complete: bool = True
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    answer: str = ""


def load_items(path: Path = QUESTIONS) -> list[Item]:
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = Item(**json.loads(line))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        if item.expect not in EXPECTATIONS:
            raise ValueError(f"{path}:{number}: expect must be one of {EXPECTATIONS}")
        items.append(item)
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: item ids must be unique")
    return items


def gold(conn: sqlite3.Connection, item: Item) -> Gold:
    """The item's languoid and its main known value of the feature."""
    if item.glottocode:
        glottocode = item.glottocode
    else:
        candidates = find_languages(conn, item.language)
        glottocode = candidates[0].glottocode if candidates else None
    profile = language_profile(conn, glottocode) if glottocode else None
    if profile is None:
        return Gold(None, None)
    facts = [
        fact
        for fact in profile.facts
        if f"{fact.source}:{fact.feature}" == item.feature and fact.known
    ]
    main = [fact for fact in facts if fact.main] or facts
    return Gold(glottocode, main[0].value if main else None)


def item_problem(conn: sqlite3.Connection, item: Item) -> str | None:
    """Why the item contradicts the data, or None when it is consistent."""
    if conn.execute("SELECT 1 FROM features WHERE id = ?", (item.feature,)).fetchone() is None:
        return f"unknown feature {item.feature}"
    expected = gold(conn, item)
    if expected.glottocode is None:
        return f"language {item.language!r} ({item.glottocode}) not found"
    if item.glottocode:
        candidates = find_languages(conn, item.language)
        resolved = candidates[0].glottocode if candidates else None
        if resolved != item.glottocode:
            return f"{item.language!r} resolves to {resolved}, not {item.glottocode}"
    if item.expect == "value" and expected.value is None:
        return "no known value, expect should be no_data"
    if item.expect == "no_data" and expected.value is not None:
        return f"value {expected.value!r} is recorded, expect should be value"
    return None


def check_items(conn: sqlite3.Connection, items: Iterable[Item]) -> list[str]:
    """Problems that make items meaningless; an empty list means all is well."""
    return [
        f"{item.id}: {problem}"
        for item in items
        if (problem := item_problem(conn, item)) is not None
    ]


def score(item: Item, expected: Gold, answer: Answer) -> Score:
    context = answer.context
    glottocode = context.languoid.glottocode if context.languoid else None
    language = glottocode == expected.glottocode
    retrieved = item.feature in [e.feature_id for e in context.evidence]
    feature_values = [c for c in answer.cited if c.feature_id == item.feature]
    correct = honest = None
    if item.expect == "value":
        correct = language and any(c.value == expected.value for c in feature_values)
    else:
        claims_value = any(c.value is not None for c in feature_values)
        admits = bool(MISSING_DATA.search(answer.text))
        honest = language and retrieved and not claims_value and admits
    input_tokens, output_tokens = answer.tokens
    return Score(
        id=item.id,
        question=item.question,
        expect=item.expect,
        gold_glottocode=expected.glottocode,
        gold_value=expected.value,
        glottocode=glottocode,
        language=language,
        retrieved=retrieved,
        correct=correct,
        honest=honest,
        invalid_citations=len(answer.invalid_labels),
        rewrite_ok=answer.rewrite.ok,
        complete=answer.complete,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        answer=answer.text,
    )


def _share(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})" if values else "-"


def summarize(scores: list[Score]) -> dict[str, str]:
    answered = [s for s in scores if s.error is None]
    return {
        "items": str(len(scores)),
        "errors": str(len(scores) - len(answered)),
        "degraded (rewrite unreadable or answer cut)": str(
            sum(not (s.rewrite_ok and s.complete) for s in answered)
        ),
        "language resolved": _share([s.language for s in scores]),
        "feature retrieved": _share([s.retrieved for s in scores]),
        "correct (language and cited value)": _share(
            [bool(s.correct) for s in scores if s.expect == "value"]
        ),
        "honest about missing data": _share(
            [bool(s.honest) for s in scores if s.expect == "no_data"]
        ),
        "answers without invalid citations": _share([s.invalid_citations == 0 for s in answered]),
        "tokens (in / out)": (
            f"{sum(s.input_tokens for s in scores)} / {sum(s.output_tokens for s in scores)}"
        ),
    }


def evaluate(conn: sqlite3.Connection, model: ChatModel, items: Iterable[Item]) -> list[Score]:
    """Score every item; a failing call is recorded as an error and the run goes on."""
    scores = []
    for item in items:
        expected = gold(conn, item)
        try:
            scores.append(score(item, expected, answer_question(conn, model, item.question)))
        except Exception as error:  # noqa: BLE001 - one bad call must not lose a paid run
            scores.append(
                Score(
                    id=item.id,
                    question=item.question,
                    expect=item.expect,
                    gold_glottocode=expected.glottocode,
                    gold_value=expected.value,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return scores


def experiment_identity(items_path: Path) -> dict:
    """What a report depends on: data releases, prompts and the question set."""
    prompt_text = prompts.REWRITE_SYSTEM + prompts.ANSWER_SYSTEM + prompts.ANSWER_USER
    return {
        "data": RELEASES,
        "prompts_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "questions_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
    }


def gaps(conn: sqlite3.Connection, feature: str, limit: int = 20) -> list[tuple]:
    """Well documented languages of the feature's source without a known value for it."""
    source = feature.split(":", 1)[0]
    params = {"feature": feature, "source": source, "limit": limit}
    return conn.execute(GAPS, params).fetchall()


def main(argv: list[str] | None = None, model: ChatModel | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on known questions.")
    parser.add_argument("command", choices=("check", "gaps", "run"))
    parser.add_argument("feature", nargs="?", help="for gaps: a feature id, e.g. wals:81A")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, help="use only the first N items")
    parser.add_argument("-m", "--model", help="model name, e.g. aliceai-llm")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    args = parser.parse_args(argv)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if args.command == "gaps":
            if not args.feature:
                parser.error("gaps needs a feature id")
            for glottocode, name, facts, value in gaps(conn, args.feature, args.limit or 20):
                state = "coded ?" if value == "?" else "no entry"
                print(f"{glottocode} | {name} | {facts} known facts | {state}")
            return
        items = load_items(args.questions)[: args.limit]
        problems = check_items(conn, items)
        if args.command == "check":
            for item in items:
                expected = gold(conn, item)
                name = conn.execute(
                    "SELECT name FROM features WHERE id = ?", (item.feature,)
                ).fetchone()
                print(f"{item.id}: {item.question}")
                print(f"    {expected.glottocode} | {item.feature} {name and name[0]}")
                print(f"    -> {expected.value!r}")
            print("\n".join(problems) or f"All {len(items)} items are consistent with the data.")
            return
        if problems:
            print("Left out, inconsistent with the data:\n" + "\n".join(problems) + "\n")
        bad = {problem.split(":", 1)[0] for problem in problems}
        model = model or CachedModel(YandexModel(args.model), use_cache=not args.no_cache)
        scores = evaluate(conn, model, [item for item in items if item.id not in bad])
    for s in scores:
        verdict = s.error or (s.correct if s.correct is not None else s.honest)
        print(f"{s.id}: language={s.language} retrieved={s.retrieved} ok={verdict}")
    print()
    summary = summarize(scores)
    for name, value in summary.items():
        print(f"{name}: {value}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.-]", "_", model.name)
    report = REPORTS / f"eval-{safe_name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = {
        "model": model.name,
        **experiment_identity(args.questions),
        "left_out": problems,
        "summary": summary,
        "items": [asdict(s) for s in scores],
    }
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
