"""Evaluate the question answering pipeline on a set of questions with known answers.

Each item names a question, the language it is about, the feature that answers it and
whether the databases record a value ("value") or not ("no_data"). The expected value
itself is never written by hand: it is read from the databases, so the gold answers are
as verifiable as the data.

Metrics, per item and averaged:
- language: the pipeline resolved the same languoid as the item's language name;
- retrieved: the expected feature is in the evidence shown to the model;
- correct (value items): the answer cites a value line of the expected feature whose
  value equals the recorded main value;
- honest (no_data items): the answer cites no value for the feature and says that the
  data is missing;
- invalid citations: the answer cites labels that are not in the evidence.

Usage:
    python -m glotto_lens.evaluate check          # validate the items, no model calls
    python -m glotto_lens.evaluate run --limit 5  # answer and score
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from glotto_lens.answer import Answer, answer_question
from glotto_lens.languages import find_languages, language_profile
from glotto_lens.llm import CachedModel, ChatModel, YandexModel
from glotto_lens.sources import DATA_DIR, DB_PATH

QUESTIONS = Path("eval/questions.jsonl")
REPORTS = DATA_DIR / "reports"
EXPECTATIONS = ("value", "no_data")
# Phrases with which an answer says that the data is missing (Russian and English).
MISSING_DATA = re.compile(
    r"нет (данных|значени|записанного|информаци|сведени)|данных нет|не (содерж|указ|зафиксир)"
    r"|no (data|value)|not recorded",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    language: str
    feature: str
    expect: str


@dataclass(frozen=True)
class Gold:
    glottocode: str | None
    value: str | None  # the main recorded value of the feature, None if there is none


@dataclass(frozen=True)
class Score:
    id: str
    expect: str
    gold_glottocode: str | None
    gold_value: str | None
    glottocode: str | None
    language: bool
    retrieved: bool
    correct: bool | None  # value items only
    honest: bool | None  # no_data items only
    invalid_citations: int
    input_tokens: int
    output_tokens: int
    answer: str


def load_items(path: Path = QUESTIONS) -> list[Item]:
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = Item(**json.loads(line))
        if item.expect not in EXPECTATIONS:
            raise ValueError(f"{path}:{number}: expect must be one of {EXPECTATIONS}")
        items.append(item)
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: item ids must be unique")
    return items


def gold(conn: sqlite3.Connection, item: Item) -> Gold:
    """The languoid of the item's language name and its main known value of the feature."""
    candidates = find_languages(conn, item.language)
    if not candidates:
        return Gold(None, None)
    glottocode = candidates[0].glottocode
    profile = language_profile(conn, glottocode)
    facts = [
        fact
        for fact in (profile.facts if profile else [])
        if f"{fact.source}:{fact.feature}" == item.feature and fact.known
    ]
    main = [fact for fact in facts if fact.main] or facts
    return Gold(glottocode, main[0].value if main else None)


def check_items(conn: sqlite3.Connection, items: Iterable[Item]) -> list[str]:
    """Problems that would make an item meaningless; an empty list means all is well."""
    problems = []
    for item in items:
        expected = gold(conn, item)
        feature = conn.execute("SELECT 1 FROM features WHERE id = ?", (item.feature,)).fetchone()
        if feature is None:
            problems.append(f"{item.id}: unknown feature {item.feature}")
        elif expected.glottocode is None:
            problems.append(f"{item.id}: language {item.language!r} not found")
        elif item.expect == "value" and expected.value is None:
            problems.append(f"{item.id}: no recorded value, expect should be no_data")
        elif item.expect == "no_data" and expected.value is not None:
            problems.append(f"{item.id}: value {expected.value!r} is recorded, expect value")
    return problems


def score(item: Item, expected: Gold, answer: Answer) -> Score:
    context = answer.context
    glottocode = context.languoid.glottocode if context.languoid else None
    feature_values = [c for c in answer.cited if c.feature_id == item.feature]
    correct = honest = None
    if item.expect == "value":
        correct = any(c.value == expected.value for c in feature_values)
    else:
        claims_value = any(c.value is not None for c in feature_values)
        honest = not claims_value and bool(MISSING_DATA.search(answer.text))
    input_tokens, output_tokens = answer.tokens
    return Score(
        id=item.id,
        expect=item.expect,
        gold_glottocode=expected.glottocode,
        gold_value=expected.value,
        glottocode=glottocode,
        language=glottocode == expected.glottocode,
        retrieved=item.feature in [e.feature_id for e in context.evidence],
        correct=correct,
        honest=honest,
        invalid_citations=len(answer.invalid_labels),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        answer=answer.text,
    )


def _share(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})" if values else "-"


def summarize(scores: list[Score]) -> dict[str, str]:
    value_items = [s.correct for s in scores if s.correct is not None]
    no_data_items = [s.honest for s in scores if s.honest is not None]
    return {
        "items": str(len(scores)),
        "language resolved": _share([s.language for s in scores]),
        "feature retrieved": _share([s.retrieved for s in scores]),
        "correct value cited": _share(value_items),
        "honest about missing data": _share(no_data_items),
        "answers without invalid citations": _share([s.invalid_citations == 0 for s in scores]),
        "tokens (in / out)": (
            f"{sum(s.input_tokens for s in scores)} / {sum(s.output_tokens for s in scores)}"
        ),
    }


def evaluate(conn: sqlite3.Connection, model: ChatModel, items: Iterable[Item]) -> list[Score]:
    scores = []
    for item in items:
        answer = answer_question(conn, model, item.question)
        scores.append(score(item, gold(conn, item), answer))
    return scores


def main(argv: list[str] | None = None, model: ChatModel | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on known questions.")
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, help="evaluate only the first N items")
    parser.add_argument("-m", "--model", help="model name, e.g. aliceai-llm")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    args = parser.parse_args(argv)
    items = load_items(args.questions)[: args.limit]
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if args.command == "check":
            problems = check_items(conn, items)
            for item in items:
                expected = gold(conn, item)
                print(f"{item.id}: {expected.glottocode} {item.feature} -> {expected.value!r}")
            print("\n".join(problems) or f"All {len(items)} items are consistent with the data.")
            return
        model = model or CachedModel(YandexModel(args.model), use_cache=not args.no_cache)
        scores = evaluate(conn, model, items)
    for s in scores:
        verdict = s.correct if s.correct is not None else s.honest
        print(f"{s.id}: language={s.language} retrieved={s.retrieved} ok={verdict}")
    print()
    summary = summarize(scores)
    for name, value in summary.items():
        print(f"{name}: {value}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"eval-{model.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = {"model": model.name, "summary": summary, "items": [asdict(s) for s in scores]}
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
