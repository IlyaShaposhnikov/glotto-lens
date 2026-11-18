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
- invalid citations: the answer cites labels that are not in the evidence (full runs only).
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
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from glotto_lens import context as context_settings
from glotto_lens import prompts
from glotto_lens.answer import Answer, answer_question, retrieve
from glotto_lens.documents import RELEASES
from glotto_lens.languages import KNOWN_FACTS, find_languages, language_profile
from glotto_lens.llm import CachedModel, ChatModel, YandexModel
from glotto_lens.sources import DATA_DIR, DB_PATH

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions.jsonl"
REPORTS = DATA_DIR / "reports"
EXPECTATIONS = ("value", "no_data")
# How an answer says that the data is missing. This is part of the metric's definition,
# so every phrase is covered by a test. Only Russian and English answers are recognised:
# questions in other languages would score as dishonest, so the set keeps to these two.
# For a valid no_data item no cited value of the feature can be coded, so "honest" comes
# down to: right language, feature retrieved, and one of these phrases in the answer.
# Known limit: "не известно" also matches a claim about the language ("род неизвестен"),
# and a phrase about another feature in the same sentence counts as well.
MISSING_DATA = re.compile(
    r"нет (данных|значени|записанного|записей|информаци|сведени)|данных нет"
    r"|(данные|сведения|значения?) отсутству|отсутству\w* (данные|сведения|значени)"
    r"|не (содерж|указ|зафиксир)\w* (данн|значени|информаци|сведени|запис)"
    # a value or the data, then up to 12 words naming the feature within the same sentence,
    # then the negation: "значение по соответствующему признаку в WALS не зафиксировано"
    r"|(данн|значени|сведени)\w*(\s+[^\s.!?;]+){0,12}?\s+не (указ|зафиксир|приведен|записан)"
    r"|не удалось найти|не ?известн"
    r"|no (data|value|information)|not (record|known|available)",
    re.IGNORECASE,
)
# Quoted text is a name, not a sentence: "значение для признака «Are there articles?» не
# записано" is one statement. Quotes are emptied before matching, so a question mark in a
# feature name does not end the sentence.
QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”")


def admits_missing_data(text: str) -> bool:
    """The answer says that the data is missing (see MISSING_DATA)."""
    return bool(MISSING_DATA.search(QUOTED.sub("«»", text)))


# Well documented languoids of a source without own coded value of a feature. These are only
# candidates: gaps() keeps those whose gold value (dialects included) is really missing.
GAP_CANDIDATES = f"""
    SELECT l.glottocode, l.name, {KNOWN_FACTS} AS facts,
           (SELECT GROUP_CONCAT(DISTINCT v.value) FROM feature_values v
            WHERE v.glottocode = l.glottocode AND v.feature_id = :feature)
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
    invalid_citations: int | None = None  # None when no answer was generated
    rewrite_ok: bool = True
    complete: bool = True
    error: str | None = None
    query: str = ""
    features: tuple[str, ...] = ()  # the features shown to the model, in order
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
        claims_value = any(c.known for c in feature_values)
        admits = admits_missing_data(answer.text)
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
        query=answer.rewrite.query,
        features=tuple(e.feature_id for e in context.evidence),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        answer=answer.text,
    )


def score_retrieval(item: Item, expected: Gold, rewrite, context, call) -> Score:
    """Score only language and retrieval: no answer is generated."""
    glottocode = context.languoid.glottocode if context.languoid else None
    features = tuple(e.feature_id for e in context.evidence)
    return Score(
        id=item.id,
        question=item.question,
        expect=item.expect,
        gold_glottocode=expected.glottocode,
        gold_value=expected.value,
        glottocode=glottocode,
        language=glottocode == expected.glottocode,
        retrieved=item.feature in features,
        rewrite_ok=rewrite.ok,
        complete=call.complete,
        query=rewrite.query,
        features=features,
        input_tokens=0 if call.cached else call.input_tokens,
        output_tokens=0 if call.cached else call.output_tokens,
    )


def _share(values: list[bool]) -> str:
    if not values:
        return "- (no items to measure)"
    return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})"


def summarize(scores: list[Score]) -> dict[str, str]:
    """Quality metrics over answered items only; failed calls are counted separately."""
    answered = [s for s in scores if s.error is None]
    return {
        "items": str(len(scores)),
        "errors (left out of the metrics below)": str(len(scores) - len(answered)),
        "degraded (a model reply unreadable or cut off)": str(
            sum(not (s.rewrite_ok and s.complete) for s in answered)
        ),
        "language resolved": _share([s.language for s in answered]),
        "feature retrieved": _share([s.retrieved for s in answered]),
        "correct (language and cited value)": _share(
            [s.correct for s in answered if s.correct is not None]
        ),
        "honest about missing data": _share([s.honest for s in answered if s.honest is not None]),
        "answers without invalid citations": _share(
            [s.invalid_citations == 0 for s in answered if s.invalid_citations is not None]
        ),
        "tokens (in / out)": (
            f"{sum(s.input_tokens for s in scores)} / {sum(s.output_tokens for s in scores)}"
        ),
    }


def evaluate_items(
    conn: sqlite3.Connection,
    model: ChatModel,
    items: Iterable[Item],
    retrieval_only: bool = False,
) -> Iterator[Score]:
    """Score items one by one; a failing call is recorded as an error and the run goes on.

    With retrieval_only, only the rewrite is requested (usually from the cache), so
    retrieval settings can be compared for free.
    """
    for item in items:
        expected = Gold(item.glottocode, None)
        try:
            expected = gold(conn, item)
            if retrieval_only:
                yield score_retrieval(item, expected, *retrieve(conn, model, item.question))
            else:
                yield score(item, expected, answer_question(conn, model, item.question))
        except Exception as error:  # noqa: BLE001 - one bad call must not lose a paid run
            yield Score(
                id=item.id,
                question=item.question,
                expect=item.expect,
                gold_glottocode=expected.glottocode,
                gold_value=expected.value,
                error=f"{type(error).__name__}: {error}",
            )


def evaluate(conn: sqlite3.Connection, model: ChatModel, items: Iterable[Item]) -> list[Score]:
    return list(evaluate_items(conn, model, items))


def write_report(path: Path, data: dict) -> None:
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def experiment_identity(items_path: Path) -> dict:
    """What a report depends on: data releases, prompts and the question set."""
    prompt_text = "".join(
        (prompts.REWRITE_SYSTEM, prompts.REWRITE_USER, prompts.ANSWER_SYSTEM, prompts.ANSWER_USER)
    )
    return {
        "data": RELEASES,
        "prompts_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "questions_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(
            (MISSING_DATA.pattern + QUOTED.pattern).encode("utf-8")
        ).hexdigest(),
        "retrieval": retrieval_settings(),
    }


def gaps(conn: sqlite3.Connection, feature: str, limit: int = 20) -> list[tuple]:
    """Well documented languages without a known value of the feature, by the same rule
    as gold(): (glottocode, name, known facts, recorded values or None)."""
    source = feature.split(":", 1)[0]
    params = {"feature": feature, "source": source, "limit": limit * 3}
    found = []
    for glottocode, name, facts, values in conn.execute(GAP_CANDIDATES, params).fetchall():
        item = Item("gap", "", name, feature, "no_data", glottocode)
        if gold(conn, item).value is None:
            found.append((glottocode, name, facts, values))
    return found[:limit]


def retrieval_settings() -> dict:
    """Retrieval constants that change results, recorded with every report."""
    from glotto_lens import context, search

    return {
        "text_limit": context.TEXT_LIMIT,
        # features (one document each) really searched; older reports had "feature_pool",
        # counted in documents, so the two are not comparable
        "features_searched": context.pool_size(),
        "feature_text_limit": context.FEATURE_TEXT_LIMIT,
        "language_text_limit": context.LANGUAGE_TEXT_LIMIT,
        "max_features": context.MAX_FEATURES,
        "search_sql_sha256": hashlib.sha256(
            (search.SEARCH + search.PER_FEATURE).encode("utf-8")
        ).hexdigest(),
        "stopwords": sorted(search.STOPWORDS),
    }


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        # SQLite reads LIMIT -1 as "no limit", and a negative slice drops the last items.
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def main(argv: list[str] | None = None, model: ChatModel | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on known questions.")
    parser.add_argument("command", choices=("check", "gaps", "run"))
    parser.add_argument("feature", nargs="?", help="for gaps: a feature id, e.g. wals:81A")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=_non_negative, help="use only the first N items")
    parser.add_argument("--only", nargs="+", metavar="ID", help="use only these item ids")
    parser.add_argument("-m", "--model", help="model name, e.g. aliceai-llm")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="score language and retrieval only"
    )
    parser.add_argument(
        "--feature-text-limit", type=_non_negative, help="override FEATURE_TEXT_LIMIT"
    )
    parser.add_argument("--max-features", type=_non_negative, help="override MAX_FEATURES")
    args = parser.parse_args(argv)
    if args.feature_text_limit is not None:
        context_settings.FEATURE_TEXT_LIMIT = args.feature_text_limit
    if args.max_features is not None:
        context_settings.MAX_FEATURES = args.max_features
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if args.command == "gaps":
            if not args.feature:
                parser.error("gaps needs a feature id")
            for glottocode, name, facts, values in gaps(conn, args.feature, args.limit or 20):
                state = "no entry" if values is None else f"recorded: {values}"
                if values == "?":
                    state = "coded ?"
                print(f"{glottocode} | {name} | {facts} known facts | {state}")
            return
        items = load_items(args.questions)
        if args.only:
            wanted = set(args.only)
            unknown = wanted - {item.id for item in items}
            if unknown:
                parser.error(f"unknown item ids: {', '.join(sorted(unknown))}")
            items = [item for item in items if item.id in wanted]
        items = items[: args.limit]
        checked = [(item, item_problem(conn, item)) for item in items]
        problems = [f"{item.id}: {problem}" for item, problem in checked if problem]
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
        usable = [item for item, problem in checked if problem is None]
        model = model or CachedModel(YandexModel(args.model), use_cache=not args.no_cache)
        REPORTS.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w.-]", "_", model.name)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report = REPORTS / f"eval-{safe_name}-{stamp}-{time.time_ns() % 10**6:06d}.json"
        scores: list[Score] = []
        try:
            runs = evaluate_items(conn, model, usable, args.retrieval_only)
            for number, s in enumerate(runs, start=1):
                scores.append(s)
                verdict = s.error or (s.correct if s.correct is not None else s.honest)
                missing = "" if s.retrieved else f" query={s.query!r} features={list(s.features)}"
                print(
                    f"[{number}/{len(usable)}] {s.id}: language={s.language} "
                    f"retrieved={s.retrieved} ok={verdict}{missing}",
                    flush=True,
                )
        finally:
            summary = summarize(scores)
            try:
                write_report(
                    report,
                    {
                        "model": model.name,
                        "mode": "retrieval only" if args.retrieval_only else "full",
                        **experiment_identity(args.questions),
                        "selection": {"only": args.only, "limit": args.limit},
                        "finished": len(scores) == len(usable),
                        "left_out": problems,
                        "summary": summary,
                        "items": [asdict(s) for s in scores],
                    },
                )
            except Exception as error:  # noqa: BLE001 - never hide the original exception
                print(f"Warning: the report could not be written: {error}", file=sys.stderr)
    print()
    for name, value in summary.items():
        print(f"{name}: {value}")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
