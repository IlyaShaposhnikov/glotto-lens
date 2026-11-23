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
- invalid citations: the answer cites labels that are not in the evidence (full runs only);
- with --judge, an LLM judge reads each answer with the recorded value: for value items,
  "content correct" = right language and the answer states the value in any wording (the
  citation is not required); for no_data items, "admits missing data" = the answer says
  the value is missing or unknown and gives none. It runs beside the phrase check, so the
  two can be compared. By default the judge is the model being judged; --judge-model
  names another one, which avoids a model grading its own wording.
Items that contradict the data are reported and left out of every metric.

Usage:
    python -m glotto_lens.evaluate check                 # validate the items, free
    python -m glotto_lens.evaluate gaps grambank:GB020   # candidates for no_data items
    python -m glotto_lens.evaluate run --limit 5         # answer and score
    python -m glotto_lens.evaluate run --judge           # plus an LLM judge per answer
"""

import argparse
import hashlib
import inspect
import json
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from glotto_lens import context as context_settings
from glotto_lens import prompts
from glotto_lens.answer import Answer, answer_question, first_json_object, retrieve
from glotto_lens.cliargs import non_negative
from glotto_lens.documents import RELEASES
from glotto_lens.languages import KNOWN_FACTS, find_languages, language_profile
from glotto_lens.llm import APIS, CachedModel, ChatModel, Completion, YandexModel
from glotto_lens.sources import DATA_DIR, DB_PATH, read_only

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
    # a value or the data, then up to 20 words naming the feature within the same sentence,
    # then the negation: "значение по соответствующему признаку в WALS не зафиксировано"
    r"|(данн|значени|сведени)\w*(\s+[^\s.!?;]+){0,20}?"
    r"\s+не (указ|зафиксир|приведен|записан|установлен)"
    r"|не удалось найти|не ?известн"
    r"|no (data|value|information)|not (record|known|available)"
    r"|not (provide|give|list)\w* (a |any )?(value|data|information)",
    re.IGNORECASE,
)
# Punctuation inside quotes does not end a sentence: in "значение для признака «Are there
# articles?» не записано" the question mark belongs to a feature name. Only that
# punctuation is removed; the quoted words stay, because an admission can itself be quoted:
# "значение отмечено как «не известно»".
QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”")
QUOTED_PUNCTUATION = re.compile(r"[.!?;]")


def admits_missing_data(text: str) -> bool:
    """The answer says that the data is missing (see MISSING_DATA)."""
    text = QUOTED.sub(lambda quote: QUOTED_PUNCTUATION.sub("", quote.group()), text)
    return bool(MISSING_DATA.search(text))


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
    judged: bool | None = None  # the judge's verdict (with --judge), see the module docstring
    judge_reason: str = ""
    judge_error: str | None = None  # the judge call failed (network, API): no verdict to count
    judge_input_tokens: int = 0  # kept apart from the answer's tokens: runs stay comparable
    judge_output_tokens: int = 0


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


FEATURE_NAME = "SELECT name FROM features WHERE id = ?"
# Reasoning models (DeepSeek) think before they answer and count those tokens as output:
# with 200 they ran out on the hardest items and wrote no verdict at all.
JUDGE_MAX_TOKENS = 2000  # 1000 still cut DeepSeek off once (kom-30A)


def expected_text(conn: sqlite3.Connection, item: Item, expected: Gold) -> str:
    """What the judge compares the answer with: the feature and what is recorded for it.

    Without a coded value there are three different states, and the judge is told which:
    a value written only as text, a value coded as not known (?), or no entry at all.
    """
    row = conn.execute(FEATURE_NAME, (item.feature,)).fetchone()
    feature = f"{item.feature} {row[0] if row else ''}".strip()
    profile = language_profile(conn, expected.glottocode) if expected.glottocode else None
    facts = [
        fact
        for fact in (profile.facts if profile else [])
        if f"{fact.source}:{fact.feature}" == item.feature
    ]
    if expected.value is not None:
        # where the value comes from, as the evidence says it: an answer that mentions the
        # dialect or the secondary entry is right, not hedging
        source = next((f for f in facts if f.known and f.value == expected.value), None)
        note = ""
        if source is not None and source.from_dialect:
            note = f" (recorded for its dialect {source.glottocode})"
        elif source is not None and not source.main:
            note = " (from a secondary entry of the dataset)"
        return f"Feature {feature}: {expected.value}{note}"
    texts = [fact.value for fact in facts if fact.value is not None and not fact.known]
    if texts:
        return f"Feature {feature}: no coded value; recorded only as text: " + "; ".join(texts)
    if facts:
        return f"Feature {feature}: the value is coded as not known (?) for this language."
    return f"Feature {feature}: no value is recorded for this language."


def judge(model: ChatModel, item: Item, expected_record: str, answer_text: str) -> Completion:
    return model.complete(
        prompts.JUDGE_SYSTEM,
        prompts.JUDGE_USER.format(
            question=prompts.escape_tags(item.question),
            expected=prompts.escape_tags(expected_record),
            answer=prompts.escape_tags(answer_text),
        ),
        max_tokens=JUDGE_MAX_TOKENS,
    )


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """The judge's verdict and reason; None when the reply cannot be read."""
    data = first_json_object(text)
    verdict = data.get("verdict") if data else None
    if isinstance(verdict, str) and verdict.strip().lower() in ("true", "false"):
        verdict = verdict.strip().lower() == "true"
    if not isinstance(verdict, bool):
        return None, f"unreadable verdict: {text.strip()[:200]}"
    return verdict, str(data.get("reason", ""))


def apply_judge(score_: Score, call: Completion) -> Score:
    if call.complete:
        verdict, reason = parse_verdict(call.text)
    else:
        verdict, reason = None, f"judge reply cut off ({call.stop_reason or 'unknown reason'})"
    if verdict is not None:
        # the judge does not know the glottocode: an answer about another language is wrong,
        # whether it gives a value or admits a gap
        verdict = verdict and score_.language
    return replace(
        score_,
        judged=verdict,
        judge_reason=reason,
        judge_input_tokens=0 if call.cached else call.input_tokens,
        judge_output_tokens=0 if call.cached else call.output_tokens,
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
        **_judge_summary(answered),
        "tokens (in / out)": (
            f"{sum(s.input_tokens for s in scores)} / {sum(s.output_tokens for s in scores)}"
        ),
    }


def _judge_summary(answered: list[Score]) -> dict[str, str]:
    judged = [s for s in answered if s.judged is not None]
    if not judged and not any(s.judge_reason or s.judge_error for s in answered):
        return {}  # the judge was not used
    values = [s for s in judged if s.expect == "value"]
    no_data_items = [s for s in judged if s.expect == "no_data"]
    return {
        "judge: content correct": _share([s.judged for s in values]),
        "judge: admits missing data": _share([s.judged for s in no_data_items]),
        # "correct" also needs the right citation, so the two disagree by design sometimes
        "judge yes, citation check no": str(sum(s.judged and not s.correct for s in values)),
        "judge no, citation check yes": str(sum(s.correct and not s.judged for s in values)),
        "judge agrees with honest": _share([s.judged == s.honest for s in no_data_items]),
        # a failed call is not an unreadable verdict: rerun, the cache keeps the others
        "judge calls failed (rerun to fill)": str(sum(s.judge_error is not None for s in answered)),
        "judge verdicts unreadable": str(
            sum(s.judged is None and s.judge_error is None for s in answered)
        ),
        "judge tokens (in / out)": (
            f"{sum(s.judge_input_tokens for s in answered)}"
            f" / {sum(s.judge_output_tokens for s in answered)}"
        ),
    }


def evaluate_items(
    conn: sqlite3.Connection,
    model: ChatModel,
    items: Iterable[Item],
    retrieval_only: bool = False,
    use_judge: bool = False,
    judge_model: ChatModel | None = None,
    rewrite_model: ChatModel | None = None,
    embedder=None,
) -> Iterator[Score]:
    """Score items one by one; a failing call is recorded as an error and the run goes on.

    With retrieval_only, only the rewrite is requested (usually from the cache), so
    retrieval settings can be compared for free. With use_judge, each answer also gets an
    LLM verdict from judge_model (default: model); a failing judge call leaves it empty.
    rewrite_model (default: model) rewrites the questions, so that several answer models can
    be compared on the same evidence. embedder: None for BM25, an embedder for hybrid.
    """
    for item in items:
        expected = Gold(item.glottocode, None)
        try:
            expected = gold(conn, item)
            if retrieval_only:
                rewriter = rewrite_model or model
                found = retrieve(conn, rewriter, item.question, embedder=embedder)
                yield score_retrieval(item, expected, *found)
            else:
                answer = answer_question(
                    conn, model, item.question, rewrite_model=rewrite_model, embedder=embedder
                )
                result = score(item, expected, answer)
                if use_judge:
                    record = expected_text(conn, item, expected)
                    try:
                        verdict = judge(judge_model or model, item, record, result.answer)
                        result = apply_judge(result, verdict)
                    except Exception as error:  # noqa: BLE001 - keep the paid answer
                        result = replace(result, judge_error=f"{type(error).__name__}: {error}")
                yield result
        except Exception as error:  # noqa: BLE001 - one bad call must not lose a paid run
            yield Score(
                id=item.id,
                question=item.question,
                expect=item.expect,
                gold_glottocode=expected.glottocode,
                gold_value=expected.value,
                error=f"{type(error).__name__}: {error}",
            )


def evaluate(
    conn: sqlite3.Connection,
    model: ChatModel,
    items: Iterable[Item],
    retrieval_only: bool = False,
    use_judge: bool = False,
    judge_model: ChatModel | None = None,
    rewrite_model: ChatModel | None = None,
    embedder=None,
) -> list[Score]:
    """All scores at once, see evaluate_items."""
    return list(
        evaluate_items(
            conn, model, items, retrieval_only, use_judge, judge_model, rewrite_model, embedder
        )
    )


def write_report(path: Path, data: dict) -> None:
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def metric_fingerprint() -> str:
    """Hash of everything that defines the honesty metric: its patterns and the code that
    applies them. A missing source (a frozen build) must not cost a paid run its report."""
    try:
        source = inspect.getsource(admits_missing_data)
    except OSError:
        source = "source unavailable"
    parts = (MISSING_DATA.pattern, QUOTED.pattern, QUOTED_PUNCTUATION.pattern, source)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def code_version() -> str | None:
    """The git commit of the code, with "+dirty" for uncommitted changes; None outside git.

    Prompts, metric and retrieval settings are hashed separately, but the policy around
    them (feature selection, query building, document chunking) is only in the code.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}+dirty" if changes else commit


def experiment_identity(
    items_path: Path, use_judge: bool = False, judge_model: str | None = None, embedder=None
) -> dict:
    """What a report depends on: the code commit, data releases, prompts, the question set,
    the metric, the retrieval settings and, with a judge, the judge prompts."""
    prompt_text = "".join(
        (prompts.REWRITE_SYSTEM, prompts.REWRITE_USER, prompts.ANSWER_SYSTEM, prompts.ANSWER_USER)
    )
    identity = {
        "code": code_version(),
        "data": RELEASES,
        "prompts_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "questions_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
        "metrics_sha256": metric_fingerprint(),
        "retrieval": retrieval_settings(embedder),
    }
    if use_judge:
        identity["judge_sha256"] = _sha256(prompts.JUDGE_SYSTEM + prompts.JUDGE_USER)
        identity["judge_model"] = judge_model
        identity["judge_max_tokens"] = JUDGE_MAX_TOKENS
    return identity


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def retrieval_settings(embedder=None) -> dict:
    """Retrieval constants that change results, recorded with every report.

    embedder: the one the run searched with; None for BM25 runs."""
    from glotto_lens import context, search

    settings = {
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
    if embedder is not None:  # absent for BM25 runs, so their identity is unchanged
        from glotto_lens import vectors

        try:  # read-only: a report never creates the database
            with closing(read_only(DB_PATH)) as conn:
                settings["hybrid"] = vectors.settings(conn, embedder)
        except sqlite3.OperationalError:
            settings["hybrid"] = vectors.settings(embedder=embedder)
    return settings


def main(
    argv: list[str] | None = None,
    model: ChatModel | None = None,
    judge_model: ChatModel | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on known questions.")
    parser.add_argument("command", choices=("check", "gaps", "run"))
    parser.add_argument("feature", nargs="?", help="for gaps: a feature id, e.g. wals:81A")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=non_negative, help="use only the first N items")
    parser.add_argument("--only", nargs="+", metavar="ID", help="use only these item ids")
    parser.add_argument("-m", "--model", help="model name, e.g. aliceai-llm")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="score language and retrieval only"
    )
    parser.add_argument(
        "--feature-text-limit", type=non_negative, help="override FEATURE_TEXT_LIMIT"
    )
    parser.add_argument("--max-features", type=non_negative, help="override MAX_FEATURES")
    parser.add_argument(
        "--hybrid", action="store_true", help="BM25 and embeddings (needs 'glotto-lens embed')"
    )
    parser.add_argument(
        "--fusion", choices=("slots", "rrf"), help="how --hybrid fuses the rankings"
    )
    parser.add_argument(
        "--judge", action="store_true", help="add an LLM verdict on each answer (full runs)"
    )
    parser.add_argument("--judge-model", help="model for the judge (default: the answer model)")
    parser.add_argument("--api", choices=APIS, help="AI Studio API for the answer model")
    parser.add_argument("--judge-api", choices=APIS, help="AI Studio API for the judge model")
    parser.add_argument(
        "--rewrite-model",
        help="model for the rewrite step (default: the answer model); fix it to compare models",
    )
    parser.add_argument("--reasoning-effort", help='answer model, e.g. "none" for no reasoning')
    parser.add_argument("--judge-reasoning-effort", help="the same for the judge model")
    args = parser.parse_args(argv)
    if args.judge and args.retrieval_only:
        parser.error("--judge needs answers; drop --retrieval-only")
    if (args.judge_model or args.judge_api or args.judge_reasoning_effort) and not args.judge:
        parser.error("--judge-model, --judge-api and --judge-reasoning-effort need --judge")
    if args.feature_text_limit is not None:
        context_settings.FEATURE_TEXT_LIMIT = args.feature_text_limit
    if args.max_features is not None:
        context_settings.MAX_FEATURES = args.max_features
    if args.fusion and not args.hybrid:
        parser.error("--fusion needs --hybrid")
    embedder = None
    if args.hybrid:
        from glotto_lens import vectors

        if args.fusion:
            vectors.FUSION = args.fusion
        try:
            embedder = vectors.default_embedder()
            with closing(sqlite3.connect(DB_PATH)) as conn:
                vectors.check_ready(conn, embedder.name)
        except RuntimeError as error:
            parser.error(str(error))
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if args.command == "gaps":
            if not args.feature:
                parser.error("gaps needs a feature id")
            limit = 20 if args.limit is None else args.limit
            for glottocode, name, facts, values in gaps(conn, args.feature, limit):
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
        model = model or CachedModel(
            YandexModel(args.model, api=args.api, reasoning_effort=args.reasoning_effort),
            use_cache=not args.no_cache,
        )
        rewrite_model = None
        if args.rewrite_model:
            rewrite_model = CachedModel(
                YandexModel(args.rewrite_model), use_cache=not args.no_cache
            )
        if judge_model is None and (
            args.judge_model or args.judge_api or args.judge_reasoning_effort
        ):
            judge_model = CachedModel(
                YandexModel(
                    args.judge_model,
                    api=args.judge_api,
                    reasoning_effort=args.judge_reasoning_effort,
                ),
                use_cache=not args.no_cache,
            )
        judge_model = judge_model or model
        REPORTS.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w.-]", "_", model.name)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report = REPORTS / f"eval-{safe_name}-{stamp}-{time.time_ns() % 10**6:06d}.json"
        scores: list[Score] = []
        try:
            runs = evaluate_items(
                conn,
                model,
                usable,
                args.retrieval_only,
                args.judge,
                judge_model,
                rewrite_model,
                embedder,
            )
            for number, s in enumerate(runs, start=1):
                scores.append(s)
                verdict = s.error or (s.correct if s.correct is not None else s.honest)
                if args.judge:
                    verdict = f"{verdict} judge={'error' if s.judge_error else s.judged}"
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
                        "model_settings": getattr(model, "settings", {}),
                        "rewrite_model": (rewrite_model or model).name,
                        **(
                            {"judge_settings": getattr(judge_model, "settings", {})}
                            if args.judge
                            else {}
                        ),
                        "mode": "retrieval only" if args.retrieval_only else "full",
                        **experiment_identity(
                            args.questions, args.judge, judge_model.name, embedder
                        ),
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
