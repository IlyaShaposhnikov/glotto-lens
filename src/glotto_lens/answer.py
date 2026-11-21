"""Answer a question: rewrite it for search, collect evidence, let the model answer.

The pipeline has three steps:
1. the model turns the question (in any language) into a language name and an English
   search query in typological terms;
2. retrieval collects labelled evidence for that query and language (context.py);
3. the model answers from the evidence only, citing labels such as [V1] and [T2].
Citations are then checked against the evidence: a label the context does not contain is
marked in the answer text and listed separately instead of being shown as a source.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from glotto_lens import prompts
from glotto_lens.context import Citation, Context, build_context, citations, render_context
from glotto_lens.documents import RELEASES
from glotto_lens.llm import APIS, CachedModel, ChatModel, Completion, ModelError, YandexModel
from glotto_lens.sources import DATA_DIR, DB_PATH

ANSWER_LOG = DATA_DIR / "answers.jsonl"
BRACKETS = re.compile(r"\[([^\[\]]+)\]")
LABEL = re.compile(r"[VT]\s?\d+", re.IGNORECASE)
NO_LANGUAGE = frozenset({"null", "none", "unknown", "n/a", "na", "-", "нет"})


# part of every cache key: the LangChain chain uses the same limits to share the cache
REWRITE_MAX_TOKENS = 200
ANSWER_MAX_TOKENS = 1000


@dataclass(frozen=True)
class Rewrite:
    language: str | None
    query: str
    ok: bool = True  # False: the model's output could not be read, the question is searched


@dataclass
class Answer:
    question: str
    rewrite: Rewrite
    context: Context
    text: str
    cited: list[Citation] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)

    @property
    def tokens(self) -> tuple[int, int]:
        """Input and output tokens of the calls that were not served from the cache."""
        fresh = [c for c in self.completions if not c.cached]
        return sum(c.input_tokens for c in fresh), sum(c.output_tokens for c in fresh)

    @property
    def complete(self) -> bool:
        """Whether the final answer was finished (the rewrite has its own flag)."""
        return self.completions[-1].complete if self.completions else True


def first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def parse_rewrite(text: str, question: str) -> Rewrite:
    """Read the model's JSON; fall back to searching the question as it is."""
    data = first_json_object(text)
    if data is None:
        return Rewrite(language=None, query=question, ok=False)
    language, query = data.get("language"), data.get("query")
    language = language.strip() if isinstance(language, str) else None
    if not language or language.casefold() in NO_LANGUAGE:
        language = None
    if not isinstance(query, str) or not query.strip():
        return Rewrite(language=language, query=question, ok=False)
    return Rewrite(language=language, query=query.strip())


def cited_labels(text: str) -> list[str]:
    """Labels in square brackets, also in lists such as [V1, T2] or [V1; see T2].

    Labels are normalized to the form of the evidence: [v 1] counts as V1.
    """
    labels = []
    for group in BRACKETS.findall(text):
        labels += [label.upper().replace(" ", "") for label in LABEL.findall(group)]
    return list(dict.fromkeys(labels))


def check_citations(text: str, context: Context) -> tuple[list[Citation], list[str]]:
    """Split the labels cited in the answer into known evidence and unknown labels."""
    known = citations(context)
    labels = cited_labels(text)
    return [known[label] for label in labels if label in known], [
        label for label in labels if label not in known
    ]


def mark_invalid(text: str, invalid: list[str]) -> str:
    """Flag unknown labels inside the answer text: [V9] becomes [V9?]."""
    if not invalid:
        return text
    # Match the labels as the model wrote them: [v 9] is the invalid label V9 too.
    variants = [rf"{label[0]}\s?{label[1:]}" for label in invalid]
    pattern = re.compile(r"\b(" + "|".join(variants) + r")\b(?!\?)", re.IGNORECASE)
    return BRACKETS.sub(lambda match: "[" + pattern.sub(r"\1?", match.group(1)) + "]", text)


def retrieve(
    conn: sqlite3.Connection, model: ChatModel, question: str, language: str | None = None
) -> tuple[Rewrite, Context, Completion]:
    """Steps 1 and 2: rewrite the question and collect the evidence for it."""
    safe_question = prompts.escape_tags(question)
    rewrite_call = model.complete(
        prompts.REWRITE_SYSTEM,
        prompts.REWRITE_USER.format(question=safe_question),
        max_tokens=REWRITE_MAX_TOKENS,
    )
    rewrite = parse_rewrite(rewrite_call.text, question)
    if language:
        rewrite = Rewrite(language=language, query=rewrite.query, ok=rewrite.ok)
    return rewrite, build_context(conn, rewrite.query, rewrite.language), rewrite_call


def answer_question(
    conn: sqlite3.Connection,
    model: ChatModel,
    question: str,
    language: str | None = None,
    rewrite_model: ChatModel | None = None,
) -> Answer:
    """Run the whole pipeline; an explicit language overrides the one the model finds.

    rewrite_model (default: model) does step 1. Keeping it fixed while the answer model
    changes gives every model the same evidence, so their answers can be compared.
    """
    rewrite, context, rewrite_call = retrieve(conn, rewrite_model or model, question, language)
    safe_question = prompts.escape_tags(question)
    evidence = prompts.escape_tags(render_context(context))
    answer_call = model.complete(
        prompts.ANSWER_SYSTEM,
        prompts.ANSWER_USER.format(question=safe_question, evidence=evidence),
        max_tokens=ANSWER_MAX_TOKENS,
    )
    cited, invalid = check_citations(answer_call.text, context)
    text = mark_invalid(answer_call.text, invalid)
    return Answer(question, rewrite, context, text, cited, invalid, [rewrite_call, answer_call])


def render_answer(answer: Answer, show_context: bool = False) -> str:
    lines = [f"Search: {answer.rewrite.query!r}, language: {answer.rewrite.language!r}"]
    if not answer.rewrite.ok:
        lines.append("Warning: the rewrite could not be read; the question was searched as is.")
    lines += ["", answer.text.strip(), ""]
    if not answer.complete:
        reason = answer.completions[-1].stop_reason or "unknown reason"
        lines.append(f"Warning: the model stopped early ({reason}); the answer may be cut.")
    if not answer.cited and citations(answer.context):
        lines.append("Warning: the answer cites no evidence.")
    if answer.cited:
        lines.append("Sources:")
        lines += [f"[{c.label}] {c.title} {c.url}" for c in answer.cited]
    if answer.invalid_labels:
        labels = ", ".join(answer.invalid_labels)
        lines.append(f"Labels not found in the evidence (marked with ?): {labels}")
    input_tokens, output_tokens = answer.tokens
    cached = sum(c.cached for c in answer.completions)
    lines.append(
        f"\nTokens: {input_tokens} in, {output_tokens} out; "
        f"{cached} of {len(answer.completions)} calls from cache"
    )
    if show_context:
        lines += ["", "=" * 40, render_context(answer.context)]
    return "\n".join(lines)


def log_answer(answer: Answer, model_name: str, path: Path | None = None) -> None:
    """Append the question, the retrieval decisions and the answer to a JSONL log."""
    path = path or ANSWER_LOG
    languoid = answer.context.languoid
    evidence = render_context(answer.context)
    record = {
        "time": time.time(),
        "model": model_name,
        "data": RELEASES,
        "question": answer.question,
        "rewrite": {"language": answer.rewrite.language, "query": answer.rewrite.query},
        "rewrite_ok": answer.rewrite.ok,
        "glottocode": languoid.glottocode if languoid else None,
        "features": [evidence.feature_id for evidence in answer.context.evidence],
        "documents": [hit.document_id for hit in answer.context.hits],
        "evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        "cited": [
            {"label": c.label, "title": c.title, "url": c.url, "feature": c.feature_id}
            for c in answer.cited
        ],
        "invalid": answer.invalid_labels,
        "complete": answer.complete,
        "tokens": answer.tokens,
        "answer": answer.text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None, model: ChatModel | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer a question about a language.")
    parser.add_argument("question", help="question in any language")
    parser.add_argument("-l", "--language", help="language name or code, overrides detection")
    parser.add_argument("-m", "--model", help="model name, e.g. aliceai-llm")
    parser.add_argument("--show-context", action="store_true", help="print the evidence too")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    parser.add_argument("--api", choices=APIS, help="AI Studio API to call (default: responses)")
    parser.add_argument(
        "--reasoning-effort", help='for reasoning models, e.g. "none" to answer without reasoning'
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="BM25 and embeddings (needs 'glotto-lens embed')"
    )
    args = parser.parse_args(argv)
    if args.hybrid:
        from glotto_lens import context
        from glotto_lens.vectors import check_ready, default_embedder

        try:
            context.EMBEDDER = default_embedder()
            with closing(sqlite3.connect(DB_PATH)) as conn:
                check_ready(conn, context.EMBEDDER.name)
        except RuntimeError as error:
            sys.exit(str(error))
    model = model or CachedModel(
        YandexModel(args.model, api=args.api, reasoning_effort=args.reasoning_effort),
        use_cache=not args.no_cache,
    )
    with closing(sqlite3.connect(DB_PATH)) as conn:
        try:
            answer = answer_question(conn, model, args.question, args.language)
        except ModelError as error:
            sys.exit(f"The model call failed: {error}")
        except RuntimeError as error:  # e.g. --hybrid before 'glotto-lens embed'
            sys.exit(str(error))
        log_answer(answer, model.name)
        print(render_answer(answer, args.show_context))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
