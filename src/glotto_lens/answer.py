"""Answer a question: rewrite it for search, collect evidence, let the model answer.

The pipeline has three steps:
1. the model turns the question (in any language) into a language name and an English
   search query in typological terms;
2. retrieval collects labelled evidence for that query and language (context.py);
3. the model answers from the evidence only, citing labels such as [V1] and [T2].
Citations are then checked against the evidence: a label the context does not contain
is reported as invalid instead of being shown as a source.
"""

import argparse
import json
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field

from glotto_lens import prompts
from glotto_lens.context import Context, Source, build_context, render_context, sources
from glotto_lens.llm import CachedModel, ChatModel, Completion, YandexModel
from glotto_lens.sources import DB_PATH

LABEL = re.compile(r"\[([VT]\d+)\]")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Rewrite:
    language: str | None
    query: str


@dataclass
class Answer:
    question: str
    rewrite: Rewrite
    context: Context
    text: str
    cited: list[Source] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)

    @property
    def tokens(self) -> tuple[int, int]:
        """Input and output tokens of the calls that were not served from the cache."""
        fresh = [c for c in self.completions if not c.cached]
        return sum(c.input_tokens for c in fresh), sum(c.output_tokens for c in fresh)


def parse_rewrite(text: str, question: str) -> Rewrite:
    """Read the model's JSON; fall back to searching the question as it is."""
    match = JSON_OBJECT.search(text)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    language = data.get("language")
    query = data.get("query")
    return Rewrite(
        language=language.strip() or None if isinstance(language, str) else None,
        query=query.strip() if isinstance(query, str) and query.strip() else question,
    )


def check_citations(text: str, context: Context) -> tuple[list[Source], list[str]]:
    """Split the labels cited in the answer into known sources and unknown labels."""
    known = sources(context)
    labels = list(dict.fromkeys(LABEL.findall(text)))
    return [known[label] for label in labels if label in known], [
        label for label in labels if label not in known
    ]


def answer_question(
    conn: sqlite3.Connection, model: ChatModel, question: str, language: str | None = None
) -> Answer:
    """Run the whole pipeline; an explicit language overrides the one the model finds."""
    rewrite_call = model.complete(
        prompts.REWRITE_SYSTEM, prompts.REWRITE_USER.format(question=question), max_tokens=200
    )
    rewrite = parse_rewrite(rewrite_call.text, question)
    if language:
        rewrite = Rewrite(language, rewrite.query)
    context = build_context(conn, rewrite.query, rewrite.language)
    evidence = render_context(context)
    answer_call = model.complete(
        prompts.ANSWER_SYSTEM,
        prompts.ANSWER_USER.format(question=question, evidence=evidence),
        max_tokens=1000,
    )
    cited, invalid = check_citations(answer_call.text, context)
    return Answer(
        question, rewrite, context, answer_call.text, cited, invalid, [rewrite_call, answer_call]
    )


def render_answer(answer: Answer, show_context: bool = False) -> str:
    lines = [
        f"Search: {answer.rewrite.query!r}, language: {answer.rewrite.language!r}",
        "",
        answer.text.strip(),
        "",
        "Sources:",
    ]
    lines += [f"[{source.label}] {source.title} {source.url}" for source in answer.cited]
    if not answer.cited:
        lines.append("(the answer cites no sources)")
    if answer.invalid_labels:
        lines.append(f"Labels not found in the evidence: {', '.join(answer.invalid_labels)}")
    input_tokens, output_tokens = answer.tokens
    cached = sum(c.cached for c in answer.completions)
    lines.append(
        f"\nTokens: {input_tokens} in, {output_tokens} out; "
        f"{cached} of {len(answer.completions)} calls from cache"
    )
    if show_context:
        lines += ["", "=" * 40, render_context(answer.context)]
    return "\n".join(lines)


def main(argv: list[str] | None = None, model: ChatModel | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer a question about a language.")
    parser.add_argument("question", help="question in any language")
    parser.add_argument("-l", "--language", help="language name or code, overrides detection")
    parser.add_argument("-m", "--model", help="model name, e.g. aliceai-llm")
    parser.add_argument("--show-context", action="store_true", help="print the evidence too")
    args = parser.parse_args(argv)
    model = model or CachedModel(YandexModel(args.model))
    with closing(sqlite3.connect(DB_PATH)) as conn:
        answer = answer_question(conn, model, args.question, args.language)
    print(render_answer(answer, args.show_context))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
