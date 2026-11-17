"""Assemble the evidence for one question: matching texts and the language's recorded values.

This is the retrieval half of question answering. The rendered context lists every piece of
evidence with a label ([V1], [T1]) and a link, so an answer can cite what it relies on.
"""

import argparse
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field

from glotto_lens.documents import DISPLAY_NAMES, FEATURE_URLS
from glotto_lens.languages import (
    UNKNOWN_LABEL,
    Fact,
    Languoid,
    find_languages,
    language_profile,
)
from glotto_lens.search import Hit, search
from glotto_lens.sources import DB_PATH

TEXT_LIMIT = 5
LANGUAGE_TEXT_LIMIT = 3
MAX_FEATURES = 8
VALUE_URLS = {
    "wals": "https://wals.info/datapoint/{feature}/wals_code_{entry}",
    "grambank": "https://grambank.clld.org/languages/{entry}",
}

CHAPTER_FEATURES = """
    SELECT f.id FROM document_chapters dc
    JOIN features f ON f.chapter_id = dc.chapter_id AND f.source = 'wals'
    WHERE dc.document_id = ? ORDER BY dc.chapter_id, f.rowid"""

FEATURE = "SELECT source, local_id, name FROM features WHERE id = ?"


@dataclass(frozen=True)
class FeatureEvidence:
    """A feature the texts point to, with the language's values for it (may be empty)."""

    feature_id: str
    source: str
    local_id: str
    name: str
    url: str
    facts: tuple[Fact, ...] = ()


@dataclass
class Context:
    question: str
    language_query: str | None = None
    languoid: Languoid | None = None
    other_candidates: list[Languoid] = field(default_factory=list)
    status: dict[str, str | None] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    evidence: list[FeatureEvidence] = field(default_factory=list)


def _merge(*groups: list[Hit]) -> list[Hit]:
    """Keep the first occurrence of each document and of each identical passage."""
    merged, documents, snippets = [], set(), set()
    for hit in (hit for group in groups for hit in group):
        if hit.document_id in documents or hit.snippet in snippets:
            continue
        documents.add(hit.document_id)
        snippets.add(hit.snippet)
        merged.append(hit)
    return merged


def related_features(conn: sqlite3.Connection, hits: list[Hit]) -> list[str]:
    """Features the hits are about: a card names its feature, a chapter names its maps."""
    features: list[str] = []
    for hit in hits:
        if hit.feature_id:
            features.append(hit.feature_id)
        else:
            features += [row[0] for row in conn.execute(CHAPTER_FEATURES, (hit.document_id,))]
    return list(dict.fromkeys(features))[:MAX_FEATURES]


def build_context(conn: sqlite3.Connection, question: str, language: str | None = None) -> Context:
    """Collect texts for the question and, if a language is given, its values."""
    context = Context(question, language)
    candidates = find_languages(conn, language) if language else []
    if candidates:
        context.languoid, *context.other_candidates = candidates
    glottocode = context.languoid.glottocode if context.languoid else None

    general = search(conn, question, TEXT_LIMIT)
    mentioning = search(conn, question, LANGUAGE_TEXT_LIMIT, glottocode) if glottocode else []
    context.hits = _merge(general, mentioning)

    profile = language_profile(conn, glottocode) if glottocode else None
    if profile:
        context.status = profile.status
    for feature_id in related_features(conn, context.hits):
        source, local_id, name = conn.execute(FEATURE, (feature_id,)).fetchone()
        facts = tuple(
            fact
            for fact in (profile.facts if profile else [])
            if f"{fact.source}:{fact.feature}" == feature_id
        )
        url = FEATURE_URLS[source].format(local_id)
        context.evidence.append(FeatureEvidence(feature_id, source, local_id, name, url, facts))
    return context


def _value_line(fact: Fact) -> str:
    value = UNKNOWN_LABEL if fact.value is None else fact.value
    notes = [f"entry: {fact.entry}"]
    if fact.from_dialect:
        notes.append(f"recorded for dialect {fact.glottocode}")
    elif not fact.main:
        notes.append("secondary entry")
    url = VALUE_URLS[fact.source].format(feature=fact.feature, entry=fact.entry_id)
    return f"{value} ({'; '.join(notes)}) {url}"


def render_context(context: Context) -> str:
    """Render the context as plain text with labelled, linked evidence."""
    lines = [f"Question: {context.question}", ""]
    languoid = context.languoid
    if context.language_query and languoid is None:
        lines += [f"Language: {context.language_query!r} was not found in Glottolog.", ""]
    elif languoid:
        lines.append(
            f"Language: {languoid.name} (glottocode {languoid.glottocode}; "
            f"family {languoid.family or 'none (isolate or unclassified)'}; "
            f"area {languoid.macroarea}; https://glottolog.org/resource/languoid/id/"
            f"{languoid.glottocode})"
        )
        if context.other_candidates:
            others = ", ".join(f"{c.name} ({c.glottocode})" for c in context.other_candidates)
            lines.append(f"Other languages matching {context.language_query!r}: {others}")
        for key, value in context.status.items():
            lines.append(f"Glottolog {key}: {value}")
        lines += ["", f"Values recorded for {languoid.name}:"]
        number = 0
        for evidence in context.evidence:
            source = DISPLAY_NAMES[evidence.source]
            title = f"{source} {evidence.local_id} {evidence.name}"
            if not evidence.facts:
                number += 1
                lines.append(f"[V{number}] {title}: no value recorded {evidence.url}")
            for fact in evidence.facts:
                number += 1
                lines.append(f"[V{number}] {title}: {_value_line(fact)}")
        lines.append("")
    lines.append("Texts:")
    for number, hit in enumerate(context.hits, start=1):
        lines.append(f"[T{number}] {hit.title} / {hit.section} {' '.join(hit.urls)}")
        lines += [hit.text, ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show the evidence collected for a question.")
    parser.add_argument("question", help="question in English, e.g. 'word order'")
    parser.add_argument("-l", "--language", help="language name or code, e.g. Russian or rus")
    args = parser.parse_args(argv)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        print(render_context(build_context(conn, args.question, args.language)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
