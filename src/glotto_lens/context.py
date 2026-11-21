"""Assemble the evidence for one question: matching texts and the language's recorded values.

This is the retrieval half of question answering. The rendered context lists every piece of
evidence with a label ([V1], [T1]) and a link, so an answer can cite what it relies on.

Which features to report values for:
- feature documents (cards and their sections) name their feature exactly, so they decide.
  Features are chosen from a pool of the best-matching features, each with its best
  document only (MAX_FEATURES of them, or more when more texts are shown): a Grambank
  feature has several section documents, which would otherwise fill the pool with one
  feature. The best documents of the first
  FEATURE_TEXT_LIMIT features are shown as texts; the other features add value lines (one
  short line per feature), not long passages;
- only when no feature document matches, the chapter of the best chapter text decides.
Chapter texts found because they mention the language are kept only if their chapter
describes one of those features; otherwise they are mentions of the language on
unrelated topics.
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
FEATURE_TEXT_LIMIT = 3  # of their documents, shown as texts
LANGUAGE_TEXT_LIMIT = 3
MAX_FEATURES = 8
# An embedder (glotto_lens.vectors) turns on hybrid search for the feature pool: BM25 and
# embeddings fused with RRF. None: BM25 only.
EMBEDDER = None
SETTING = object()  # build_context(embedder=SETTING): use the module's EMBEDDER (the CLI)
VALUE_URLS = {
    "wals": "https://wals.info/datapoint/{feature}/wals_code_{entry}",
    "grambank": "https://grambank.clld.org/valuesets/{feature}-{entry}",
}
STATUS_NAMES = {"aes": "endangerment status", "med": "most extensive description"}

CHAPTER_FEATURES = """
    SELECT f.id FROM document_chapters dc
    JOIN features f ON f.chapter_id = dc.chapter_id AND f.source = 'wals'
    WHERE dc.document_id = ? ORDER BY dc.chapter_id, f.rowid"""

DOCUMENT_CHAPTERS = "SELECT chapter_id FROM document_chapters WHERE document_id = ?"

AUTHORS = """
    SELECT c.contributors FROM document_chapters dc JOIN chapters c ON c.id = dc.chapter_id
    WHERE dc.document_id = ? AND c.contributors IS NOT NULL ORDER BY c.rowid LIMIT 1"""

FEATURE = "SELECT source, local_id, name, chapter_id FROM features WHERE id = ?"


@dataclass(frozen=True)
class FeatureEvidence:
    """A feature the texts point to, with the language's values for it (may be empty)."""

    feature_id: str
    source: str
    local_id: str
    name: str
    url: str
    facts: tuple[Fact, ...] = ()


@dataclass(frozen=True)
class Citation:
    """A labelled piece of evidence as shown to the model: [V1], [T1], ..."""

    label: str
    title: str
    url: str
    feature_id: str | None = None  # for [V…]: the feature the value belongs to
    value: str | None = None  # for [V…]: the recorded value, None when there is none
    known: bool = False  # for [V…]: the value is coded (not '?' and not free text)


@dataclass
class Context:
    question: str
    language_query: str | None = None
    languoid: Languoid | None = None
    other_candidates: list[Languoid] = field(default_factory=list)
    status: dict[str, str | None] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    authors: dict[int, str] = field(default_factory=dict)
    evidence: list[FeatureEvidence] = field(default_factory=list)
    dropped_features: int = 0


def _merge(*groups: list[Hit]) -> list[Hit]:
    """Join hit lists, drop repeated passages and order everything by score."""
    merged, seen = [], set()
    for hit in (hit for group in groups for hit in group):
        key = (hit.title, hit.section, hit.text)
        if hit.document_id in seen or key in seen:
            continue
        seen.update({hit.document_id, key})
        merged.append(hit)
    return sorted(merged, key=lambda hit: (-hit.score, hit.document_id))


def related_features(conn: sqlite3.Connection, hits: list[Hit]) -> list[str]:
    """Features the hits are about: feature documents first, else the best chapter's maps."""
    features = [hit.feature_id for hit in hits if hit.feature_id]
    if not features:
        chapters = [hit for hit in hits if hit.kind == "chapter"]
        if chapters:
            rows = conn.execute(CHAPTER_FEATURES, (chapters[0].document_id,))
            features = [row[0] for row in rows]
    return list(dict.fromkeys(features))


def pool_size() -> int:
    """How many features are searched. The pool has one document per feature, so features
    beyond MAX_FEATURES would be dropped anyway; it still covers the shown texts."""
    return max(MAX_FEATURES, FEATURE_TEXT_LIMIT)


def build_context(
    conn: sqlite3.Connection,
    question: str,
    language: str | None = None,
    embedder=SETTING,
) -> Context:
    """Collect texts for the question and, if a language is given, its values.

    embedder: an embedder for hybrid search, None for BM25 only, or by default the module
    setting EMBEDDER. Library callers pass it, so concurrent calls never share state.
    """
    if embedder is SETTING:
        embedder = EMBEDDER
    context = Context(question, language)
    candidates = find_languages(conn, language) if language else []
    if candidates:
        context.languoid, *context.other_candidates = candidates
    glottocode = context.languoid.glottocode if context.languoid else None

    if embedder is not None:
        from glotto_lens.vectors import hybrid_features  # numpy only when it is used

        # Fused scores (about 1/60) cannot be sorted together with BM25 scores (units to
        # tens), so the pool alone ranks the features, and the general search adds only
        # chapter texts: its BM25-ranked feature documents would bring back the order the
        # hybrid pool is there to correct.
        general = search(conn, question, TEXT_LIMIT, kind="chapter")
        pool = hybrid_features(conn, question, pool_size(), embedder)
        features = related_features(conn, [*pool, *general])
    else:
        general = search(conn, question, TEXT_LIMIT)
        pool = search(conn, question, pool_size(), per_feature=True)
        features = related_features(conn, _merge(general, pool))
    # texts only for features whose values are shown
    found = _merge(general, pool[: min(FEATURE_TEXT_LIMIT, MAX_FEATURES)])
    context.dropped_features = max(0, len(features) - MAX_FEATURES)
    features = features[:MAX_FEATURES]

    rows = [conn.execute(FEATURE, (feature_id,)).fetchone() for feature_id in features]
    chapters = {row[3] for row in rows if row[0] == "wals" and row[3]}
    mentioning = [
        hit
        for hit in (search(conn, question, LANGUAGE_TEXT_LIMIT, glottocode) if glottocode else [])
        if any(row[0] in chapters for row in conn.execute(DOCUMENT_CHAPTERS, (hit.document_id,)))
    ]
    context.hits = _merge(found, mentioning)
    for hit in context.hits:
        author = conn.execute(AUTHORS, (hit.document_id,)).fetchone()
        if author and hit.kind == "chapter":
            context.authors[hit.document_id] = author[0]

    profile = language_profile(conn, glottocode) if glottocode else None
    if profile:
        context.status = profile.status
    for feature_id, (source, local_id, name, _) in zip(features, rows, strict=True):
        facts = tuple(
            fact
            for fact in (profile.facts if profile else [])
            if f"{fact.source}:{fact.feature}" == feature_id
        )
        url = FEATURE_URLS[source].format(local_id)
        context.evidence.append(FeatureEvidence(feature_id, source, local_id, name, url, facts))
    return context


def _value_text(fact: Fact) -> str:
    value = UNKNOWN_LABEL if fact.value is None else fact.value
    notes = [f"entry: {fact.entry}"]
    if fact.from_dialect:
        notes.append(f"recorded for dialect {fact.glottocode}")
    elif not fact.main:
        notes.append("secondary entry")
    return f"{value} ({'; '.join(notes)})"


def value_citations(context: Context) -> list[Citation]:
    """The [V…] lines of the context, in the order they are rendered."""
    if context.languoid is None:
        return []
    found: list[Citation] = []
    for evidence in context.evidence:
        title = f"{DISPLAY_NAMES[evidence.source]} {evidence.local_id} {evidence.name}"
        if not evidence.facts:
            label = f"V{len(found) + 1}"
            text = f"{title}: no value recorded"
            found.append(Citation(label, text, evidence.url, evidence.feature_id))
        for fact in evidence.facts:
            url = VALUE_URLS[fact.source].format(feature=fact.feature, entry=fact.entry_id)
            text = f"{title}: {_value_text(fact)}"
            label = f"V{len(found) + 1}"
            found.append(Citation(label, text, url, evidence.feature_id, fact.value, fact.known))
    return found


def text_citations(context: Context) -> list[Citation]:
    """The [T…] passages of the context, in the order they are rendered."""
    found = []
    for number, hit in enumerate(context.hits, start=1):
        author = context.authors.get(hit.document_id)
        by = f" (by {author})" if author else ""
        found.append(Citation(f"T{number}", f"{hit.title} / {hit.section}{by}", " ".join(hit.urls)))
    return found


def citations(context: Context) -> dict[str, Citation]:
    """All labelled evidence of the context by label."""
    found = value_citations(context) + text_citations(context)
    return {citation.label: citation for citation in found}


def _language_lines(context: Context) -> list[str]:
    languoid = context.languoid
    if languoid is None:
        if context.language_query:
            return [f"Language: {context.language_query!r} was not found in Glottolog.", ""]
        return []
    lines = [
        f"Language: {languoid.name} (glottocode {languoid.glottocode}; "
        f"family {languoid.family or 'none (isolate or unclassified)'}; "
        f"area {languoid.macroarea}; "
        f"https://glottolog.org/resource/languoid/id/{languoid.glottocode})"
    ]
    if context.other_candidates:
        others = ", ".join(f"{c.name} ({c.glottocode})" for c in context.other_candidates)
        lines.append(f"Other languages matching {context.language_query!r}: {others}")
    for key, value in context.status.items():
        lines.append(f"Glottolog {STATUS_NAMES.get(key, key)}: {value}")
    lines += ["", f"Values recorded for {languoid.name}:"]
    for source in value_citations(context):
        lines.append(f"[{source.label}] {source.title} {source.url}")
    if not context.evidence:
        lines.append("No feature matched the question.")
    if context.dropped_features:
        lines.append(f"({context.dropped_features} more related features not shown.)")
    return [*lines, ""]


def render_context(context: Context) -> str:
    """Render the context as plain text with labelled, linked evidence."""
    lines = [f"Question: {context.question}", "", *_language_lines(context), "Texts:"]
    if not context.hits:
        lines.append("No text matched the question.")
    for source, hit in zip(text_citations(context), context.hits, strict=True):
        lines.append(f"[{source.label}] {source.title} {source.url}")
        lines += [hit.text, ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show the evidence collected for a question.")
    parser.add_argument("question", help="question in English, e.g. 'word order'")
    parser.add_argument("-l", "--language", help="language name or code, e.g. Russian or rus")
    parser.add_argument(
        "--hybrid", action="store_true", help="BM25 and embeddings (needs 'glotto-lens embed')"
    )
    args = parser.parse_args(argv)
    global EMBEDDER
    try:
        if args.hybrid:
            from glotto_lens.vectors import default_embedder

            EMBEDDER = default_embedder()
        with closing(sqlite3.connect(DB_PATH)) as conn:
            print(render_context(build_context(conn, args.question, args.language)))
    except RuntimeError as error:  # no vectors or no API key; ModelError is one too
        sys.exit(str(error))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
