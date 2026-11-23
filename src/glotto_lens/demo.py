"""What the Streamlit demo shows, as plain data: tested without Streamlit.

The page itself is streamlit_app.py in the repository root. It asks a question through
the same pipeline as glotto-lens ask, or, without an API key, shows the evidence for an
English search query.
"""

import os
import re
import sqlite3
from contextlib import closing

from glotto_lens import sources
from glotto_lens.answer import Answer, label_spans
from glotto_lens.context import Context, citations, text_citations, value_citations

MODELS = {
    "aliceai-llm": "Alice AI LLM",
    "aliceai-llm-flash": "Alice AI LLM Flash",
    "yandexgpt-5.1": "YandexGPT 5.1 Pro",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
}
NO_REASONING = {"deepseek-v4.1-flash"}  # answered with reasoning off in the comparison
EXAMPLES = [
    "Сколько падежей в финском языке?",
    "Есть ли тоны в языке йоруба?",
    "Does Swahili have grammatical gender or noun classes?",
    "Какой базовый порядок слов в ирландском?",
    "Есть ли в баскском определенный артикль?",
]
# Markdown that could add a link, an image or a raw URL of the model's choosing; the
# remaining marks (*, _, -, #) only format, so lists and emphasis still render.
MARKDOWN = re.compile(r"([\\`\[\]()!<>:@|~$])")


def log_enabled() -> bool:
    """Whether the demo appends questions to answers.jsonl, as glotto-lens ask does.
    Set GLOTTO_LENS_LOG=0 for a public deployment, where questions come from visitors."""
    return os.environ.get("GLOTTO_LENS_LOG", "1") != "0"


def language_line(context: Context) -> str | None:
    languoid = context.languoid
    if languoid is None:
        return None
    parts = [f"{languoid.name} ({languoid.glottocode})"]
    parts += [part for part in (languoid.family, languoid.macroarea) if part]
    return " · ".join(parts)


def value_rows(context: Context) -> list[dict]:
    """The [V…] lines as table rows."""
    return [
        {
            "label": citation.label,
            "value": citation.title,
            "coded": citation.known,
            "url": citation.url,
        }
        for citation in value_citations(context)
    ]


def text_items(context: Context) -> list[dict]:
    """The [T…] passages with their text."""
    return [
        {
            "label": citation.label,
            "title": citation.title,
            "url": hit.urls[0] if hit.urls else None,
            "text": hit.text,
        }
        for citation, hit in zip(text_citations(context), context.hits, strict=True)
    ]


def escape(text: str) -> str:
    """Model text as inert Markdown: no links, images or autolinks of its own."""
    return MARKDOWN.sub(r"\\\1", text).replace("www.", "www\\.")


def linked_answer(answer: Answer) -> str:
    """The answer in Markdown, each valid label linked to its source.

    The whole text is escaped first and only the labels become links, so the model cannot
    add a link or an image. Labels are found as the citation check finds them: [V1, T2]
    gives two links, [v 1] links to V1.
    """
    urls = {citation.label: citation.url.split()[0] for citation in answer.cited if citation.url}
    text = answer.text.strip()
    parts, position = [], 0
    for start, end, label in label_spans(text):
        parts.append(escape(text[position:start]))
        written = escape(text[start:end])
        parts.append(f"[{written}]({urls[label]})" if label in urls else written)
        position = end
    parts.append(escape(text[position:]))
    return "".join(parts)


def warnings(answer: Answer) -> list[str]:
    found = []
    if not answer.rewrite.ok:
        found.append("The rewrite could not be read; the question was searched as is.")
    if not answer.complete:
        found.append("The model stopped early; the answer may be cut.")
    if answer.invalid_labels:
        found.append("Labels not found in the evidence: " + ", ".join(answer.invalid_labels))
    if not answer.cited and citations(answer.context):  # as render_answer decides
        found.append("The answer cites no evidence.")
    return found


def hybrid_ready(db_path=None) -> bool:
    """Whether the database has vectors for every feature document (glotto-lens embed)."""
    from glotto_lens.vectors import YandexEmbedder, coverage

    try:
        with closing(sources.read_only(db_path)) as conn:
            counts = coverage(conn, YandexEmbedder.name)
    except sqlite3.Error:
        return False
    return counts["feature_documents"] > 0 and counts["vectors"] == counts["feature_documents"]
