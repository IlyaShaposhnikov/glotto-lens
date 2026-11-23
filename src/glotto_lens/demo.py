"""What the Streamlit demo shows, as plain data: tested without Streamlit.

The page itself is streamlit_app.py in the repository root. It asks a question through
the same pipeline as glotto-lens ask, or, without an API key, shows the evidence for an
English search query.

A public deployment (GLOTTO_LENS_PUBLIC=1) answers only with the visitor's own key, logs
no questions and can download the database once (GLOTTO_LENS_DB_URL).
"""

import gzip
import hashlib
import os
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.request import Request, urlopen

from glotto_lens import sources
from glotto_lens.answer import Answer, label_spans
from glotto_lens.context import Context, citations, text_citations, value_citations
from glotto_lens.llm import load_env

MODELS = {
    "aliceai-llm": "Alice AI LLM",
    "aliceai-llm-flash": "Alice AI LLM Flash",
    "yandexgpt-5.1": "YandexGPT 5.1 Pro",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
}
NO_REASONING = {"deepseek-v4.1-flash"}  # answered with reasoning off in the comparison
EXAMPLES = [
    "How many cases does Finnish have?",
    "Is Yoruba a tone language?",
    "Does Swahili have grammatical gender or noun classes?",
    "What is the basic word order of Irish?",
    "Does Basque have a definite article?",
]
# Markdown that could add a link, an image or a raw URL of the model's choosing; the
# remaining marks (*, _, -, #) only format, so lists and emphasis still render.
MARKDOWN = re.compile(r"([\\`\[\]()!<>:@|~$])")
WWW = re.compile(r"\b(www)\.", re.IGNORECASE)  # autolinked without "://"


def is_public() -> bool:
    """A deployment for visitors: their own key only, no question log."""
    return os.environ.get("GLOTTO_LENS_PUBLIC") == "1"


def log_enabled() -> bool:
    """Whether the demo appends questions to answers.jsonl, as glotto-lens ask does.
    Off in a public deployment, where questions come from visitors, or with GLOTTO_LENS_LOG=0."""
    return not is_public() and os.environ.get("GLOTTO_LENS_LOG", "1") != "0"


def credentials(api_key: str = "", folder_id: str = "") -> tuple[str, str] | None:
    """The key and folder to answer with: the visitor's, else (locally) those in .env.

    A public deployment never falls back to the server's key, so each visitor pays for
    their own questions.
    """
    api_key, folder_id = api_key.strip(), folder_id.strip()
    if api_key and folder_id:
        return api_key, folder_id
    if is_public():
        return None
    load_env()
    api_key = api_key or os.environ.get("YANDEX_API_KEY", "")
    folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID", "")
    return (api_key, folder_id) if api_key and folder_id else None


MAX_DOWNLOAD = 2 * 1024**3  # 2 GiB: a wrong URL cannot fill the disk of a small host
TABLES = re.compile(r"CREATE (?:VIRTUAL )?TABLE (\w+)")


def fetch_database(url: str, sha256: str, target: Path | None = None) -> Path:
    """Download the built database (a gzipped release asset or a plain file).

    sha256 is that of the downloaded file, as `sha256sum glottolens.sqlite.gz` prints it.
    It is required, and it is kept next to the database: a new asset with a new hash
    replaces the old database, the same hash is not downloaded again. A wrong, cut or
    outdated file never replaces the target, and partial files are removed.
    """
    from glotto_lens.database import SCHEMA

    sha256 = (sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Set GLOTTO_LENS_DB_SHA256 to the SHA-256 of the database asset")
    target = Path(target or sources.DB_PATH)
    stamp = target.with_name(target.name + ".sha256")
    if target.exists() and stamp.exists() and stamp.read_text().strip() == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    download = target.with_name(target.name + ".download")
    partial = target.with_name(target.name + ".part")
    try:
        digest, size = hashlib.sha256(), 0
        request = Request(url, headers={"User-Agent": sources.USER_AGENT})
        with urlopen(request, timeout=300) as response, download.open("wb") as out:
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ValueError(f"The database download is over {MAX_DOWNLOAD} bytes")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"The download from {url} has a different SHA-256")
        with download.open("rb") as head:
            gzipped = head.read(2) == b"\x1f\x8b"  # by content: a URL may hide its suffix
        if gzipped:
            with gzip.open(download, "rb") as source, partial.open("wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            download.unlink()
        else:
            download.replace(partial)
        with closing(sources.read_only(partial)) as conn:
            present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            missing = set(TABLES.findall(SCHEMA)) - present
            if missing:
                tables = ", ".join(sorted(missing))
                raise ValueError(f"The downloaded database is from another build: no {tables}")
            if not conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]:
                raise ValueError("The downloaded database has no documents")
        partial.replace(target)
        stamp.write_text(sha256)
    except BaseException:
        download.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        raise
    return target


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
    """Model text as inert Markdown: no links, images or autolinks of its own.
    The backslash is escaped too, so a segment ending in one cannot eat the next link."""
    return WWW.sub(r"\1\\.", MARKDOWN.sub(r"\\\1", text))


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
        # <…> keeps a ")" in a record id from ending the link early
        parts.append(f"[{written}](<{urls[label]}>)" if label in urls else written)
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
