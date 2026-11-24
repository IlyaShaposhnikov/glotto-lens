"""GlottoLens demo: streamlit run streamlit_app.py

Needs the database (glotto-lens build). Answers need YANDEX_API_KEY and YANDEX_FOLDER_ID
in .env or the visitor's own key; without a key the page still shows the evidence for an
English search query. Questions are logged to answers.jsonl unless GLOTTO_LENS_LOG=0.

Public deployment (root-level Streamlit secrets become environment variables):
GLOTTO_LENS_PUBLIC=1 makes visitors answer with their own key and logs no questions;
GLOTTO_LENS_DB_URL and GLOTTO_LENS_DB_SHA256 download the built database on first start.
"""

import os
import sqlite3
from contextlib import closing

import streamlit as st

from glotto_lens import demo, sources
from glotto_lens.answer import answer_question, log_answer
from glotto_lens.context import build_context
from glotto_lens.llm import CachedModel, ModelError, YandexModel

st.set_page_config(page_title="GlottoLens", page_icon="🔎", layout="wide")
ERRORS = (ModelError, RuntimeError, ValueError)  # ValueError: e.g. a bad YANDEX_TIMEOUT
NO_SECRETS = (
    FileNotFoundError,
    getattr(st.errors, "StreamlitSecretNotFoundError", FileNotFoundError),
)
OUR_SECRETS = ("GLOTTO_LENS_", "YANDEX_")

try:  # our root-level secrets as environment variables, whether or not Streamlit set them
    secrets = dict(st.secrets)
except NO_SECRETS:  # no secrets file when run locally; a broken one still shows its error
    secrets = {}
for secret_name, secret_value in secrets.items():
    if isinstance(secret_value, str) and secret_name.startswith(OUR_SECRETS):
        os.environ.setdefault(secret_name, secret_value)


@st.cache_resource(show_spinner="Downloading the database (first start only)...")
def download_database(url: str, sha256: str) -> None:
    """Cached per (url, hash): only a success is kept, a failure is retried next run."""
    demo.fetch_database(url, sha256)


def database_ready() -> bool:
    url = os.environ.get("GLOTTO_LENS_DB_URL")
    if url:
        download_database(url, os.environ.get("GLOTTO_LENS_DB_SHA256", ""))
    return sources.DB_PATH.exists()  # not cached: a database built meanwhile is found


# Built per request, never cached across sessions: they hold the visitor's key.
def model(name: str, key: tuple[str, str]):
    effort = "none" if name in demo.NO_REASONING else None
    return CachedModel(YandexModel(name, api_key=key[0], folder_id=key[1], reasoning_effort=effort))


def embedder(key: tuple[str, str]):
    from glotto_lens.vectors import CachedEmbedder, YandexEmbedder

    return CachedEmbedder(YandexEmbedder(api_key=key[0], folder_id=key[1]))


def close(*wrappers) -> None:
    """Close the HTTP clients of per-request models, so connections do not pile up."""
    for wrapper in wrappers:
        if wrapper is not None:
            wrapper.close()


def show_evidence(context) -> None:
    line = demo.language_line(context)
    if line:
        st.caption(f"Language: {line}")
    rows = demo.value_rows(context)
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("source"),
                "coded": st.column_config.CheckboxColumn("coded"),
            },
        )
    elif context.languoid is None:
        st.info("No language recognised, so no values: only passages.")
    for item in demo.text_items(context):
        with st.expander(f"[{item['label']}] {item['title']}"):
            st.write(item["text"])
            if item["url"]:
                st.markdown(f"[{item['url']}]({item['url']})")


def show_answer(answer, name: str) -> None:
    st.caption(f"Search: “{answer.rewrite.query}”, language: {answer.rewrite.language}")
    st.markdown(demo.linked_answer(answer))
    for warning in demo.warnings(answer):
        st.warning(warning)
    input_tokens, output_tokens = answer.tokens
    cached = sum(c.cached for c in answer.completions)
    st.caption(
        f"{demo.MODELS.get(name, name)} · {input_tokens} tokens in, {output_tokens} out · "
        f"{cached} of {len(answer.completions)} calls from cache"
    )
    st.subheader("Evidence")
    show_evidence(answer.context)


def use_example() -> None:
    if st.session_state.example:
        st.session_state.question = st.session_state.example


EVIDENCE_EXAMPLES = {demo.evidence_label(example): example for example in demo.EVIDENCE_EXAMPLES}


def use_evidence_example() -> None:
    if st.session_state.evidence_example:
        query, language = EVIDENCE_EXAMPLES[st.session_state.evidence_example]
        st.session_state.evidence_query = query
        st.session_state.evidence_language = language
        st.session_state.run_evidence = True


try:
    ready_db = database_ready()
except Exception as error:  # noqa: BLE001 - show why the download failed
    st.error(f"The database could not be downloaded: {error}")
    st.stop()
if not ready_db:
    st.error("No database yet: run `glotto-lens build` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    name = st.selectbox("Answer model", list(demo.MODELS), format_func=demo.MODELS.get)
    with st.expander("Your Yandex AI Studio key", expanded=demo.is_public()):
        api_key = st.text_input("API key", type="password")
        folder_id = st.text_input("Folder ID")
        st.caption(
            "Used only for this session's requests to Yandex AI Studio and never stored."
            + ("" if demo.is_public() else " Empty: the key from .env is used.")
        )
    key = demo.credentials(api_key, folder_id)
    ready = demo.hybrid_ready()
    hybrid = st.toggle(
        "Hybrid search (BM25 + embeddings)",
        value=False,
        disabled=not ready,
        help="Run `glotto-lens embed` once to enable."
        if not ready
        else "Embeds the query: one extra API call, so it needs a key on both tabs. "
        "On the evidence tab it searches again at once.",
    )
    st.markdown(
        "Data: [Grambank](https://grambank.clld.org) · [WALS](https://wals.info) · "
        "[Glottolog](https://glottolog.org), CC BY 4.0"
    )

st.title("GlottoLens")
st.write(
    "Questions about the grammar of the world's languages, answered from WALS, Grambank and "
    "Glottolog with a citation for every claim."
)
st.caption(
    "Ask in English or in another language: the answer comes in the language of the question "
    "(checked in English, Russian, Spanish, German, French, Portuguese, Italian, Chinese, "
    "Japanese and Korean)."
)

ask_tab, evidence_tab = st.tabs(["Ask a question", "Browse evidence (no key needed)"])

with ask_tab:
    st.session_state.setdefault("question", "")
    st.pills("Examples", demo.EXAMPLES, key="example", on_change=use_example)
    with st.form("ask_form"):
        question = st.text_input("Question", key="question", placeholder=demo.EXAMPLES[0])
        ask_language = st.text_input(
            "Language to look up (optional)",
            help="The language the facts are about; overrides the one named in the question. "
            "The answer is written in the language of the question.",
        )
        asked = st.form_submit_button("Ask", type="primary")
    if asked:
        if not question.strip():
            st.warning("Type a question or pick an example.")
        elif key is None:
            st.warning("Enter your Yandex AI Studio API key and folder ID in the sidebar.")
        else:
            answer_model = search = None
            try:  # building the models reads YANDEX_TIMEOUT, which can be invalid
                answer_model = model(name, key)
                search = embedder(key) if hybrid else None
                with (
                    st.spinner("Searching and answering..."),
                    closing(sqlite3.connect(sources.DB_PATH)) as conn,
                ):
                    answer = answer_question(
                        conn, answer_model, question, ask_language or None, embedder=search
                    )
                if demo.log_enabled():
                    log_answer(answer, name)
                st.session_state.answer_result = (answer, name)
            except ERRORS as error:
                st.error(str(error))
            finally:
                close(answer_model, search)
    if "answer_result" in st.session_state:  # kept when other widgets rerun the page
        show_answer(*st.session_state.answer_result)

with evidence_tab:
    st.session_state.setdefault("evidence_query", demo.EVIDENCE_EXAMPLES[0][0])
    st.session_state.setdefault("evidence_language", demo.EVIDENCE_EXAMPLES[0][1])
    st.pills(
        "Examples",
        list(EVIDENCE_EXAMPLES),
        key="evidence_example",
        on_change=use_evidence_example,
    )
    with st.form("evidence_form"):
        query = st.text_input("English search query", key="evidence_query")
        evidence_language = st.text_input("Language to look up (optional)", key="evidence_language")
        searched = st.form_submit_button("Search", type="primary")
    use_vectors = hybrid and key is not None
    if hybrid and key is None:
        st.info("Hybrid search needs a key: searching with BM25 only.")
    previous = st.session_state.get("evidence_result")
    picked = st.session_state.pop("run_evidence", False)  # popped on every run
    rerun = (
        searched
        or picked
        or previous is None  # the first visit shows an example
        or previous[1] != use_vectors  # the hybrid toggle changed
    )
    if rerun and not query.strip():
        st.warning("Type a search query in English.")
    elif rerun:
        search = None
        try:
            search = embedder(key) if use_vectors else None
            with closing(sqlite3.connect(sources.DB_PATH)) as conn:
                context = build_context(conn, query, evidence_language or None, embedder=search)
            st.session_state.evidence_result = (context, use_vectors)
        except ERRORS as error:
            st.error(str(error))
        finally:
            close(search)
    if "evidence_result" in st.session_state:
        show_evidence(st.session_state.evidence_result[0])
