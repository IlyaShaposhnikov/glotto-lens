"""GlottoLens demo: streamlit run streamlit_app.py

Needs the database (glotto-lens build). Answers need YANDEX_API_KEY and YANDEX_FOLDER_ID
in .env; without them the page still shows the evidence for an English search query.
Questions are logged to answers.jsonl unless GLOTTO_LENS_LOG=0.

Public deployment (Streamlit secrets become environment variables):
GLOTTO_LENS_PUBLIC=1 makes visitors answer with their own key and logs nothing;
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
try:  # root-level secrets as environment variables, whether or not Streamlit set them yet
    for secret_name, secret_value in st.secrets.items():
        if isinstance(secret_value, str):
            os.environ.setdefault(secret_name, secret_value)
except Exception:  # noqa: BLE001, S110 - no secrets file when run locally
    pass
ERRORS = (ModelError, RuntimeError, ValueError)  # ValueError: e.g. a bad YANDEX_TIMEOUT


@st.cache_resource(show_spinner="Downloading the database (first start only)...")
def database_ready() -> bool:
    url = os.environ.get("GLOTTO_LENS_DB_URL")
    if url and not sources.DB_PATH.exists():
        demo.fetch_database(url, os.environ.get("GLOTTO_LENS_DB_SHA256"))
    return sources.DB_PATH.exists()


# Built per request, never cached across sessions: they hold the visitor's key.
def model(name: str, key: tuple[str, str]):
    effort = "none" if name in demo.NO_REASONING else None
    api_key, folder_id = key
    return CachedModel(
        YandexModel(name, api_key=api_key, folder_id=folder_id, reasoning_effort=effort)
    )


def embedder(key: tuple[str, str]):
    from glotto_lens.vectors import CachedEmbedder, YandexEmbedder

    return CachedEmbedder(YandexEmbedder(*key))


def show_evidence(context) -> None:
    line = demo.language_line(context)
    if line:
        st.caption(f"Language: {line}")
    rows = demo.value_rows(context)
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("source")},
        )
    elif context.languoid is None:
        st.info("No language recognised, so no values: only passages.")
    for item in demo.text_items(context):
        with st.expander(f"[{item['label']}] {item['title']}"):
            st.write(item["text"])
            if item["url"]:
                st.markdown(f"[{item['url']}]({item['url']})")


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
    mode = st.radio(
        "Mode",
        ["Answer (needs an API key)", "Evidence only (no API key)"],
        help="Evidence only searches with an English query and calls no model.",
    )
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
        disabled=not ready or key is None,
        help="Run `glotto-lens embed` once to enable."
        if not ready
        else "Embeds the query: one extra API call, so it needs a key in both modes.",
    )
    language = st.text_input("Language (optional)", help="Overrides the language in the question.")
    st.markdown(
        "Data: [Grambank](https://grambank.clld.org) · [WALS](https://wals.info) · "
        "[Glottolog](https://glottolog.org), CC BY 4.0"
    )

st.title("GlottoLens")
st.write(
    "Questions about the grammar of the world's languages, answered from WALS, Grambank and "
    "Glottolog with a citation for every claim."
)

if mode.startswith("Answer"):
    example = st.selectbox("Examples", ["—", *demo.EXAMPLES])
    question = st.text_input(
        "Question", value="" if example == "—" else example, placeholder=demo.EXAMPLES[0]
    )
    if key is None:
        st.info("Enter your Yandex AI Studio API key and folder ID in the sidebar to ask.")
    if st.button("Ask", type="primary", disabled=not question.strip() or key is None):
        try:
            with (
                st.spinner("Searching and answering..."),
                closing(sqlite3.connect(sources.DB_PATH)) as conn,
            ):
                answer = answer_question(
                    conn,
                    model(name, key),
                    question,
                    language or None,
                    embedder=embedder(key) if hybrid else None,
                )
            if demo.log_enabled():
                log_answer(answer, name)
        except ERRORS as error:
            st.error(str(error))
            st.stop()
        st.caption(f"Search: “{answer.rewrite.query}”, language: {answer.rewrite.language}")
        st.markdown(demo.linked_answer(answer))
        for warning in demo.warnings(answer):
            st.warning(warning)
        input_tokens, output_tokens = answer.tokens
        cached = sum(c.cached for c in answer.completions)
        st.caption(
            f"{demo.MODELS[name]} · {input_tokens} tokens in, {output_tokens} out · "
            f"{cached} of {len(answer.completions)} calls from cache"
        )
        st.subheader("Evidence")
        show_evidence(answer.context)
else:
    query = st.text_input("English search query", value="number of cases")
    if query.strip():
        try:
            with closing(sqlite3.connect(sources.DB_PATH)) as conn:
                context = build_context(
                    conn, query, language or None, embedder=embedder(key) if hybrid else None
                )
        except ERRORS as error:
            st.error(str(error))
            st.stop()
        show_evidence(context)
