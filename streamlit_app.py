"""GlottoLens demo: streamlit run streamlit_app.py

Needs the database (glotto-lens build). Answers need YANDEX_API_KEY and YANDEX_FOLDER_ID
in .env; without them the page still shows the evidence for an English search query.
"""

import sqlite3
from contextlib import closing

import streamlit as st

from glotto_lens import demo, sources
from glotto_lens.answer import answer_question, log_answer
from glotto_lens.context import build_context
from glotto_lens.llm import CachedModel, ModelError, YandexModel

st.set_page_config(page_title="GlottoLens", page_icon="🔎", layout="wide")


@st.cache_resource
def model(name: str):
    effort = "none" if name in demo.NO_REASONING else None
    return CachedModel(YandexModel(name, reasoning_effort=effort))


@st.cache_resource
def embedder():
    from glotto_lens.vectors import default_embedder

    return default_embedder()


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


if not sources.DB_PATH.exists():
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
    ready = demo.hybrid_ready()
    hybrid = st.toggle(
        "Hybrid search (BM25 + embeddings)",
        value=False,
        disabled=not ready,
        help="Run `glotto-lens embed` once to enable." if not ready else "One extra API call.",
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
    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with (
                st.spinner("Searching and answering..."),
                closing(sqlite3.connect(sources.DB_PATH)) as conn,
            ):
                answer = answer_question(
                    conn,
                    model(name),
                    question,
                    language or None,
                    embedder=embedder() if hybrid else None,
                )
            log_answer(answer, name)
        except (ModelError, RuntimeError, ValueError) as error:
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
                    conn, query, language or None, embedder=embedder() if hybrid else None
                )
        except (ModelError, RuntimeError) as error:
            st.error(str(error))
            st.stop()
        show_evidence(context)
