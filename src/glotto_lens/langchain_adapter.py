"""GlottoLens as LangChain components: a retriever and an LCEL chain.

The pipeline itself does not depend on LangChain (see answer.py). This module exposes the
same steps through LangChain interfaces, so GlottoLens can be dropped into a LangChain
application or driven by any LangChain chat model:

- GlottoLensRetriever is a BaseRetriever: an English search query in, the evidence out as
  Documents, one per [V…] value line and one per [T…] passage, with labels and URLs in the
  metadata.
- build_chain(llm) is the whole pipeline as an LCEL chain: rewrite → retrieve → answer →
  citation check. llm is a GlottoLens model (YandexModel, CachedModel), used with the
  same token limits as glotto-lens ask, or any LangChain chat model or runnable.

Install with: pip install -e ".[langchain]"

    from glotto_lens.langchain_adapter import build_chain
    from glotto_lens.llm import CachedModel, YandexModel

    chain = build_chain(CachedModel(YandexModel()))
    result = chain.invoke({"question": "Сколько падежей в финском языке?"})
    print(result["answer"], result["sources"])

The chain sends the same prompts with the same token limits as glotto-lens ask (a test
compares them), so with a CachedModel the two share the cache. It returns the answer, its
sources and the invalid labels; unlike ask it writes no answers.jsonl line and does not
warn when the model stopped early.
"""

import sqlite3
from contextlib import closing

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from glotto_lens import prompts, sources
from glotto_lens.answer import (
    ANSWER_MAX_TOKENS,
    REWRITE_MAX_TOKENS,
    answer_prompt,
    check_citations,
    mark_invalid,
    parse_rewrite,
    question_language,
)
from glotto_lens.context import (
    Context,
    build_context,
    text_citations,
    value_citations,
)
from glotto_lens.llm import ChatModel


def context_documents(context: Context) -> list[Document]:
    """The evidence of a context as Documents, in the order the model sees it."""
    language = context.languoid.glottocode if context.languoid else None
    documents = [
        Document(
            page_content=citation.title,
            metadata={
                "label": citation.label,
                "kind": "value",
                "url": citation.url,
                "urls": [citation.url],
                "feature_id": citation.feature_id,
                "value": citation.value,
                "known": citation.known,
                "glottocode": language,
            },
        )
        for citation in value_citations(context)
    ]
    for citation, hit in zip(text_citations(context), context.hits, strict=True):
        documents.append(
            Document(
                page_content=hit.text,
                metadata={
                    "label": citation.label,
                    "kind": "text",
                    "title": citation.title,
                    "url": hit.urls[0] if hit.urls else None,
                    "urls": list(hit.urls),
                    "feature_id": hit.feature_id,
                    "glottocode": language,
                },
            )
        )
    return documents


def collect_context(
    question: str, language: str | None, db_path=None, hybrid: bool = False
) -> Context:
    """build_context with its own connection. The embedder is passed, not set globally, so
    concurrent calls (LangChain batches, async) cannot switch each other's search mode."""
    embedder = None
    if hybrid:
        from glotto_lens.vectors import default_embedder

        embedder = default_embedder()
    with closing(sqlite3.connect(db_path or sources.DB_PATH)) as conn:
        if embedder is not None:
            from glotto_lens.vectors import check_ready

            check_ready(conn, embedder.name)
        return build_context(conn, question, language, embedder=embedder)


class GlottoLensRetriever(BaseRetriever):
    """Evidence for an English search query in WALS/Grambank terms ("number of cases").

    With a language, the documents include its recorded values; without one, only
    passages. The query is not rewritten here: build_chain does that step.
    """

    db_path: str | None = None  # None: sources.DB_PATH at the time of the call
    language: str | None = None
    hybrid: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return context_documents(collect_context(query, self.language, self.db_path, self.hybrid))


def as_runnable(model: ChatModel, max_tokens: int = ANSWER_MAX_TOKENS) -> Runnable:
    """A GlottoLens model (YandexModel, CachedModel) as a runnable over chat messages."""

    def call(messages) -> str:
        if hasattr(messages, "to_messages"):  # a ChatPromptValue
            messages = messages.to_messages()
        system = "\n".join(m.content for m in messages if isinstance(m, SystemMessage))
        user = "\n".join(m.content for m in messages if not isinstance(m, SystemMessage))
        return model.complete(system, user, max_tokens=max_tokens).text

    return RunnableLambda(call, name=f"glotto-lens:{model.name}")


def _rewrite_messages(inputs: dict) -> list[BaseMessage]:
    question = prompts.escape_tags(inputs["question"])
    return [
        SystemMessage(prompts.REWRITE_SYSTEM),
        HumanMessage(prompts.REWRITE_USER.format(question=question)),
    ]


def _answer_messages(inputs: dict) -> list[BaseMessage]:
    user = answer_prompt(inputs["question"], inputs["context"], inputs.get("question_language"))
    return [SystemMessage(prompts.ANSWER_SYSTEM), HumanMessage(user)]


def _runnable(model, max_tokens: int) -> Runnable:
    """A GlottoLens model with the given token limit, or a LangChain runnable as it is."""
    return as_runnable(model, max_tokens) if hasattr(model, "complete") else model


def _flatten(key: str) -> Runnable:
    """Move the fields of inputs[key] up into the inputs."""
    return RunnableLambda(
        lambda inputs: {**{k: v for k, v in inputs.items() if k != key}, **inputs[key]}
    )


def build_chain(
    llm,
    rewrite_llm=None,
    db_path: str | None = None,
    hybrid: bool = False,
) -> Runnable:
    """The GlottoLens pipeline as an LCEL chain.

    Input: {"question": ..., "language": optional}. Output: the question, the rewritten
    query and language, the context, the answer text (invented labels marked with "?"),
    the cited sources and the invalid labels.
    """
    if hybrid:  # fail now, before the chain pays for a rewrite
        from glotto_lens.vectors import check_ready, default_embedder

        with closing(sqlite3.connect(db_path or sources.DB_PATH)) as conn:
            check_ready(conn, default_embedder().name)
    to_text = StrOutputParser()
    answer_llm = _runnable(llm, ANSWER_MAX_TOKENS)
    rewrite_llm = _runnable(rewrite_llm or llm, REWRITE_MAX_TOKENS)

    def parse(inputs: dict) -> dict:
        rewrite = parse_rewrite(inputs["rewrite_text"], inputs["question"])
        language = inputs.get("language") or rewrite.language
        return {
            "query": rewrite.query,
            "language": language,
            "question_language": question_language(inputs["question"], rewrite),
            "rewrite_ok": rewrite.ok,
        }

    def retrieve(inputs: dict) -> Context:
        return collect_context(inputs["query"], inputs["language"], db_path, hybrid)

    def check(inputs: dict) -> dict:
        cited, invalid = check_citations(inputs["answer_text"], inputs["context"])
        return {
            "answer": mark_invalid(inputs["answer_text"], invalid),
            "sources": [{"label": c.label, "title": c.title, "url": c.url} for c in cited],
            "invalid": invalid,
        }

    return (
        RunnablePassthrough.assign(
            rewrite_text=RunnableLambda(_rewrite_messages) | rewrite_llm | to_text
        )
        | RunnablePassthrough.assign(parsed=RunnableLambda(parse))
        | _flatten("parsed")
        | RunnablePassthrough.assign(context=RunnableLambda(retrieve))
        | RunnablePassthrough.assign(
            answer_text=RunnableLambda(_answer_messages) | answer_llm | to_text
        )
        | RunnablePassthrough.assign(checked=RunnableLambda(check))
        | _flatten("checked")
    ).with_config(run_name="glotto-lens")
