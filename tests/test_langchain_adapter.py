import sqlite3
from contextlib import closing

import pytest

pytest.importorskip("langchain_core")  # the extra: pip install -e ".[langchain]"

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from glotto_lens.answer import answer_question  # noqa: E402
from glotto_lens.langchain_adapter import GlottoLensRetriever, build_chain  # noqa: E402
from glotto_lens.llm import Completion  # noqa: E402

REWRITE = '{"language": "Russian", "query": "feature distribution"}'
ANSWER = "В русском значение «yes» [V1], см. описание [T1]."


class ScriptedModel:
    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.limits = []
        self.prompts = []

    def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
        self.limits.append(max_tokens)
        self.prompts.append((system, user))
        return Completion(self.texts.pop(0), self.name)


def test_the_retriever_returns_the_evidence_as_labelled_documents(db_path):
    retriever = GlottoLensRetriever(db_path=str(db_path), language="Russian")
    documents = retriever.invoke("feature distribution")
    labels = [document.metadata["label"] for document in documents]
    assert labels[0] == "V1" and any(label.startswith("T") for label in labels)
    value = documents[0]
    assert value.metadata["kind"] == "value" and value.metadata["glottocode"] == "russ1263"
    assert value.metadata["url"].startswith("https://")
    text = next(d for d in documents if d.metadata["kind"] == "text")
    assert text.page_content and text.metadata["title"]


def test_without_a_language_there_are_only_passages(db_path):
    documents = GlottoLensRetriever(db_path=str(db_path)).invoke("feature distribution")
    assert documents and all(d.metadata["kind"] == "text" for d in documents)


def test_the_chain_gives_the_same_answer_as_the_pipeline(db_path):
    question = "Есть ли у русского <признак>?"  # tags are escaped on both paths
    pipeline_model = ScriptedModel(REWRITE, ANSWER)
    with closing(sqlite3.connect(db_path)) as conn:
        expected = answer_question(conn, pipeline_model, question)
    model = ScriptedModel(REWRITE, ANSWER)
    result = build_chain(model, db_path=str(db_path)).invoke({"question": question})
    assert result["answer"] == expected.text
    assert result["query"] == expected.rewrite.query and result["language"] == "Russian"
    assert [s["label"] for s in result["sources"]] == [c.label for c in expected.cited]
    assert model.prompts == pipeline_model.prompts  # the same messages: the same cache keys
    assert model.limits == pipeline_model.limits == [200, 1000]


def test_any_langchain_chat_model_drives_the_chain(db_path):
    llm = FakeListChatModel(responses=[REWRITE, "Да [V1] [V99]."])
    result = build_chain(llm, db_path=str(db_path)).invoke({"question": "q"})
    assert result["answer"] == "Да [V1] [V99?]."
    assert result["invalid"] == ["V99"]


def test_an_explicit_language_overrides_the_rewrite(db_path):
    model = ScriptedModel(REWRITE, "Ответ.")
    chain = build_chain(model, db_path=str(db_path))
    result = chain.invoke({"question": "q", "language": "Basque"})
    assert result["language"] == "Basque"
    assert result["context"].languoid.glottocode != "russ1263"


def test_a_cli_search_setting_does_not_leak_into_the_retriever(db_path, monkeypatch):
    # the search mode is passed down, not read from the module, so concurrent calls cannot
    # switch each other's mode; an unusable module setting would fail if it were read
    from glotto_lens import context

    monkeypatch.setattr(context, "EMBEDDER", object())
    retriever = GlottoLensRetriever(db_path=str(db_path), language="Russian")
    assert retriever.invoke("feature distribution")


def test_the_default_database_follows_the_settings(db_path, monkeypatch):
    from glotto_lens import sources

    monkeypatch.setattr(sources, "DB_PATH", db_path)
    assert GlottoLensRetriever(language="Russian").invoke("feature distribution")


def test_text_documents_have_one_url_and_the_full_list(db_path):
    documents = GlottoLensRetriever(db_path=str(db_path)).invoke("feature distribution")
    text = documents[0]
    assert " " not in (text.metadata["url"] or "")
    assert text.metadata["url"] in text.metadata["urls"]
