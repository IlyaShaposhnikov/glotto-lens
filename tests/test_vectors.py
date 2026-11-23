import sqlite3
from contextlib import closing

import numpy as np
import pytest

from glotto_lens import vectors
from glotto_lens.database import SCHEMA
from glotto_lens.search import search
from glotto_lens.vectors import (
    CachedEmbedder,
    embed_documents,
    hybrid_features,
    reciprocal_rank_fusion,
    vector_ranking,
)

# feature documents: (id, feature, title, section, text)
DOCUMENTS = [
    (1, "wals:30A", "Number of Genders", "Summary", "Languages with two, three or more genders."),
    (2, "grambank:GB051", "Gender by sex", "Summary", "Is there a gender system based on sex?"),
    (3, "wals:1A", "Consonant Inventories", "Summary", "Small and large consonant inventories."),
    (4, "wals:1A", "Consonant Inventories", "Values", "Moderately large: 26 to 33 consonants."),
]
# the meaning of each text as a direction in the vector space: what a real embedding
# model would see, and BM25 would not ("noun classes" shares no word with "genders")
TOPICS = {"gender": 0, "class": 0, "sex": 1, "consonant": 2}


class TopicEmbedder:
    name = "topics"

    def __init__(self):
        self.calls = []

    def embed(self, texts, kind):
        self.calls.append((kind, len(texts)))
        result = []
        for text in texts:
            vector = np.zeros(vectors.DIMENSIONS)
            for word, axis in TOPICS.items():
                if word in text.lower():
                    vector[axis] += 1
            vector[255] = 0.01  # never a zero vector
            result.append(vector.tolist())
        return result


@pytest.fixture
def conn():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO documents (id, kind, title, section, text, position, feature_id)"
            " VALUES (?, 'feature', ?, ?, ?, 0, ?)",
            [(i, title, section, text, feature) for i, feature, title, section, text in DOCUMENTS],
        )
        connection.execute("INSERT INTO documents_fts (documents_fts) VALUES ('rebuild')")
        yield connection


def test_rrf_rewards_agreement_and_keeps_first_seen_order_on_ties():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert [key for key, _ in fused] == ["a", "c", "b"]
    assert reciprocal_rank_fusion([["x"], ["y"]])[0][0] == "x"


def test_documents_are_embedded_once_and_resumable(conn):
    embedder = TopicEmbedder()
    assert embed_documents(conn, embedder, batch=3) == 4
    assert embedder.calls == [("doc", 3), ("doc", 1)]
    assert embed_documents(conn, embedder) == 0  # nothing left to do
    stored = conn.execute("SELECT COUNT(*) FROM document_vectors").fetchone()[0]
    assert stored == 4


def test_vector_ranking_keeps_one_document_per_feature(conn):
    embed_documents(conn, TopicEmbedder())
    query = TopicEmbedder().embed(["consonant"], "query")[0]
    ranking = vector_ranking(conn, query, "topics", limit=10)
    assert ranking[0][1] == "wals:1A"
    assert [feature for _, feature, _ in ranking].count("wals:1A") == 1


def test_hybrid_finds_a_feature_that_shares_no_word_with_the_query(conn):
    embed_documents(conn, TopicEmbedder())
    question = "noun classes"
    assert search(conn, question, 5, per_feature=True) == []
    hits = hybrid_features(conn, question, 3, TopicEmbedder())
    assert hits[0].feature_id == "wals:30A"
    assert hits[0].snippet == DOCUMENTS[0][4]  # found by vector only: its text is the snippet
    assert all(hit.score > 0 for hit in hits)


def test_hybrid_needs_embedded_documents(conn):
    with pytest.raises(RuntimeError, match="glotto-lens embed"):
        hybrid_features(conn, "genders", 3, TopicEmbedder())


def test_cache_answers_repeated_texts_without_calling_the_embedder(tmp_path):
    inner = TopicEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    first = cached.embed(["gender", "consonant"], "doc")
    second = cached.embed(["consonant", "gender"], "doc")
    assert inner.calls == [("doc", 2)]
    assert second == [first[1], first[0]]
    cached.embed(["gender"], "query")  # a query vector is a different model
    assert inner.calls[-1] == ("query", 1)


def test_hybrid_settings_are_recorded_only_for_hybrid_runs(monkeypatch):
    from glotto_lens import context
    from glotto_lens.evaluate import retrieval_settings

    monkeypatch.setattr(context, "EMBEDDER", TopicEmbedder())  # the global is not read
    assert "hybrid" not in retrieval_settings()
    assert retrieval_settings(TopicEmbedder())["hybrid"]["embedder"] == "topics"


def test_context_uses_the_hybrid_pool_when_an_embedder_is_set(conn, monkeypatch):
    from glotto_lens import context

    embed_documents(conn, TopicEmbedder())
    conn.executemany(
        "INSERT INTO features (id, source, local_id, name) VALUES (?, ?, ?, ?)",
        [
            ("wals:30A", "wals", "30A", "Number of Genders"),
            ("grambank:GB051", "grambank", "GB051", "Gender by sex"),
            ("wals:1A", "wals", "1A", "Consonant Inventories"),
        ],
    )
    monkeypatch.setattr(context, "EMBEDDER", None)
    assert context.build_context(conn, "noun classes").evidence == []
    monkeypatch.setattr(context, "EMBEDDER", TopicEmbedder())
    found = [item.feature_id for item in context.build_context(conn, "noun classes").evidence]
    assert found[0] == "wals:30A"


def test_no_query_call_for_an_empty_limit_and_no_silent_cap(conn, monkeypatch):
    embed_documents(conn, TopicEmbedder())
    embedder = TopicEmbedder()
    assert hybrid_features(conn, "genders", 0, embedder) == []
    assert embedder.calls == []
    monkeypatch.setattr(vectors, "CANDIDATES", 1)  # fewer than asked: the limit wins
    assert len(hybrid_features(conn, "genders consonant sex", 3, embedder)) == 3


def test_in_hybrid_mode_the_pool_alone_orders_the_features(conn, monkeypatch):
    # BM25 scores (units) and fused scores (about 1/60) must not be sorted together:
    # the general search would put its own feature documents first
    from glotto_lens import context
    from glotto_lens.search import document_hit

    conn.executemany(
        "INSERT INTO features (id, source, local_id, name) VALUES (?, ?, ?, ?)",
        [
            ("wals:30A", "wals", "30A", "Number of Genders"),
            ("grambank:GB051", "grambank", "GB051", "Gender by sex"),
            ("wals:1A", "wals", "1A", "Consonant Inventories"),
        ],
    )
    monkeypatch.setattr(context, "EMBEDDER", TopicEmbedder())
    monkeypatch.setattr(
        vectors, "hybrid_features", lambda conn, q, limit, e: [document_hit(conn, 3, 0.03)]
    )
    found = [item.feature_id for item in context.build_context(conn, "sex gender").evidence]
    assert found == ["wals:1A"]


def test_reserved_slots_keep_bm25_first_choices_and_add_vector_only_features():
    keyword = ["wals:2A", "wals:10A", "GB1", "GB2"]
    semantic = ["GB1", "wals:30A", "wals:10A", "GB9"]
    # BM25's first two stay; the two places left go to vector features BM25 did not bring
    assert vectors.reserved_slots(keyword, semantic, 4, 2) == [
        "wals:2A",
        "wals:10A",
        "GB1",
        "wals:30A",
    ]
    # short rankings: free places are filled from both, nothing repeats
    assert vectors.reserved_slots(["a"], ["a", "b"], 4, 2) == ["a", "b"]
    assert vectors.reserved_slots(["a", "b"], ["c"], 1, 2) == ["c"]


def test_rrf_fusion_is_still_available(conn, monkeypatch):
    embed_documents(conn, TopicEmbedder())
    monkeypatch.setattr(vectors, "FUSION", "rrf")
    hits = hybrid_features(conn, "noun classes", 3, TopicEmbedder())
    assert hits[0].feature_id == "wals:30A"
    assert vectors.settings()["rrf_k"] == vectors.RRF_K


def test_a_partly_embedded_database_is_caught_before_any_paid_call(conn):
    with pytest.raises(RuntimeError, match="0 of 4 feature documents"):
        vectors.check_ready(conn, "topics")
    embed_documents(conn, TopicEmbedder())
    vectors.check_ready(conn, "topics")
    assert vectors.coverage(conn, "topics") == {"feature_documents": 4, "vectors": 4}


def test_damaged_vectors_and_missing_documents_are_clear_errors(conn):
    from glotto_lens.search import document_hit

    embed_documents(conn, TopicEmbedder())
    conn.execute("UPDATE document_vectors SET vector = x'00' WHERE document_id = 1")
    with pytest.raises(RuntimeError, match="not 256 floats"):
        vectors.stored_vectors(conn, "topics")
    with pytest.raises(RuntimeError, match="no text"):
        document_hit(conn, 99, 1.0)


def test_damaged_cache_files_and_stale_vectors_count_as_missing(conn, tmp_path):
    inner = TopicEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    cached.embed(["gender"], "doc")
    cached._path("gender", "doc").write_text("[1.0, 2.0]")  # wrong size
    cached.embed(["gender"], "doc")
    assert len(inner.calls) == 2
    embed_documents(conn, TopicEmbedder())
    conn.execute("DELETE FROM documents WHERE id = 4")
    conn.execute(
        "INSERT INTO documents (id, kind, title, section, text, position, feature_id)"
        " VALUES (5, 'feature', 't', 's', 'new text', 0, 'wals:1A')"
    )
    with pytest.raises(RuntimeError, match="3 of 4 feature documents"):
        vectors.check_ready(conn, "topics")


def test_vector_features_rank_by_cosine_one_document_per_feature(conn):
    from glotto_lens.vectors import vector_features

    embed_documents(conn, TopicEmbedder())
    embedder = TopicEmbedder()
    assert vector_features(conn, "consonant", 0, embedder) == []
    assert embedder.calls == []
    hits = vector_features(conn, "consonant", 10, embedder)
    assert hits[0].feature_id == "wals:1A" and 0.9 < hits[0].score < 1.001
    assert [hit.feature_id for hit in hits].count("wals:1A") == 1


def test_a_fresh_vector_of_the_wrong_size_is_not_cached(tmp_path):
    from glotto_lens.llm import ModelError

    class Short:
        name = "short"

        def embed(self, texts, kind):
            return [[1.0, 2.0] for _ in texts]

    with pytest.raises(ModelError, match="not 256"):
        CachedEmbedder(Short(), tmp_path).embed(["x"], "doc")
    assert not list(tmp_path.glob("*.json"))


def test_embed_command_reports_coverage(db_path, monkeypatch, capsys):
    monkeypatch.setattr(vectors, "DB_PATH", db_path)
    vectors.main([], embedder=TopicEmbedder())
    out = capsys.readouterr().out
    assert "feature documents embedded" in out
