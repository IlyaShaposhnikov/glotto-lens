import sqlite3
from contextlib import closing

import pytest

from glotto_lens import demo
from glotto_lens.answer import answer_question
from glotto_lens.context import build_context
from glotto_lens.llm import Completion


class ScriptedModel:
    name = "scripted"

    def __init__(self, *texts):
        self.texts = list(texts)

    def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
        return Completion(self.texts.pop(0), self.name)


@pytest.fixture
def conn(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        yield connection


def test_evidence_becomes_rows_and_passages(conn):
    context = build_context(conn, "feature distribution", "Russian", embedder=None)
    assert "russ1263" in demo.language_line(context)
    rows = demo.value_rows(context)
    assert rows[0]["label"] == "V1" and rows[0]["url"].startswith("https://")
    items = demo.text_items(context)
    assert items and items[0]["label"] == "T1" and items[0]["text"]


def test_without_a_language_there_is_no_language_line(conn):
    context = build_context(conn, "feature distribution", embedder=None)
    assert demo.language_line(context) is None and demo.value_rows(context) == []


def test_valid_labels_become_links_and_invented_ones_stay_marked(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1], но [V99].")
    answer = answer_question(conn, model, "q", embedder=None)
    text = demo.linked_answer(answer)
    assert "\\[[V1](<https://" in text
    assert "\\[V99?\\]" in text
    assert any("V99" in warning for warning in demo.warnings(answer))


def test_hybrid_is_not_offered_without_vectors(db_path, tmp_path):
    assert demo.hybrid_ready(db_path) is False
    assert demo.hybrid_ready(tmp_path / "missing.sqlite") is False


def test_the_demo_models_are_in_the_comparison():
    assert "aliceai-llm" in demo.MODELS and demo.NO_REASONING <= set(demo.MODELS)


def test_every_label_the_citation_check_accepts_is_linked(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1, T1], [v 1].")
    answer = answer_question(conn, model, "q", embedder=None)
    text = demo.linked_answer(answer)
    assert "[V1](<https://" in text and "[T1](<https://" in text and "[v 1](<https://" in text


def test_links_and_images_written_by_the_model_stay_text(conn):
    reply = "Да [V1]. [x](http://evil.example) ![](http://evil.example/p.png) www.evil.example"
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', reply)
    text = demo.linked_answer(answer_question(conn, model, "q", embedder=None))
    assert "](http://evil" not in text and "http://evil" not in text
    assert "www.evil" not in text and "\\[x\\]" in text


def test_an_answer_with_only_passages_uncited_is_flagged(conn):
    model = ScriptedModel('{"language": null, "query": "feature distribution"}', "Нет ссылок.")
    answer = answer_question(conn, model, "q", embedder=None)
    assert answer.context.languoid is None and answer.context.hits
    assert "The answer cites no evidence." in demo.warnings(answer)


@pytest.mark.parametrize("folder", ["plain", "with space", "odd#name%20"])
def test_hybrid_check_opens_absolute_paths(db_path, tmp_path, folder):
    import shutil

    target = tmp_path / folder / "glottolens.sqlite"
    target.parent.mkdir()
    shutil.copy(db_path, target)
    with closing(sqlite3.connect(target)) as connection:  # the database itself opens
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] > 0
    assert demo.hybrid_ready(target.resolve()) is False  # no vectors, but no error either
    from glotto_lens.sources import read_only

    with closing(read_only(target.resolve())) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] > 0


def test_logging_can_be_turned_off(monkeypatch):
    monkeypatch.delenv("GLOTTO_LENS_LOG", raising=False)
    assert demo.log_enabled()
    monkeypatch.setenv("GLOTTO_LENS_LOG", "0")
    assert not demo.log_enabled()


def test_a_backslash_before_a_label_does_not_break_its_link(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Путь C:\\[V1].")
    text = demo.linked_answer(answer_question(conn, model, "q", embedder=None))
    assert "\\\\\\[[V1](<https://" in text  # escaped backslash, escaped bracket, then link


def test_uppercase_www_and_brackets_in_urls_are_safe():
    assert "WWW\\.EVIL" in demo.escape("see WWW.EVIL.EXAMPLE")
    assert "www\\.x" in demo.escape("www.x")


def test_label_links_keep_a_closing_parenthesis_in_the_url(conn):
    model = ScriptedModel('{"language": "Russian", "query": "feature"}', "Да [V1].")
    answer = answer_question(conn, model, "q", embedder=None)
    cited = answer.cited[0]
    answer.cited[0] = type(cited)(**{**vars(cited), "url": "https://example.org/a(b)"})
    assert "[V1](<https://example.org/a(b)>)" in demo.linked_answer(answer)


def test_public_demo_uses_only_the_visitors_key(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "server")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder")
    monkeypatch.delenv("GLOTTO_LENS_PUBLIC", raising=False)
    assert demo.credentials() == ("server", "folder")
    assert demo.credentials(" mine ", " f ") == ("mine", "f")
    monkeypatch.setenv("GLOTTO_LENS_PUBLIC", "1")
    assert demo.credentials() is None and demo.credentials("mine", "") is None
    assert demo.credentials("mine", "f") == ("mine", "f")
    assert not demo.log_enabled()


def test_the_embedder_takes_the_visitors_key(monkeypatch, tmp_path):
    from glotto_lens.vectors import YandexEmbedder

    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    embedder = YandexEmbedder("key", "folder")
    assert (embedder.api_key, embedder.folder_id) == ("key", "folder")


def _asset(source, tmp_path, name="glottolens.sqlite.gz", packed=True):
    import gzip
    import hashlib

    asset = tmp_path / "asset" / name
    asset.parent.mkdir(exist_ok=True)
    data = source.read_bytes()
    asset.write_bytes(gzip.compress(data) if packed else data)
    return asset, hashlib.sha256(asset.read_bytes()).hexdigest()


def _documents(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def test_the_database_is_downloaded_once_and_checked(db_path, tmp_path):
    asset, digest = _asset(db_path, tmp_path)
    target = tmp_path / "data" / "glottolens.sqlite"
    assert demo.fetch_database(asset.as_uri(), digest, target) == target
    assert _documents(target) > 0
    asset.unlink()  # the same hash is not downloaded again
    assert demo.fetch_database(asset.as_uri(), digest, target) == target


def test_a_new_asset_replaces_the_old_database(db_path, tmp_path):
    target = tmp_path / "data" / "glottolens.sqlite"
    asset, digest = _asset(db_path, tmp_path)
    demo.fetch_database(asset.as_uri(), digest, target)
    newer = tmp_path / "newer.sqlite"
    newer.write_bytes(db_path.read_bytes())
    with closing(sqlite3.connect(newer)) as connection:
        connection.execute("DELETE FROM documents WHERE id = (SELECT MIN(id) FROM documents)")
        connection.commit()
    before = _documents(target)
    asset, new_digest = _asset(newer, tmp_path, "v2.sqlite.gz")
    demo.fetch_database(asset.as_uri(), new_digest, target)
    assert _documents(target) == before - 1
    assert (target.parent / "glottolens.sqlite.sha256").read_text() == new_digest


def test_gzip_is_recognised_by_content_not_by_name(db_path, tmp_path):
    asset, digest = _asset(db_path, tmp_path, "db-latest", packed=True)
    plain, plain_digest = _asset(db_path, tmp_path, "plain.sqlite.gz", packed=False)
    demo.fetch_database(asset.as_uri(), digest, tmp_path / "a" / "glottolens.sqlite")
    demo.fetch_database(plain.as_uri(), plain_digest, tmp_path / "b" / "glottolens.sqlite")
    assert _documents(tmp_path / "a" / "glottolens.sqlite") > 0
    assert _documents(tmp_path / "b" / "glottolens.sqlite") > 0


@pytest.mark.parametrize("digest", ["", "z" * 64, "0" * 64])
def test_a_missing_or_wrong_hash_leaves_nothing_behind(db_path, tmp_path, digest):
    asset, _ = _asset(db_path, tmp_path)
    target = tmp_path / "data" / "glottolens.sqlite"
    with pytest.raises(ValueError, match="SHA-256"):
        demo.fetch_database(asset.as_uri(), digest, target)
    assert not target.parent.exists() or list(target.parent.iterdir()) == []


def test_an_oversized_download_stops(db_path, tmp_path, monkeypatch):
    asset, digest = _asset(db_path, tmp_path)
    monkeypatch.setattr(demo, "MAX_DOWNLOAD", 10)
    target = tmp_path / "data" / "glottolens.sqlite"
    with pytest.raises(ValueError, match="over"):
        demo.fetch_database(asset.as_uri(), digest, target)
    assert list(target.parent.iterdir()) == []


def test_a_database_from_another_build_is_refused(db_path, tmp_path):
    older = tmp_path / "older.sqlite"
    older.write_bytes(db_path.read_bytes())
    with closing(sqlite3.connect(older)) as connection:
        connection.execute("DROP TABLE languoid_facts")
        connection.commit()
    asset, digest = _asset(older, tmp_path)
    target = tmp_path / "data" / "glottolens.sqlite"
    with pytest.raises(ValueError, match="languoid_facts"):
        demo.fetch_database(asset.as_uri(), digest, target)
    assert not target.exists()


def test_per_request_clients_are_closed():
    from glotto_lens.llm import CachedModel, YandexModel
    from glotto_lens.vectors import CachedEmbedder, YandexEmbedder

    class Client:
        closed = False

        def close(self):
            self.closed = True

    for inner, wrapper in (
        (YandexModel("m", api_key="k", folder_id="f"), CachedModel),
        (YandexEmbedder(api_key="k", folder_id="f"), CachedEmbedder),
    ):
        client = inner._client = Client()
        wrapper(inner).close()
        assert client.closed and inner._client is None
        inner.close()  # nothing open: no error
