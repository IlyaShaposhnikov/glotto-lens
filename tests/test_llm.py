import json
import os

import pytest

from glotto_lens.llm import CachedModel, Completion, YandexModel, load_env


class CountingModel:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, *, temperature=0.0, max_tokens=1500):
        self.calls += 1
        return Completion(f"echo {user}", self.name, input_tokens=10, output_tokens=2)


def test_cache_answers_repeated_requests_without_calling_the_model(tmp_path):
    inner = CountingModel()
    model = CachedModel(inner, tmp_path / "cache", tmp_path / "usage.jsonl")
    first = model.complete("system", "hello")
    second = model.complete("system", "hello")
    assert inner.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.text == "echo hello"


def test_cache_key_includes_the_prompt_and_parameters(tmp_path):
    inner = CountingModel()
    model = CachedModel(inner, tmp_path / "cache", tmp_path / "usage.jsonl")
    model.complete("system", "hello")
    model.complete("system", "hello", temperature=0.5)
    model.complete("other", "hello")
    assert inner.calls == 3


def test_real_calls_are_logged_without_text(tmp_path):
    model = CachedModel(CountingModel(), tmp_path / "cache", tmp_path / "usage.jsonl")
    model.complete("system", "secret question")
    model.complete("system", "secret question")
    records = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert (records[0]["input_tokens"], records[0]["output_tokens"]) == (10, 2)
    assert "text" not in records[0]


def test_load_env_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nGL_TEST_A='one'\nGL_TEST_B=two\n\nbroken line\n")
    monkeypatch.setenv("GL_TEST_B", "kept")
    monkeypatch.delenv("GL_TEST_A", raising=False)
    load_env(env)
    assert (os.environ["GL_TEST_A"], os.environ["GL_TEST_B"]) == ("one", "kept")
    monkeypatch.delenv("GL_TEST_A")


def test_yandex_model_requires_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    with pytest.raises(RuntimeError, match="YANDEX_API_KEY"):
        YandexModel()
