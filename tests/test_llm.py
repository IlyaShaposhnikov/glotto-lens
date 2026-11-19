import json
import os
from types import SimpleNamespace

import pytest

from glotto_lens.llm import CachedModel, Completion, YandexModel, load_env, response_text


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


def test_damaged_or_outdated_cache_files_count_as_misses(tmp_path):
    inner = CountingModel()
    model = CachedModel(inner, tmp_path / "cache", tmp_path / "usage.jsonl")
    model.complete("system", "hello")
    (path,) = (tmp_path / "cache").glob("*.json")
    path.write_text("{broken", encoding="utf-8")
    assert model.complete("system", "hello").cached is False
    path.write_text(json.dumps({"version": 0, "text": "old", "model": "fake"}), encoding="utf-8")
    assert model.complete("system", "hello").cached is False
    assert inner.calls == 3
    assert not list((tmp_path / "cache").glob("*.part"))


def test_no_cache_mode_calls_the_model_and_refreshes_the_cache(tmp_path):
    inner = CountingModel()
    CachedModel(inner, tmp_path / "cache", tmp_path / "u.jsonl").complete("s", "u")
    fresh = CachedModel(inner, tmp_path / "cache", tmp_path / "u.jsonl", use_cache=False)
    assert fresh.complete("s", "u").cached is False
    assert inner.calls == 2


def test_response_text_skips_blocks_without_text():
    reasoning = SimpleNamespace(type="reasoning", content=None)
    message = SimpleNamespace(type="message", content=[SimpleNamespace(text="Answer")])
    assert response_text(SimpleNamespace(output_text="", output=[reasoning, message])) == "Answer"
    with pytest.raises(RuntimeError, match="reasoning"):
        response_text(SimpleNamespace(output_text=None, output=[reasoning]))


class FakeError(Exception):
    def __init__(self, status_code, message="denied"):
        super().__init__(message)
        self.status_code, self.message = status_code, message


def fake_client(**handlers):
    def create(handler):
        return lambda **kwargs: handler(kwargs)

    return SimpleNamespace(
        responses=SimpleNamespace(create=create(handlers.get("responses"))),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create(handlers.get("chat")))),
    )


def test_chat_completions_api(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    seen = {}

    def chat(kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content="ответ")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )

    model = YandexModel("qwen", api="chat")
    model._client = fake_client(chat=chat)
    completion = model.complete("system", "user", max_tokens=5)
    assert seen["model"] == "gpt://f/qwen" and seen["messages"][0]["role"] == "system"
    assert (completion.text, completion.input_tokens, completion.complete) == ("ответ", 7, False)
    assert completion.stop_reason == "length"


def test_api_errors_name_the_model_and_the_way_out(monkeypatch):
    from glotto_lens.llm import ModelError

    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    def refuse(kwargs):
        raise FakeError(403)

    model = YandexModel("gpt-oss-120b")
    model._client = fake_client(responses=refuse)
    with pytest.raises(ModelError, match="'gpt-oss-120b' through the responses API: HTTP 403"):
        model.complete("s", "u")
    with pytest.raises(ModelError, match="try --api chat"):
        model.complete("s", "u")


def test_unknown_api_is_rejected(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    with pytest.raises(ValueError):
        YandexModel(api="grpc")


def test_chat_api_has_its_own_cache_entries(tmp_path):
    inner = CountingModel()
    model = CachedModel(inner, tmp_path / "cache", tmp_path / "usage.jsonl")
    model.complete("system", "hello")
    inner.api = "chat"
    model.complete("system", "hello")
    assert inner.calls == 2


def test_reasoning_that_spends_the_budget_is_an_incomplete_reply(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    def thinking(kwargs):
        return SimpleNamespace(
            output_text="",
            output=[SimpleNamespace(type="reasoning", content=[])],
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=SimpleNamespace(input_tokens=300, output_tokens=200),
        )

    model = YandexModel("deepseek-v32")
    model._client = fake_client(responses=thinking)
    completion = model.complete("s", "u", max_tokens=200)
    assert (completion.text, completion.complete) == ("", False)
    assert completion.stop_reason == "max_output_tokens"


def test_network_errors_become_model_errors(monkeypatch):
    from glotto_lens.llm import ModelError

    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    class APITimeoutError(Exception):
        pass

    def slow(kwargs):
        raise APITimeoutError("timed out")

    model = YandexModel("aliceai-llm")
    model._client = fake_client(responses=slow)
    with pytest.raises(ModelError, match="no response"):
        model.complete("s", "u")


@pytest.mark.parametrize(("status", "hint"), [(429, "wait"), (400, "temperature")])
def test_error_hints(monkeypatch, status, hint):
    from glotto_lens.llm import ModelError

    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")

    def refuse(kwargs):
        raise FakeError(status)

    model = YandexModel("aliceai-llm")
    model._client = fake_client(responses=refuse)
    with pytest.raises(ModelError, match=hint):
        model.complete("s", "u")


def test_reasoning_effort_reaches_both_apis_and_the_cache_key(monkeypatch, tmp_path):
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    seen = []

    def responses(kwargs):
        seen.append(kwargs.get("reasoning"))
        return SimpleNamespace(output_text="ok", status="completed", usage=None)

    def chat(kwargs):
        seen.append(kwargs.get("reasoning_effort"))
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    for api in ("responses", "chat"):
        model = YandexModel("qwen", api=api, reasoning_effort="none")
        model._client = fake_client(responses=responses, chat=chat)
        model.complete("s", "u")
    assert seen == [{"effort": "none"}, "none"]
    assert model.settings == {"api": "chat", "reasoning_effort": "none"}

    inner = CountingModel()
    cached = CachedModel(inner, tmp_path / "cache", tmp_path / "usage.jsonl")
    cached.complete("s", "u")
    inner.reasoning_effort = "none"
    cached.complete("s", "u")
    assert inner.calls == 2
