"""Language model access: Yandex AI Studio through the OpenAI SDK, with a disk cache.

Every call goes through ChatModel.complete(system, user). The cache makes repeated
experiments free and reproducible; token usage of real calls is appended to a log.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from glotto_lens.sources import DATA_DIR

BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_MODEL = "aliceai-llm"
CACHE_DIR = DATA_DIR / "cache" / "llm"
CACHE_VERSION = 2
USAGE_LOG = DATA_DIR / "llm_usage.jsonl"
TIMEOUT_SECONDS = 60  # default; YANDEX_TIMEOUT overrides it, reasoning models are slower
MAX_RETRIES = 2
# AI Studio sometimes answers 403 for a model the key can use, then serves the same request a
# little later (seen in long eval runs); the SDK does not retry 403, so these waits do
FORBIDDEN_RETRY_SECONDS = (5, 15, 30)
APIS = ("responses", "chat")  # Responses API or Chat Completions; models differ in support


class ModelError(RuntimeError):
    """A model call failed; the message says which model, why, and what to try."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    complete: bool = True  # False when the model stopped early, see stop_reason
    stop_reason: str | None = None


class ChatModel(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion: ...


def timeout_seconds() -> int:
    """YANDEX_TIMEOUT from the environment or .env, read when a model is created."""
    text = os.environ.get("YANDEX_TIMEOUT", str(TIMEOUT_SECONDS))
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"YANDEX_TIMEOUT must be a positive number of seconds, not {text!r}")
    return value


def load_env(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into os.environ without overriding variables already set.

    The file is looked up in the current directory, so commands run from the project root.
    Supported: blank lines, lines starting with #, values in single or double quotes.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def response_text(response) -> str:
    """Text of a Responses API result: the first message with text, skipping other blocks."""
    text = getattr(response, "output_text", None)
    if text:
        return text
    kinds = []
    for item in getattr(response, "output", None) or []:
        kinds.append(getattr(item, "type", type(item).__name__))
        if getattr(item, "type", None) not in (None, "message"):
            continue
        for part in getattr(item, "content", None) or []:
            if getattr(part, "text", None):
                return part.text
    raise RuntimeError(f"The model returned no text; output blocks: {kinds or 'none'}")


class YandexModel:
    """A text model of Yandex AI Studio, through the Responses API or Chat Completions.

    Alice AI, YandexGPT and DeepSeek answer through the Responses API; some open models of
    AI Studio may be reachable only through Chat Completions (api="chat").
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        folder_id: str | None = None,
        api: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        load_env()  # before reading any setting, so values from .env count
        self.timeout = timeout_seconds()
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        if not self.api_key or not self.folder_id:
            raise RuntimeError("Set YANDEX_API_KEY and YANDEX_FOLDER_ID in the .env file")
        self.name = model or os.environ.get("YANDEX_MODEL", DEFAULT_MODEL)
        self.api = api or os.environ.get("YANDEX_API", "responses")
        if self.api not in APIS:
            raise ValueError(f"api must be one of {APIS}, not {self.api!r}")
        # None: the model's default. DeepSeek and Qwen reason by default; "none" turns it off
        self.reasoning_effort = reasoning_effort or os.environ.get("YANDEX_REASONING_EFFORT")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily: tests never need the SDK

            self._client = OpenAI(
                api_key=self.api_key,
                project=self.folder_id,
                base_url=BASE_URL,
                timeout=self.timeout,
                max_retries=MAX_RETRIES,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client, if one was opened; the next call opens a new one."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion:
        call = self._chat if self.api == "chat" else self._responses
        for wait in (*FORBIDDEN_RETRY_SECONDS, None):
            try:
                return call(system, user, temperature, max_tokens)
            except Exception as error:
                if getattr(error, "status_code", None) == 403 and wait is not None:
                    time.sleep(wait)
                    continue
                explained = self._model_error(error)
                if explained is None:
                    raise
                raise explained from error
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _model_error(self, error: Exception) -> ModelError | None:
        """A ModelError that says what to try, or None for errors this class does not know."""
        status = getattr(error, "status_code", None)
        if status is not None:
            return ModelError(self._explain(status, error))
        if type(error).__name__ in ("APIConnectionError", "APITimeoutError"):
            return ModelError(
                f"model {self.name!r}: no response ({type(error).__name__}); check the"
                f" network, or raise YANDEX_TIMEOUT (now {self.timeout} s)"
            )
        return None

    def _explain(self, status: int, error: Exception) -> str:
        kind = type(error).__name__
        text = f"model {self.name!r} through the {self.api} API: HTTP {status} ({kind})"
        if status in (401, 403):
            other = "chat" if self.api == "responses" else "responses"
            text += (
                ". The key or the folder may have no access to this model, or the model may"
                f" not be served through this API; try --api {other}"
            )
            if status == 403:
                text += f" (refused again after waits of {FORBIDDEN_RETRY_SECONDS} s)"
        elif status == 404:
            text += ". Check the model name in the AI Studio model list"
        elif status == 429:
            text += ". Too many requests or the quota is spent; wait and run again"
        elif status == 400:
            text += (
                ". The request was rejected; some models do not accept every temperature or"
                " max_tokens value"
            )
        details = getattr(error, "message", "") or ""
        return f"{text}. {details}".strip()

    @property
    def settings(self) -> dict:
        """What besides the name changes the model's replies, for reports."""
        return {"api": self.api, "reasoning_effort": self.reasoning_effort}

    def _responses(self, system: str, user: str, temperature: float, max_tokens: int):
        extra = {"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}
        response = self.client.responses.create(
            model=f"gpt://{self.folder_id}/{self.name}",
            instructions=system,
            input=user,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **extra,
        )
        usage = getattr(response, "usage", None)
        status = getattr(response, "status", None)
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        try:
            text = response_text(response)
        except RuntimeError:
            if status in (None, "completed"):
                raise
            # a reasoning model can spend the whole token budget before it writes any text
            text = ""
        return Completion(
            text=text,
            model=self.name,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            complete=status in (None, "completed"),
            stop_reason=reason,
        )

    def _chat(self, system: str, user: str, temperature: float, max_tokens: int):
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        response = self.client.chat.completions.create(
            model=f"gpt://{self.folder_id}/{self.name}",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        choice = response.choices[0]
        text = choice.message.content
        if not text:
            raise RuntimeError(f"The model returned no text (finish reason {choice.finish_reason})")
        usage = getattr(response, "usage", None)
        reason = choice.finish_reason
        return Completion(
            text=text,
            model=self.name,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            complete=reason in (None, "stop"),
            stop_reason=None if reason in (None, "stop") else reason,
        )


class CachedModel:
    """Wrap a model: identical requests are answered from disk, real calls are logged."""

    def __init__(
        self,
        model: ChatModel,
        cache_dir: Path = CACHE_DIR,
        usage_log: Path = USAGE_LOG,
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.name = model.name
        self.settings = getattr(model, "settings", {})
        self.cache_dir = cache_dir
        self.usage_log = usage_log
        self.use_cache = use_cache  # False: always call the model, but refresh the cache

    def _path(self, *parts: object) -> Path:
        key = json.dumps(parts, ensure_ascii=False)
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    @staticmethod
    def _read(path: Path) -> Completion | None:
        """A cached completion, or None for a missing, damaged or outdated file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.pop("version", None) != CACHE_VERSION:
                return None
            return Completion(**{**data, "cached": True})
        except (OSError, ValueError, TypeError):
            return None

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if close is not None:
            close()

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion:
        api = getattr(self.model, "api", "responses")
        effort = getattr(self.model, "reasoning_effort", None)
        # settings join the key only when they are not the default, so older entries stay valid
        parts = (BASE_URL, self.name, system, user, temperature, max_tokens)
        extra = (*(() if api == "responses" else (api,)), *(() if effort is None else (effort,)))
        path = self._path(*parts, *extra)
        if self.use_cache and (cached := self._read(path)):
            return cached
        completion = self.model.complete(
            system, user, temperature=temperature, max_tokens=max_tokens
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        record = {"version": CACHE_VERSION, **asdict(completion)}
        partial.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        partial.replace(path)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a", encoding="utf-8") as log:
            record = {"time": time.time(), **asdict(completion)}
            del record["text"]
            log.write(json.dumps(record) + "\n")
        return completion
