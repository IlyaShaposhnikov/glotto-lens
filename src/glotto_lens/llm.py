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
CACHE_VERSION = 1
USAGE_LOG = DATA_DIR / "llm_usage.jsonl"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    complete: bool = True  # False when the model stopped at the token limit


class ChatModel(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion: ...


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
        for part in getattr(item, "content", None) or []:
            if getattr(part, "text", None):
                return part.text
    raise RuntimeError(f"The model returned no text; output blocks: {kinds or 'none'}")


class YandexModel:
    """A text model of Yandex AI Studio, called through the Responses API."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, folder_id: str | None = None
    ) -> None:
        load_env()
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        if not self.api_key or not self.folder_id:
            raise RuntimeError("Set YANDEX_API_KEY and YANDEX_FOLDER_ID in the .env file")
        self.name = model or os.environ.get("YANDEX_MODEL", DEFAULT_MODEL)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily: tests never need the SDK

            self._client = OpenAI(
                api_key=self.api_key,
                project=self.folder_id,
                base_url=BASE_URL,
                timeout=TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
        return self._client

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion:
        response = self.client.responses.create(
            model=f"gpt://{self.folder_id}/{self.name}",
            instructions=system,
            input=user,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        return Completion(
            text=response_text(response),
            model=self.name,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            complete=getattr(response, "status", None) in (None, "completed"),
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
            if data.pop("version", None) != CACHE_VERSION:
                return None
            return Completion(**{**data, "cached": True})
        except (OSError, ValueError, TypeError):
            return None

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion:
        path = self._path(self.name, system, user, temperature, max_tokens)
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
