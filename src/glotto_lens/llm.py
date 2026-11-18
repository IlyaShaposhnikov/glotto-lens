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
USAGE_LOG = DATA_DIR / "llm_usage.jsonl"


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class ChatModel(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion: ...


def load_env(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into os.environ without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


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

            self._client = OpenAI(api_key=self.api_key, project=self.folder_id, base_url=BASE_URL)
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
        text = getattr(response, "output_text", None) or response.output[0].content[0].text
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=self.name,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


class CachedModel:
    """Wrap a model: identical requests are answered from disk, real calls are logged."""

    def __init__(
        self, model: ChatModel, cache_dir: Path = CACHE_DIR, usage_log: Path = USAGE_LOG
    ) -> None:
        self.model = model
        self.name = model.name
        self.cache_dir = cache_dir
        self.usage_log = usage_log

    def _path(self, *parts: object) -> Path:
        key = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1500
    ) -> Completion:
        path = self._path(self.name, system, user, temperature, max_tokens)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Completion(**{**data, "cached": True})
        completion = self.model.complete(
            system, user, temperature=temperature, max_tokens=max_tokens
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(completion), ensure_ascii=False), encoding="utf-8")
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a", encoding="utf-8") as log:
            record = {"time": time.time(), **asdict(completion)}
            del record["text"]
            log.write(json.dumps(record) + "\n")
        return completion
