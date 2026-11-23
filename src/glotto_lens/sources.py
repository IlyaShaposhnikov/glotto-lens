"""Pinned releases of the source datasets: locations, download and table access."""

import csv
import io
import os
import shutil
import sqlite3
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(os.environ.get("GLOTTO_LENS_DATA", "data")).expanduser()
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "glottolens.sqlite"
USER_AGENT = "glotto-lens"


def read_only(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database read-only, never creating it.

    The URI is built with as_uri(): a Windows path such as C:\\data\\db.sqlite, a space or
    a "?" in a folder name would break a hand-written "file:{path}" URI.
    """
    uri = Path(path or DB_PATH).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


csv.field_size_limit(2**31 - 1)


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    tag: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}/archive/refs/tags/{self.tag}.zip"

    @property
    def filename(self) -> str:
        return f"{self.name}-{self.tag}.zip"


GRAMBANK = Source("grambank", "grambank/grambank", "v1.0.3")
WALS = Source("wals", "cldf-datasets/wals", "v2020.6")
GLOTTOLOG = Source("glottolog", "glottolog/glottolog-cldf", "v5.4")
SOURCES = (GRAMBANK, WALS, GLOTTOLOG)


def download(source: Source, raw_dir: Path = RAW_DIR) -> Path:
    """Download the pinned release archive unless it is already cached."""
    target = raw_dir / source.filename
    if target.exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    request = Request(source.url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target


def cldf_members(archive: Path) -> dict[str, str]:
    """Map paths inside the cldf/ directory to their full names in the archive."""
    members = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = info.filename.split("/")
            if not info.is_dir() and len(parts) > 2 and parts[1] == "cldf":
                members["/".join(parts[2:])] = info.filename
    return members


def read_table(archive: Path, table: str) -> Iterator[dict[str, str]]:
    """Yield rows of a CLDF table, e.g. "languages.csv", as dictionaries."""
    member = cldf_members(archive)[table]
    with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


def read_text(archive: Path, path: str) -> str:
    """Return a text file from the cldf/ directory, e.g. "docs/chapter_1.html"."""
    member = cldf_members(archive)[path]
    with zipfile.ZipFile(archive) as zf:
        return zf.read(member).decode("utf-8")


if __name__ == "__main__":
    for source in SOURCES:
        print(f"{source.name} {source.tag}: {download(source)}")
