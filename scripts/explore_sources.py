"""Download the latest releases of the source datasets and print what is inside.

Exploration helper: run once, read the output, then design the loader.
"""

import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from glotto_lens.sources import SOURCES

RAW_DIR = Path("data/raw")
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "glotto-lens"}
API = "https://api.github.com/repos"


def get_json(url: str):
    with urlopen(Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def latest_release(repo: str) -> tuple[str, str]:
    """Return (tag, zipball_url) of the latest release, falling back to the newest tag."""
    try:
        release = get_json(f"{API}/{repo}/releases/latest")
        return release["tag_name"], release["zipball_url"]
    except HTTPError as error:
        if error.code != 404:
            raise
    tag = get_json(f"{API}/{repo}/tags")[0]
    return tag["name"], tag["zipball_url"]


def download(url: str, target: Path) -> None:
    if target.exists():
        print(f"  cached: {target}", file=sys.stderr)
        return
    print(f"  downloading: {target}", file=sys.stderr)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, headers=HEADERS), timeout=300) as response:
        target.write_bytes(response.read())


def describe(name: str, tag: str, archive: Path) -> None:
    print(f"\n=== {name} {tag} ===")
    with zipfile.ZipFile(archive) as zf:
        files = [info for info in zf.infolist() if not info.is_dir()]

        top_level = Counter(info.filename.split("/")[1] for info in files)
        print("Top-level entries (number of files):")
        for entry, count in sorted(top_level.items()):
            print(f"  {entry}: {count}")

        print("CLDF directory:")
        for info in files:
            parts = info.filename.split("/")[1:]
            if parts[0] != "cldf":
                continue
            path = "/".join(parts[1:])
            if not path.endswith(".csv"):
                print(f"  {path} ({info.file_size / 1024:.0f} KB)")
                continue
            with zf.open(info) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                header = next(reader, [])
                rows = sum(1 for _ in reader)
            print(f"  {path}: {rows} rows | {', '.join(header)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    csv.field_size_limit(2**31 - 1)
    for source in SOURCES:
        tag, url = latest_release(source.repo)
        status = "pinned" if tag == source.tag else f"pinned is {source.tag}"
        print(f"{source.name}: latest {tag} ({status})", file=sys.stderr)
        archive = RAW_DIR / f"{source.name}-{tag}.zip"
        download(url, archive)
        describe(source.name, tag, archive)


if __name__ == "__main__":
    main()
