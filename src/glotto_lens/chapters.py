"""Turn WALS chapter HTML into typed text blocks with their provenance."""

import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from glotto_lens.sources import WALS, cldf_members, download, read_text

HEADINGS = {"h2": 1, "h3": 2, "h4": 3}
CONTAINERS = ("blockquote", "table", "figure")
LANGUAGE_LINK = re.compile(r"wals_code_(\w+)")
VALUE_TABLE = re.compile(r"value-table-(\w+)")


@dataclass(frozen=True)
class Block:
    kind: str  # paragraph | example | values | table | figure
    section: str
    text: str
    languages: tuple[str, ...] = ()
    feature: str | None = None


def _clean(text: str) -> str:
    return " ".join(text.split())


def _rows(element: Tag) -> list[str]:
    rows = []
    for row in element.find_all("tr"):
        cells = [_clean(cell.get_text()) for cell in row.find_all(["td", "th"])]
        line = " ".join(cell for cell in cells if cell)
        if line:
            rows.append(line)
    return rows


def _languages(element: Tag) -> tuple[str, ...]:
    codes = []
    for link in element.find_all("a", class_="Language"):
        match = LANGUAGE_LINK.search(link.get("href", ""))
        if match and match.group(1) not in codes:
            codes.append(match.group(1))
    return tuple(codes)


def _example(element: Tag) -> str:
    start = element.find("p", class_="example-start")
    lines = [_clean(start.get_text())] if start else []
    lines += _rows(element)
    return "\n".join(line for line in lines if line)


def parse_chapter(html: str) -> list[Block]:
    """Split one chapter into blocks, keeping the heading path of each block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "html.parser")
    root = soup.find(id="description") or soup
    path: dict[int, str] = {}
    blocks = []
    for element in root.find_all([*HEADINGS, "p", "figcaption", *CONTAINERS]):
        if element.find_parent(CONTAINERS) and element.name != "figcaption":
            continue
        feature = None
        if element.name in HEADINGS:
            level = HEADINGS[element.name]
            path = {key: value for key, value in path.items() if key < level}
            path[level] = _clean(element.get_text())
            continue
        if element.name == "figure":
            continue
        if element.name == "blockquote":
            kind, text = "example", _example(element)
        elif element.name == "table":
            match = VALUE_TABLE.search(element.get("id", ""))
            if match:
                caption = element.find("caption")
                kind, feature = "values", match.group(1)
                text = _clean(caption.get_text()) if caption else ""
            else:
                kind, text = "table", "\n".join(_rows(element))
        elif element.name == "figcaption":
            kind, text = "figure", _clean(element.get_text())
        else:
            kind, text = "paragraph", _clean(element.get_text())
        if text:
            section = " / ".join(path[level] for level in sorted(path))
            blocks.append(Block(kind, section, text, _languages(element), feature))
    if not blocks and _clean(soup.get_text()):
        blocks.append(Block("paragraph", "", _clean(soup.get_text())))
    return blocks


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    archive = download(WALS)
    paths = sorted(path for path in cldf_members(archive) if path.endswith(".html"))
    kinds: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    copies = defaultdict(list)
    linked = set()
    for path in paths:
        blocks = parse_chapter(read_text(archive, path))
        copies[tuple(blocks)].append(path)
        for block in blocks:
            kinds[block.kind] += 1
            chars[block.kind] += len(block.text)
            linked.update(block.languages)

    print("kind | blocks | characters")
    for kind, count in kinds.most_common():
        print(f"  {kind} | {count} | {chars[kind]}")
    print(f"Chapters: {len(paths)}, unique texts: {len(copies)}, linked languages: {len(linked)}")
    for group in copies.values():
        if len(group) > 1:
            print("  identical: " + ", ".join(group))

    print("\nFirst blocks of chapter 81")
    for block in parse_chapter(read_text(archive, "docs/chapter_81.html"))[:8]:
        print(f"\n[{block.kind}] {block.section} {block.languages}\n{block.text}")
