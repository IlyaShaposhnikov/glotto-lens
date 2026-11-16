"""Print the HTML structure of WALS chapters to plan the text extraction."""

import sys
from collections import Counter
from html.parser import HTMLParser

from glotto_lens.sources import WALS, cldf_members, download, read_text

SAMPLES = ("docs/chapter_1.html", "docs/chapter_81.html", "docs/chapter_139.html")
TAGS = ("p", "h1", "h2", "h3", "table", "tr", "img", "a", "sup", "div", "span", "blockquote")


class TagCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: Counter[str] = Counter()
        self.classes: Counter[str] = Counter()
        self.text_chars = 0

    def handle_starttag(self, tag, attrs):
        self.tags[tag] += 1
        for key, value in attrs:
            if key == "class" and value:
                self.classes[f"{tag}.{value}"] += 1

    def handle_data(self, data):
        self.text_chars += len(data.strip())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    archive = download(WALS)
    paths = sorted(path for path in cldf_members(archive) if path.endswith(".html"))
    classes: Counter[str] = Counter()

    print("chapter | KB | text chars | " + " | ".join(TAGS))
    for path in paths:
        html = read_text(archive, path)
        counter = TagCounter()
        counter.feed(html)
        classes.update(counter.classes)
        counts = " | ".join(str(counter.tags[tag]) for tag in TAGS)
        print(f"{path} | {len(html) // 1024} | {counter.text_chars} | {counts}")

    print("\nMost common tag.class pairs")
    for name, count in classes.most_common(40):
        print(f"  {name}: {count}")

    for path in SAMPLES:
        html = read_text(archive, path)
        middle = len(html) // 2
        print(f"\n=== {path}: first 3000 characters ===\n{html[:3000]}")
        print(f"\n=== {path}: 2000 characters from the middle ===\n{html[middle : middle + 2000]}")


if __name__ == "__main__":
    main()
