import zipfile

import pytest

from glotto_lens.sources import GRAMBANK, Source, cldf_members, download, read_table, read_text


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "demo-v1.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("demo-1/README.md", "readme")
        zf.writestr("demo-1/cldf/languages.csv", "ID,Name\nruss1263,Russian\nbasq1248,Basque\n")
        zf.writestr("demo-1/cldf/docs/chapter_1.html", "<p>Привет</p>")
    return path


def test_url_points_to_pinned_tag():
    assert GRAMBANK.url == "https://github.com/grambank/grambank/archive/refs/tags/v1.0.3.zip"


def test_cldf_members_keeps_only_cldf_directory(archive):
    assert set(cldf_members(archive)) == {"languages.csv", "docs/chapter_1.html"}


def test_read_table_yields_rows_as_dicts(archive):
    rows = list(read_table(archive, "languages.csv"))
    assert rows == [
        {"ID": "russ1263", "Name": "Russian"},
        {"ID": "basq1248", "Name": "Basque"},
    ]


def test_read_text_decodes_utf8(archive):
    assert read_text(archive, "docs/chapter_1.html") == "<p>Привет</p>"


def test_download_uses_cached_archive(tmp_path):
    source = Source("demo", "nobody/nothing", "v1")
    cached = tmp_path / source.filename
    cached.write_bytes(b"cached")
    assert download(source, tmp_path) == cached
