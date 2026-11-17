import pytest
from helpers import CHAPTER, GRAMBANK_DESCRIPTION, languoid, make_archive, structure

from glotto_lens.database import build


@pytest.fixture
def archives(tmp_path):
    languoids = {
        row["ID"]: row
        for row in [
            languoid("russ1263", "Russian", "language"),
            languoid("basq1248", "Basque", "language"),
            languoid("indo1319", "Indo-European", "family"),
            languoid("basq9999", "Basque", "dialect"),
            languoid("tula1234", "Tula", "language"),
            languoid("tuln1234", "Tula North", "dialect"),
        ]
    }
    languoids["russ1263"]["ISO639P3code"] = "rus"
    languoids["basq9999"]["Language_ID"] = "basq1248"
    languoids["tuln1234"]["Language_ID"] = "tula1234"
    glottolog = structure(list(languoids.values()), "aes", [("russ1263", "1")])
    glottolog["codes.csv"][0]["Description"] = "EGIDS: 6b"
    glottolog["names.csv"] = [
        {"ID": "1", "Language_ID": "russ1263", "Name": "русский", "Provider": "x", "lang": "ru"}
    ]
    grambank_values = [("russ1263", "1"), ("basq1248", "?"), ("basq9999", "1"), ("tuln1234", "1")]
    grambank = structure([languoids[code] for code, _ in grambank_values], "GB020", grambank_values)
    grambank["parameters.csv"][0]["Description"] = GRAMBANK_DESCRIPTION
    wals_languages = [
        {"ID": "rus", "Name": "Russian", "Glottocode": "russ1263"},
        {"ID": "bsq", "Name": "Basque", "Glottocode": "basq1248"},
        {"ID": "xxx", "Name": "Unmapped", "Glottocode": ""},
    ]
    wals = structure(wals_languages, "1A", [("rus", "1"), ("bsq", "2"), ("xxx", "1")])
    wals["parameters.csv"][0]["Chapter_ID"] = "1"
    wals["chapters.csv"] = [
        {"ID": "1", "Name": "Chapter", "Contributor": "A. Author", "Citation": "Cite"},
        {"ID": "s4", "Name": "No HTML", "Contributor": "", "Citation": ""},
    ]
    return {
        "glottolog": make_archive(tmp_path / "glottolog.zip", glottolog),
        "grambank": make_archive(tmp_path / "grambank.zip", grambank),
        "wals": make_archive(tmp_path / "wals.zip", wals, {"docs/chapter_1.html": CHAPTER}),
    }


@pytest.fixture
def db_path(tmp_path, archives):
    return build(tmp_path / "test.sqlite", archives)
