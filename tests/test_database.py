import csv
import io
import sqlite3
import zipfile
from contextlib import closing

import pytest

from glotto_lens.database import build, report
from glotto_lens.search import search

CHAPTER = (
    '<div id="description"><h2>1. Intro</h2><p class="Textbody">'
    '<a class="Language" href="http://wals.info/languoid/lect/wals_code_rus">Russian</a> text.</p>'
    '<table class="values" id="value-table-1A"><caption>Values of Map 1A.</caption>'
    "__values_1A__</table></div>"
)

GRAMBANK_DESCRIPTION = """## Feature

## Summary
Short summary.

## Procedure
Code 1 if present.

## Patron
Some Coder

## Related Features
* GB021

## References
Author 2020."""


def make_archive(path, tables, texts=None):
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in tables.items():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            zf.writestr(f"repo-1/cldf/{name}", buffer.getvalue())
        for name, text in (texts or {}).items():
            zf.writestr(f"repo-1/cldf/{name}", text)
    return path


def languoid(glottocode, name, level):
    return {
        "ID": glottocode,
        "Name": name,
        "Macroarea": "Eurasia",
        "Latitude": "" if level == "family" else "59.0",
        "Longitude": "" if level == "family" else "50.0",
        "Glottocode": glottocode,
        "ISO639P3code": "",
        "Level": level,
        "Countries": "",
        "Family_ID": "",
        "Language_ID": "",
        "Is_Isolate": "",
    }


def structure(languages, feature_id, values):
    return {
        "languages.csv": languages,
        "parameters.csv": [{"ID": feature_id, "Name": "Feature", "Description": "Text"}],
        "codes.csv": [
            {"ID": f"{feature_id}-1", "Parameter_ID": feature_id, "Name": "yes", "Description": ""},
            {"ID": f"{feature_id}-2", "Parameter_ID": feature_id, "Name": "no", "Description": ""},
        ],
        "values.csv": [
            {
                "ID": f"{feature_id}-{language}",
                "Language_ID": language,
                "Parameter_ID": feature_id,
                "Value": value,
                "Code_ID": "" if value == "?" else f"{feature_id}-{value}",
                "Comment": "",
            }
            for language, value in values
        ],
    }


@pytest.fixture
def archives(tmp_path):
    languoids = [
        languoid("russ1263", "Russian", "language"),
        languoid("basq1248", "Basque", "language"),
        languoid("indo1319", "Indo-European", "family"),
    ]
    glottolog = structure(languoids, "level", [("russ1263", "1")])
    glottolog["names.csv"] = [
        {"ID": "1", "Language_ID": "russ1263", "Name": "русский", "Provider": "x", "lang": "ru"}
    ]
    grambank = structure(languoids[:2], "GB020", [("russ1263", "1"), ("basq1248", "?")])
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


def query(db_path, sql):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(sql).fetchall()


def test_feature_ids_are_prefixed_with_source(db_path):
    ids = {row[0] for row in query(db_path, "SELECT id FROM features")}
    assert ids == {"glottolog:level", "grambank:GB020", "wals:1A"}


def test_values_are_keyed_by_glottocode(db_path):
    rows = query(
        db_path,
        "SELECT source_language_id, glottocode, code_id FROM feature_values "
        "WHERE source = 'wals' ORDER BY source_language_id",
    )
    assert rows == [
        ("bsq", "basq1248", "wals:1A-2"),
        ("rus", "russ1263", "wals:1A-1"),
        ("xxx", None, "wals:1A-1"),
    ]


def test_unmapped_language_is_kept_for_reporting(db_path):
    rows = query(db_path, "SELECT glottocode FROM source_languages WHERE name = 'Unmapped'")
    assert rows == [(None,)]


def test_empty_coordinates_become_null(db_path):
    rows = query(db_path, "SELECT latitude FROM languoids WHERE glottocode = 'indo1319'")
    assert rows == [(None,)]


def test_report_counts_overlap_and_unknown_values(db_path):
    result = report(db_path)
    assert result["Glottocodes shared by WALS and Grambank"] == [(2,)]
    values = result["Values: total / unknown ('?') / language without glottocode"]
    assert ("grambank", 2, 1, 0) in values
    assert ("wals", 3, 0, 1) in values


def test_dangling_code_reference_is_rejected(tmp_path, archives):
    languages = [{"ID": "rus", "Name": "Russian", "Glottocode": "russ1263"}]
    wals = structure(languages, "1A", [("rus", "1")])
    wals["values.csv"][0]["Code_ID"] = "missing"
    archives["wals"] = make_archive(tmp_path / "broken.zip", wals)
    with pytest.raises(sqlite3.IntegrityError):
        build(tmp_path / "broken.sqlite", archives)
    assert not (tmp_path / "broken.sqlite").exists()


def test_chapter_without_html_is_skipped(db_path):
    assert query(db_path, "SELECT id, url FROM chapters") == [("1", "https://wals.info/chapter/1")]


def test_blocks_keep_the_source_text_unchanged(db_path):
    rows = query(db_path, "SELECT kind, section, text, feature_id FROM blocks ORDER BY position")
    assert rows == [
        ("paragraph", "1. Intro", "Russian text.", None),
        ("values", "1. Intro", "Values of Map 1A.", "wals:1A"),
    ]


def test_block_languages_are_mapped_to_glottocodes(db_path):
    rows = query(
        db_path,
        "SELECT b.kind, l.glottocode FROM block_languages l JOIN blocks b ON b.id = l.block_id",
    )
    assert rows == [("paragraph", "russ1263")]


def test_chapter_documents_leave_value_tables_to_feature_cards(db_path):
    rows = query(db_path, "SELECT section, text FROM documents WHERE kind = 'chapter'")
    assert rows == [("1. Intro", "Russian text.")]


def test_wals_feature_card_names_chapter_and_distribution(db_path):
    rows = query(db_path, "SELECT section, text, url FROM documents WHERE feature_id = 'wals:1A'")
    card = (
        "WALS feature 1A: Feature\n"
        'Described in the WALS chapter "Chapter" by A. Author.\n\n'
        "Distribution in WALS v2020.6 data:\n"
        "- yes: 2 languages\n"
        "- no: 1 language\n"
        "Total: 3 languages"
    )
    assert rows == [
        ("Summary", card, "https://wals.info/feature/1A"),
        ("Description", "Text", "https://wals.info/feature/1A"),
    ]


def test_grambank_feature_documents_follow_description_sections(db_path):
    rows = query(
        db_path,
        "SELECT section, text FROM documents WHERE feature_id = 'grambank:GB020' ORDER BY position",
    )
    card = (
        "Grambank feature GB020: Feature\n\n"
        "Short summary.\n\n"
        "Distribution in Grambank v1.0.3 data:\n"
        "- yes: 1 language\n"
        "- no: 0 languages\n"
        "- not known (?): 1 language\n"
        "Total: 2 languages"
    )
    assert rows == [("Summary", card), ("Procedure", "Code 1 if present.")]


def test_only_wals_and_grambank_features_get_cards(db_path):
    rows = query(db_path, "SELECT DISTINCT feature_id FROM documents WHERE kind = 'feature'")
    assert {row[0] for row in rows} == {"wals:1A", "grambank:GB020"}


def test_built_database_has_no_foreign_key_violations(db_path):
    assert report(db_path)["Foreign key violations"] == []


def test_search_works_on_the_built_database(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        hits = search(conn, "present")
    assert [(hit.kind, hit.section) for hit in hits] == [("feature", "Procedure")]
    assert hits[0].urls == ("https://grambank.clld.org/parameters/GB020",)
