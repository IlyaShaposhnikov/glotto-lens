import sqlite3
from contextlib import closing

import pytest
from helpers import make_archive, query, structure

from glotto_lens.database import build, report
from glotto_lens.search import search


def test_feature_ids_are_prefixed_with_source(db_path):
    ids = {row[0] for row in query(db_path, "SELECT id FROM features")}
    assert ids == {"glottolog:aes", "grambank:GB020", "wals:1A"}


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
