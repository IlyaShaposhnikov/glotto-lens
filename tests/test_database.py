import csv
import io
import sqlite3
import zipfile
from contextlib import closing

import pytest

from glotto_lens.database import build, report


def make_archive(path, tables):
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in tables.items():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            zf.writestr(f"repo-1/cldf/{name}", buffer.getvalue())
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
            {"ID": f"{feature_id}-1", "Parameter_ID": feature_id, "Name": "yes", "Description": ""}
        ],
        "values.csv": [
            {
                "ID": f"{feature_id}-{language}",
                "Language_ID": language,
                "Parameter_ID": feature_id,
                "Value": value,
                "Code_ID": "" if value == "?" else f"{feature_id}-1",
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
    glottolog = structure(languoids, "level", [("russ1263", "language")])
    glottolog["names.csv"] = [
        {"ID": "1", "Language_ID": "russ1263", "Name": "русский", "Provider": "x", "lang": "ru"}
    ]
    grambank = structure(languoids[:2], "GB020", [("russ1263", "1"), ("basq1248", "?")])
    wals_languages = [
        {"ID": "rus", "Name": "Russian", "Glottocode": "russ1263"},
        {"ID": "xxx", "Name": "Unmapped", "Glottocode": ""},
    ]
    wals = structure(wals_languages, "1A", [("rus", "1"), ("xxx", "1")])
    return {
        "glottolog": make_archive(tmp_path / "glottolog.zip", glottolog),
        "grambank": make_archive(tmp_path / "grambank.zip", grambank),
        "wals": make_archive(tmp_path / "wals.zip", wals),
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
    rows = query(db_path, "SELECT glottocode, code_id FROM feature_values WHERE source = 'wals'")
    assert rows == [("russ1263", "wals:1A-1")]


def test_unmapped_language_is_kept_for_reporting(db_path):
    rows = query(db_path, "SELECT glottocode FROM source_languages WHERE name = 'Unmapped'")
    assert rows == [(None,)]


def test_empty_coordinates_become_null(db_path):
    rows = query(db_path, "SELECT latitude FROM languoids WHERE glottocode = 'indo1319'")
    assert rows == [(None,)]


def test_report_counts_overlap_and_unknown_values(db_path):
    result = report(db_path)
    assert result["Glottocodes shared by WALS and Grambank"] == [(1,)]
    assert ("grambank", 2, 1) in result["Values: total / unknown ('?')"]


def test_dangling_code_reference_is_rejected(tmp_path, archives):
    languages = [{"ID": "rus", "Name": "Russian", "Glottocode": "russ1263"}]
    wals = structure(languages, "1A", [("rus", "1")])
    wals["values.csv"][0]["Code_ID"] = "missing"
    archives["wals"] = make_archive(tmp_path / "broken.zip", wals)
    with pytest.raises(sqlite3.IntegrityError):
        build(tmp_path / "broken.sqlite", archives)
