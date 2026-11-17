"""Builders of tiny CLDF archives for tests."""

import csv
import io
import sqlite3
import zipfile
from contextlib import closing

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


def query(db_path, sql):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(sql).fetchall()
