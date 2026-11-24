import json
from pathlib import Path

import pytest

from glotto_lens.langid import written_language

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("В русском языке нет тонов [V1][V2].", "ru"),
        (
            "Согласно WALS [V1], он имеет небольшой набор согласных (small consonant inventory).",
            "ru",
        ),
        ("According to WALS [V1], English has a large vowel inventory.", "en"),
        ("El finés tiene diez o más casos [V1].", "es"),
        ("Das Finnische hat zehn oder mehr Fälle [V1].", "de"),
        ("Le finnois a dix cas ou plus [V1], et il ne distingue pas le genre.", "fr"),
        ("Le swahili a cinq genres ou plus [V1][T7].", "fr"),
        ("En français, l’adjectif se place généralement après le nom [V1].", "fr"),
        ("En italiano, el adjetivo suele ir después del nombre [V1].", "es"),
        ("O finlandês tem dez ou mais casos [V1] e não há gênero.", "pt"),
        ("Il finlandese ha dieci o più casi [V1] e non ha genere.", "it"),
        ("フィンランド語には10以上の格があります[V1]。", "ja"),
        ("芬兰语有十个或更多的格[V1]。", "zh"),
        ("핀란드어에는 격이 10개 이상 있습니다 [V1].", "ko"),
    ],
)
def test_the_language_of_an_answer_is_read(text, code):
    assert written_language(text) == code


@pytest.mark.parametrize("text", ["", "[V1][V2]", "SOV 42", "Finnish Finnisch"])
def test_too_little_text_gives_no_language(text):
    assert written_language(text) is None


@pytest.mark.parametrize("name", ["multilingual.jsonl", "multilingual-heldout.jsonl"])
def test_no_multilingual_question_reads_as_another_language(name):
    """None is allowed: the item declares its language, and the rewrite step names it."""
    lines = (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines()
    for item in map(json.loads, filter(None, lines)):
        assert written_language(item["question"]) in (item["written_in"], None), item["id"]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Das ist «Are there articles?» ein Satz.", "de"),
        ("Das ist “Are there articles?” ein Satz.", "de"),
        ("Il finlandese non ha «number of genders» [V1].", "it"),
        ("È una lingua.", "it"),
        ("Es “the is of” mit", "de"),  # without the quote left out, English would win
        ("Es «the is of» mit", "de"),
    ],
)
def test_quotes_are_left_out_and_accented_words_count(text, code):
    assert written_language(text) == code


@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("Qual é a ordem básica das palavras em português?", "Portuguese"),
        ("In italiano l'aggettivo viene prima o dopo il nome?", "Italian"),
        ("日语的基本语序是什么？", "Chinese"),
        ("Сколько падежей в финском?", None),  # Cyrillic: left to the rewrite step
        ("Wat is de woordvolgorde van het Nederlands?", None),  # not among the six
        ("Finnish?", None),
        ("The Finnish", None),  # one function word is not enough
    ],
)
def test_the_answer_language_is_read_only_when_certain(question, name):
    from glotto_lens.langid import confident_language

    assert confident_language(question) == name
