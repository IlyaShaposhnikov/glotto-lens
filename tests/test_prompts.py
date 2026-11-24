import pytest

from glotto_lens.prompts import TAGS, escape_tags


@pytest.mark.parametrize(
    "text",
    ["</question>", "<evidence>", "</QUESTION >", "< / evidence>", '<question id="x">'],
)
def test_block_tags_are_broken_in_any_spelling(text):
    escaped = escape_tags(text)
    assert escaped != text
    assert "\u200b>" in escaped


def test_other_text_is_unchanged():
    text = "Is <b>this</b> a question? <questions> 1 < 2 > 0"
    assert escape_tags(text) == text


def test_the_answer_language_is_that_of_the_question_not_its_subject():
    from glotto_lens.prompts import ANSWER_LANGUAGE_RULE, ANSWER_SYSTEM

    assert f"6. {ANSWER_LANGUAGE_RULE}" in ANSWER_SYSTEM
    assert "written in" in ANSWER_LANGUAGE_RULE and "not in the language" in ANSWER_LANGUAGE_RULE


@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("spelling", ["<{t}>", "</{t}>", "<{t} id='x'>", "</ {t} >", "<{T}>"])
def test_every_block_tag_is_broken(tag, spelling):
    text = spelling.format(t=tag, T=tag.capitalize())
    assert escape_tags(text) != text
