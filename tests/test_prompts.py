import pytest

from glotto_lens.prompts import escape_tags


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
