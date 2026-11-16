from glotto_lens.documents import chunk_blocks


def rows(*blocks):
    return [(index, section, kind, text) for index, (section, kind, text) in enumerate(blocks)]


def test_example_stays_with_the_paragraph_that_introduces_it():
    chunks = chunk_blocks(
        rows(("1. Intro", "paragraph", "See (1)."), ("1. Intro", "example", "(1)\nS V O"))
    )
    assert [chunk.text for chunk in chunks] == ["See (1).\n\n(1)\nS V O"]
    assert chunks[0].block_ids == [0, 1]


def test_new_section_starts_a_new_chunk():
    chunks = chunk_blocks(rows(("1. Intro", "paragraph", "One."), ("2. Map", "paragraph", "Two.")))
    assert [(chunk.section, chunk.text) for chunk in chunks] == [
        ("1. Intro", "One."),
        ("2. Map", "Two."),
    ]


def test_paragraph_opens_a_new_chunk_when_the_limit_is_exceeded():
    blocks = rows(
        ("1. Intro", "paragraph", "a" * 60),
        ("1. Intro", "paragraph", "b" * 60),
        ("1. Intro", "example", "c" * 60),
    )
    chunks = chunk_blocks(blocks, max_chars=100)
    assert [chunk.block_ids for chunk in chunks] == [[0], [1, 2]]


def test_no_blocks_give_no_chunks():
    assert chunk_blocks([]) == []
