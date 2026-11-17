from glotto_lens.chapters import Block, parse_chapter

HTML = """<html><body><div dir="ltr" id="description">
<h2 class="heading-1"><a name="1._Types"></a>1. Types</h2>
<p class="Textbody">The <a class="Language" href="http://wals.info/languoid/lect/wals_code_eng"
title="view language details">English</a> sentence in (1). </p>
<blockquote class="example"><p class="example-start">(1) </p>
<div><table class="table table-hover">
<tr><td><p class="Standard"><i>[The dog] </i></p></td>
<td><p class="Standard"><i>chased </i></p></td></tr>
<tr><td><p class="Standard">S </p></td><td><p class="Standard">V </p></td></tr>
<tr><td colspan="2"> </td></tr>
</table></div>
<p class="example-end"></p></blockquote>
<h3 class="heading-2">1.1 Values</h3>
<div><table class="table table-hover values" id="value-table-81A"><caption><a name="81A"></a>
<b>Values of Map 81A.</b> Order of Subject, Object and Verb</caption>__values_81A__</table></div>
<figure><img alt="" class="figure" src="data:image/png;base64,AAAA"/><br/>
<figcaption><b>Figure 1: </b>Histogram</figcaption></figure>
<h2 class="heading-1">2. Distribution</h2>
<p class="Firstlineindent">Both <a class="Language" href="http://wals.info/languoid/lect/wals_code_jpn">Japanese</a>
and <a class="Language" href="http://wals.info/languoid/lect/wals_code_eng">English</a>.</p>
</div></body></html>"""


def test_blocks_keep_order_kind_and_heading_path():
    blocks = parse_chapter(HTML)
    assert [(block.kind, block.section) for block in blocks] == [
        ("paragraph", "1. Types"),
        ("example", "1. Types"),
        ("values", "1. Types / 1.1 Values"),
        ("figure", "1. Types / 1.1 Values"),
        ("paragraph", "2. Distribution"),
    ]


def test_example_is_rendered_line_by_line():
    example = parse_chapter(HTML)[1]
    assert example.text == "(1)\n[The dog] chased\nS V"


def test_value_table_keeps_feature_id_without_placeholder():
    values = parse_chapter(HTML)[2]
    assert values.feature == "81A"
    assert values.text == "Values of Map 81A. Order of Subject, Object and Verb"


def test_linked_languages_are_collected_without_duplicates():
    blocks = parse_chapter(HTML)
    assert blocks[0].languages == ("eng",)
    assert blocks[4].languages == ("jpn", "eng")


def test_embedded_image_data_is_dropped():
    assert all("base64" not in block.text for block in parse_chapter(HTML))


def test_plain_text_chapter_becomes_one_paragraph():
    assert parse_chapter("Just  plain\ntext.") == [Block("paragraph", "", "Just plain text.")]


def test_fallback_uses_only_the_description_container():
    html = '<nav>Menu</nav><div id="description"><span>Only  text</span></div>'
    assert parse_chapter(html) == [Block("paragraph", "", "Only text")]
