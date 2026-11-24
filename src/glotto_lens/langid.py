"""Which language a question or an answer is written in, for the evaluation only.

A small heuristic, enough to check that an answer comes in the language of its question:
the writing system decides for Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, Korean,
Japanese and Chinese; for the Latin script, the most frequent function words decide
between English, Spanish, German, French, Portuguese and Italian. Citation labels and text
in quotes or brackets are left out first: answers quote English feature names.
"""

import re
import unicodedata

NOT_PROSE = re.compile(r"\[[^\]]*\]|«[^»]*»|\"[^\"]*\"|“[^”]*”|\([^)]*\)|\*[^*]*\*|/[^/\s]+/")
WORD = re.compile(r"[^\W\d_]+")

SCRIPTS = {  # Unicode name prefix of a letter -> language
    "CYRILLIC": "ru",
    "GREEK": "el",
    "ARABIC": "ar",
    "HEBREW": "he",
    "DEVANAGARI": "hi",
    "THAI": "th",
    "HANGUL": "ko",
    "HIRAGANA": "ja",
    "KATAKANA": "ja",
    "CJK": "zh",
    "LATIN": "latin",
}

FUNCTION_WORDS = {
    "en": "the is are and of in has have does not no a an with for this which there to as by",
    "es": "el la los las de del en es son y que un una no tiene con por se lo para hay suele",
    "de": "der die das den dem und ist sind nicht ein eine hat mit im auf es zu von kein keine",
    "fr": "le la les de des du et est sont un une en ne pas il elle dans pour que qui au aux "
    "l d n qu c se ou avec mais cette ce sur plus",
    "pt": "o os as do da dos das em no na é são e um uma não tem com há ou mas",
    "it": "il lo la gli le di del della e è sono un una non ha in con che per nel l c ma",
}
FUNCTION_WORDS = {code: set(words.split()) for code, words in FUNCTION_WORDS.items()}


def _script(char: str) -> str | None:
    name = unicodedata.name(char, "")
    for prefix, script in SCRIPTS.items():
        if name.startswith(prefix):
            return script
    return None


def written_language(text: str) -> str | None:
    """An ISO 639-1 code, or None when the text is too short or mixed to tell."""
    prose = NOT_PROSE.sub(" ", text)
    counts: dict[str, int] = {}
    for char in prose:
        if char.isalpha() and (script := _script(char)):
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    if "ja" in counts:  # Japanese mixes kana with Chinese characters
        counts["ja"] += counts.pop("zh", 0)
    script = max(counts, key=counts.get)
    if counts[script] * 2 <= sum(counts.values()):
        return None
    if script != "latin":
        return script
    words = [word.casefold() for word in WORD.findall(prose)]
    hits = {
        code: sum(word in vocabulary for word in words)
        for code, vocabulary in FUNCTION_WORDS.items()
    }
    ranked = sorted(hits.values(), reverse=True)
    if ranked[0] == 0 or ranked[0] == ranked[1]:
        return None
    return max(hits, key=hits.get)


# Codes whose reading is reliable enough to name the answer language: writing systems used
# by one language here, and the six Latin-script languages when at least two of their function words
# are found. Cyrillic, Arabic and Devanagari serve several languages, so the rewrite step
# names those.
NAMES = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
}


def confident_language(text: str) -> str | None:
    """The English name of the language of a question, only when the text leaves no doubt.

    The rewrite model took Portuguese and Italian questions for Spanish and a Chinese
    question about Japanese for Japanese; this reading does not depend on what the
    question is about.
    """
    code = written_language(text)
    if code not in NAMES:
        return None
    if code in FUNCTION_WORDS:
        words = [word.casefold() for word in WORD.findall(NOT_PROSE.sub(" ", text))]
        hits = sorted(
            (sum(word in vocabulary for word in words) for vocabulary in FUNCTION_WORDS.values()),
            reverse=True,
        )
        if hits[0] < 2:  # one function word is too little to overrule the rewrite step
            return None
    return NAMES[code]
