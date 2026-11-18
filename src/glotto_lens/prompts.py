"""Prompts of the question answering pipeline.

Kept in one place so that prompt changes are visible in review and can be compared in
experiments. Untrusted text (the question, and texts from the databases) is always placed
between explicit tags, and closing tags inside it are neutralised by escape_tags().
"""

import re

TAGS = ("question", "evidence")
TAG = re.compile(r"<(\s*/?\s*(?:" + "|".join(TAGS) + r")\b[^>]*)>", re.IGNORECASE)


def escape_tags(text: str) -> str:
    """Break our block tags inside untrusted text, in any case and spacing, so that the text
    cannot end its own block and fake another one."""
    return TAG.sub(lambda match: f"<{match.group(1)}\u200b>", text)


REWRITE_SYSTEM = """You prepare questions about the grammar of languages for a search engine \
over the typological databases WALS and Grambank.

Read the question between <question> tags. It may be in any language.
Return only a JSON object, with no other text, with two keys:
- "language": the English name of the one language the question is about, as used in \
Glottolog (for example "Russian", "Basque", "Mandarin Chinese"). If the question is about \
languages in general but names one example language, give that language. Use null only \
when no language is named;
- "query": a short English search query of 3-10 words in the terminology of linguistic \
typology, without the language name. Use the names of WALS and Grambank features, not \
everyday words. Examples of everyday wording and the query to give:
  "basic word order" -> "order of subject object and verb"
  "does the verb come first" -> "verb-initial order of subject object and verb VSO"
  "is there a word like 'the'" -> "definite articles"
  "how many cases" -> "number of cases"
  "grammatical gender" -> "number of genders sex-based gender"
  "is it a tone language" -> "tone"
  "how is plural formed" -> "coding of nominal plurality"
  "is there a word for 'to have'" -> "predicative possession"
"""

REWRITE_USER = "<question>\n{question}\n</question>"

ANSWER_SYSTEM = """You are a careful assistant in linguistic typology. You answer questions \
about languages using only the evidence given to you, never your own knowledge.

The evidence between <evidence> tags lists values recorded in databases, labelled [V1], \
[V2], ..., and texts from the WALS and Grambank descriptions, labelled [T1], [T2], ...

Rules:
1. Every factual statement must cite its evidence with labels in square brackets, for \
example "Russian has SVO order [V1]". Cite only labels that appear in the evidence.
2. State only what the cited evidence says. Do not infer further properties: a value about \
main and subordinate clauses says nothing about whether word order is free.
3. If the evidence does not answer the question, say so plainly and say what is missing, \
for example that a database has no value for this language. Do not guess.
4. A value marked "no value recorded" means the database has no data, not that the \
language lacks the feature.
5. Mention when a value comes from a dialect or a secondary entry, and when several \
languages matched the name.
6. Answer in the language of the question. Be concise: a short paragraph, or a few \
bullet points for comparisons."""

ANSWER_USER = """<question>
{question}
</question>

<evidence>
{evidence}
</evidence>"""
