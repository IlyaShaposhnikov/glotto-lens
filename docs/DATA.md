**English** | [Русский](DATA.ru.md)

# The data behind GlottoLens

GlottoLens answers from three open databases of linguistic typology, the field that compares
the grammars and sound systems of the world's languages. They are published as
[CLDF](https://cldf.clld.org) datasets: CSV tables with a common structure (languages,
parameters, codes, values) and an explicit link from every value to its source. All
figures below are counted in the database GlottoLens builds from the pinned releases, at
commit `23c957e`; the document counts also depend on how GlottoLens splits the texts.

| | Glottolog v5.4 | WALS v2020.6 | Grambank v1.0.3 |
|---|---|---|---|
| What it is | catalogue and family tree of languages | atlas of language structures | questionnaire of grammatical features |
| Unit | languoid: language, dialect or family | feature with a few categories (6 on average) | yes/no question (some with a few values) |
| Features | 8 catalogue parameters (not typological) | 192 | 195 |
| Languoids or languages with values | 27,275 languoids (all levels) | 2,660 | 2,467 |
| Values | 181,251 | 76,475 | 441,663 |
| Filled cells | — | 15% of languages × features | 92%, of which 18% coded "?" |
| Text | — | 151 chapters with maps and references | a description and coding procedure for each feature |

## Glottolog: who is who

[Glottolog](https://glottolog.org) assigns every languoid a stable identifier, the
glottocode (`russ1263` for Russian), and places it in a family tree. Version 5.4 has 8,623
languages, 13,773 dialects and 4,879 families and subgroups; 7,861 languages also have an
ISO 639-3 code. By macroarea, Africa (2,374 languages), Papunesia (2,225) and Eurasia
(2,029) together hold three quarters of them.

GlottoLens uses Glottolog mainly to turn "русский", "Basque" or `nucl1643` into a
glottocode, and also for the family, the macroarea, the endangerment status and the most
extensive description shown with every answer, and to find a dialect's value when the
language has none. Besides the main name it keeps the names other catalogues use (Multitree,
Lexvo, ELCat, AIATSIS, WALS and others, over 120,000 names in all), so the same language is
found under different spellings. The glottocode is the join key between the three sources.

## WALS: the atlas

The [World Atlas of Language Structures](https://wals.info) maps 192 features in 151
chapters, each written by a specialist. A feature sorts languages into a few categories;
WALS 49A "Number of Cases", for example, has nine: no morphological case-marking, 2 cases,
3, 4, 5, 6–7, 8–9, 10 or more, and exclusively borderline case-marking.

Coverage is uneven by design: authors chose language samples per chapter, so most of the
2,660 languages have values for only some features, and only 15% of the language-feature
pairs are filled. Russian has 156 values. WALS uses its own language codes; 17 languages
with values have no glottocode, so they cannot be found by name.

The chapters are the richest text in the project: GlottoLens splits them into 4,888 blocks
and uses them as passages that explain what a feature means.

## Grambank: the questionnaire

[Grambank](https://grambank.clld.org) asks the same 195 questions about every language,
mostly of the form "Is there X?", such as GB020 "Are there definite or specific articles?".
Each feature has a description and a procedure telling the coder when to answer 1 or 0.

It is dense: 2,467 languages and values for 92% of the pairs. The answers are mostly
binary (242,782 values "0", 112,315 "1"); a few features have more values ("2": 5,410,
"3": 1,518). The code "?" (79,638 values, 18%) means the coders looked and could not
decide, which is different from a missing row. Russian has a row for all 195 features.

## One language in all three

Russian is `russ1263` in Glottolog: a language of Eurasia, ISO code `rus`, Indo-European
family. WALS gives it, among 156 values, "6-7 case categories" for 49A; Grambank answers
"absent" for GB020, definite articles. A question about Russian cases is answered from the
WALS line, with a link to the exact datapoint.

## How it is stored

`glotto-lens build` downloads the three releases and writes one SQLite file:

- `languoids`, `names`, `source_languages`: Glottolog entries, their names, and the WALS
  and Grambank language codes mapped to glottocodes;
- `features`, `codes`, `feature_values`: every feature, its categories and every value,
  with the source's own language code and a comment where the source gives one;
- `chapters`, `blocks`: WALS chapters split into sections and paragraphs, with the
  languages each block mentions;
- `documents` and the full-text index: what search sees, 1,837 chapter passages and 863
  feature documents (feature descriptions, Grambank coding procedures and examples, value
  distributions). With hybrid search the feature documents also get vectors.

## What this means for answers

- **"No data" is common and is an answer.** WALS is sparse, and even Grambank has gaps and
  "?" codes. GlottoLens says when a value is missing instead of guessing; the evaluation
  checks it.
- **The two databases overlap but do not agree in shape.** WALS 37A and Grambank GB020 both
  ask about definite articles, with different categories, and one may have a value where
  the other has none. A question that names no database can be answered from either.
- **Dialects and varieties.** When a language has no value but one of its dialects does,
  GlottoLens shows the dialect's value and says so.

## Licenses and citation

All three releases are under Creative Commons Attribution 4.0. GlottoLens does not
redistribute them; it downloads them from the original releases
([Glottolog](https://github.com/glottolog/glottolog-cldf),
[WALS](https://github.com/cldf-datasets/wals),
[Grambank](https://github.com/grambank/grambank)). If you use the data, cite the databases
as their maintainers ask.
