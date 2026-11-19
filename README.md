**[Russian Version / На русском](README.ru.md)**

# GlottoLens

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/IlyaShaposhnikov/glotto-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/IlyaShaposhnikov/glotto-lens/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Grambank%20%C2%B7%20WALS%20%C2%B7%20Glottolog-orange)](#data-and-licenses)
[![LLM](https://img.shields.io/badge/LLM-Alice%20AI-purple)](https://yandex.cloud/en/services/ai-studio)

Ask a question about the grammar of a language, in Russian or English, and get a short
answer in which every statement points to a record in a typological database:

```text
$ glotto-lens ask "Правда ли, что в ирландском глагол стоит в начале предложения?"
Search: 'verb-initial order of subject object and verb VSO', language: 'Irish'

Да, в ирландском языке глагол стоит в начале предложения. Согласно данным WALS, для него
характерен порядок слов VSO (глагол — подлежащее — дополнение) [V3][T9]. Кроме того, по
данным Grambank, для переходных предложений в ирландском немаркированным является именно
глагольно-начальный порядок [V5].

Sources:
[V3] WALS 81A Order of Subject, Object and Verb: Verb-subject-object (VSO) (entry: Irish)
     https://wals.info/datapoint/81A/wals_code_iri
[T9] Order of Subject, Object and Verb / 1. Six types with dominant order (by Matthew S. Dryer)
     https://wals.info/chapter/81
[V5] Grambank GB131 Is a pragmatically unmarked constituent order verb-initial for transitive
     clauses?: present (entry: Irish) https://grambank.clld.org/valuesets/GB131-iris1253
```

(Source lines are wrapped to fit the page.)

GlottoLens is retrieval-augmented generation over three open linguistic databases:

- [Grambank](https://grambank.clld.org): 195 grammatical features of about 2,400 languages;
- [WALS Online](https://wals.info): 192 features with chapters written by typologists;
- [Glottolog](https://glottolog.org): the catalogue of languages, dialects and families
  that links the two through glottocodes.

The model never answers from its own knowledge. It gets the recorded values and the
relevant passages, cites them by label, and says so when the databases have no data.
Labels it invents are detected and marked.

## How it works

```text
question ─► rewrite (LLM) ─► language name + English query in WALS/Grambank terms
                                  │
            ┌─────────────────────┴──────────────────────┐
            ▼                                            ▼
   language lookup (Glottolog names,            full-text search (SQLite FTS5, BM25)
   ISO and WALS codes, dialects)                over feature cards and WALS chapters
            │                                            │
            └──────────────► evidence ◄──────────────────┘
                 recorded values [V1..] + passages [T1..], each with its URL
                                  │
                                  ▼
                answer (LLM) ─► citation check ─► answer with sources
```

- **Data.** Pinned releases of the CLDF datasets (Grambank v1.0.3, WALS v2020.6,
  Glottolog v5.4) are downloaded and built into one SQLite database. Values, codes, WALS
  chapters, Grambank feature descriptions and computed value distributions become
  searchable documents.
- **Language lookup.** Names are normalised (case, diacritics, words like "язык"), and
  matches are ranked: codes, then languages before dialects and families, then the kind of
  name. When a language has no value but its dialect does, the dialect's value is shown
  and labelled as such.
- **Retrieval.** One search over feature documents, keeping the best document of each
  feature, picks up to 8 features; the language's values for them become [V] lines.
  The best passages become [T] lines. Chapters that mention the language are added only
  when they describe one of those features.
- **Prompts.** The rewrite prompt maps everyday wording to database terminology with a
  few examples ("is there a word like 'the'" → "definite articles"). The answer prompt
  requires a citation for every statement, forbids inferring beyond the evidence and
  explains that "no value recorded" means missing data, not absence of the feature.
- **LLM.** [Alice AI](https://yandex.cloud/en/services/ai-studio) through Yandex AI
  Studio's OpenAI-compatible Responses API. Calls are cached on disk and token usage is
  logged, so repeated runs are free.

## Quickstart

Python 3.12 or newer.

```bash
git clone https://github.com/IlyaShaposhnikov/glotto-lens.git
cd glotto-lens
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

glotto-lens build                  # download the three datasets and build data/glottolens.sqlite
```

Searching and looking up languages need no API key:

```bash
glotto-lens lang русский Basque nucl1643       # language lookup: names, glottocodes, codes
glotto-lens search definite articles           # full-text search over the documents
glotto-lens evidence "definite articles" -l Basque   # the evidence the model would get
```

Answering needs a Yandex Cloud folder with access to AI Studio. Put the key in `.env`
(it is ignored by git):

```text
YANDEX_API_KEY=...
YANDEX_FOLDER_ID=...
# YANDEX_MODEL=aliceai-llm         optional
```

```bash
glotto-lens ask "Сколько падежей в финском языке?"
glotto-lens ask "Is Yoruba a tone language?" --show-context
```

`glotto-lens COMMAND --help` lists the options of each command; `python -m glotto_lens.answer`
and the other modules work the same way. Data, the cache and reports live in `data/` (or in
`GLOTTO_LENS_DATA`).

## Evaluation

The question set [`eval/questions.jsonl`](eval/questions.jsonl) has 41 questions in Russian
and English about 24 languages: 33 whose answer is recorded in the data and 8 whose answer
is not. Each item names the feature that answers it and the glottocode the language must
resolve to. The expected values are never written by hand: they are read from the
database, and `check` verifies that every item agrees with the data.

```bash
glotto-lens eval check                     # free: items agree with the data
glotto-lens eval gaps grambank:GB020       # free: languages without a value, for new no-data items
glotto-lens eval run --retrieval-only      # nearly free: rewrite and retrieval only
glotto-lens eval run                       # full run, about 105k input tokens
```

Every report records what it depends on: the git commit of the code, data releases, hashes
of the prompts, of the question set and of the metric code, and the retrieval settings.

### Metrics

| Metric | Counted on | An item passes when |
|---|---|---|
| language resolved | all items | the language resolves to the item's glottocode |
| feature retrieved | all items | the expected feature is in the evidence shown to the model |
| correct | value items | right language, and the answer cites a value line of the expected feature with the recorded value |
| honest | no-data items | right language, feature retrieved, no coded value of it cited, and the answer says the data is missing |
| no invalid citations | all items | the answer cites only labels present in the evidence |

### Results

Alice AI (`aliceai-llm`), temperature 0. Each row is one change, measured on the full set.
The reports behind the rows are in [`eval/reports/`](eval/reports): every answer, the
evidence features and the hashes of prompts, questions and metric. Rows 2 and 3 have the
same answers and differ only in `metrics_sha256`. The last row was measured at commit
`10c926e`; reports record the commit from now on, the earlier rows predate that.

| Stage | Retrieved | Correct | Honest | Invalid citations | Report |
|---|---|---|---|---|---|
| Baseline: 3 feature documents in the evidence | 33/40 | 26/33 | 4/7 | 0 | [1](eval/reports/1-baseline.json) |
| Features chosen from a wider pool; "noun classes" example in the rewrite prompt | 39/41 | 31/33 | 5/8 | 0 | [2](eval/reports/2-wider-pool.json) |
| Same answers, honesty check recognises more phrasings (metric only) | 39/41 | 31/33 | 8/8 | 0 | [3](eval/reports/3-metric-only.json) |
| Pool counts features, not documents; "noun classes → number of genders" | **40/41** | **31/33** | **8/8** | **0** | [4](eval/reports/4-pool-by-feature.json) |

What the numbers say:

- Almost all failures were retrieval failures. When the right feature reached the model, it
  cited the right value in all but one case and never invented a citation.
- Grambank describes each feature in several sections, each a separate document. A pool of
  a few documents often held one or two features, so WALS 30A "Number of Genders" was
  crowded out by a dozen Grambank noun class features. Ranking features instead of
  documents fixed most of it.
- The third row changes no answer, only the metric: honest answers such as
  "WALS does not record the number of cases for Hindi" were not recognised before. The
  baseline honesty figure was measured with an earlier version of the check.
- The set grew from 40 to 41 questions after the baseline (one English no-data question).

## Limitations

- **Tuned on the test set.** Retrieval settings and the rewrite examples were chosen while
  looking at these 41 questions; there is no held-out set yet. The figures show what the
  changes fixed, not how the system does on new questions.
- **Small set.** On 33 value questions one answer is 3 percentage points, and answers vary
  between runs when the evidence changes. In the last row `rus-117A` (possession in
  Russian) failed because the answer cited the Grambank lines instead of the WALS 117A
  line, though the content was right; a run earlier it passed.
- **Lexical search.** BM25 matches words, so extra words in the rewritten query can push the
  right feature out. `en-swa-30A` ("Does Swahili have grammatical gender or noun
  classes?") is rewritten as "number of genders noun classes", and Grambank's noun class
  features outrank WALS 30A. Embeddings as a second search channel would address this.
- **Honesty is a phrase check.** It recognises Russian and English wordings of "no data"
  with a regular expression. It also accepts some claims about the language
  ("род неизвестен"), and it took several corrections to cover the phrasings Alice AI
  actually uses. An LLM judge would be more robust and is planned.
- **One language per question.** Comparisons ("Russian vs Polish") and questions about
  languages in general are not supported yet.
- **Coverage.** The databases have gaps (Grambank has no data for Standard German, for
  example), and 17 WALS languages have no glottocode, so their values cannot be looked up
  by name through Glottolog.
- **One model.** Only Alice AI has been evaluated.

## Project layout

```text
src/glotto_lens/
  sources.py     pinned dataset releases, download, reading CLDF tables
  database.py    build the SQLite database (atomic), sanity report
  chapters.py    parse WALS chapter HTML into blocks
  documents.py   searchable documents: chapter chunks, feature cards, distributions
  search.py      FTS5 search with BM25, per-feature mode
  languages.py   name normalisation, language lookup, language profiles
  context.py     evidence for a question: features, values, passages, labels
  prompts.py     rewrite and answer prompts
  llm.py         Yandex AI Studio client, disk cache, token log
  answer.py      the pipeline: rewrite, retrieve, answer, citation check, answer log
  evaluate.py    question set checks, metrics, reports
  cli.py         the glotto-lens command
eval/questions.jsonl   the question set
tests/                 pytest suite on small synthetic archives, no network or API key
```

```bash
ruff check . && ruff format --check . && pytest
```

## Data and licenses

The code is under the MIT License. The data is not included in the repository: it is
downloaded from the original releases when the database is built, and it stays under the
licenses of those releases (Creative Commons Attribution 4.0 for all three):

- Grambank — [grambank/grambank](https://github.com/grambank/grambank), release v1.0.3;
- WALS Online — [cldf-datasets/wals](https://github.com/cldf-datasets/wals), release v2020.6;
- Glottolog — [glottolog/glottolog-cldf](https://github.com/glottolog/glottolog-cldf),
  release v5.4.

If you use answers produced with GlottoLens, cite the databases as their maintainers ask:
each release lists its recommended citation in its repository and on its website. Every
answer links to the exact WALS datapoint or Grambank valueset it relies on.

## Author

Ilya Shaposhnikov | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

Code under the [MIT license](LICENSE). Data: [Grambank](https://grambank.clld.org),
[WALS Online](https://wals.info) and [Glottolog](https://glottolog.org) under CC BY 4.0,
downloaded from their releases and not redistributed here.

**[Russian Version / На русском](README.ru.md)**
