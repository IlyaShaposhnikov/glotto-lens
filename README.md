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
  feature, picks up to 8 features; the language's values for them become [V] lines. With
  `--hybrid`, BM25 keeps the first 6 places and the last 2 go to features found by Yandex
  embeddings (256 dimensions, stored in the same SQLite file) that BM25 missed.
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
glotto-lens search definite articles --per-feature  # features ranked as for the evidence
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

Hybrid search embeds the 863 feature documents once (a few minutes, about 230k tokens,
cached on disk), then adds one embedding call per question:

```bash
glotto-lens embed
glotto-lens ask "Does Swahili have noun classes?" --hybrid
glotto-lens search number of genders noun classes --vectors-only   # the vector channel alone
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

A second set, [`eval/heldout.jsonl`](eval/heldout.jsonl), has 32 questions (23 with a value,
9 without) about 30 languages, none of them in the first set; a test keeps it so. It was
committed before its first run and is not used for tuning (`--questions eval/heldout.jsonl`).

```bash
glotto-lens eval check                     # free: items agree with the data
glotto-lens eval gaps grambank:GB020       # free: languages without a value, for new no-data items
glotto-lens eval run --retrieval-only      # nearly free: rewrite and retrieval only
glotto-lens eval run                       # full run, about 105k input tokens
glotto-lens eval run --judge --judge-model deepseek-v4.1-flash   # plus an LLM verdict per answer
glotto-lens eval run -m yandexgpt-5.1 --rewrite-model aliceai-llm  # another answer model, same evidence
```

Every report records what it depends on: the git commit of the code, data releases, hashes
of the prompts, of the question set and of the metric code, the retrieval settings, and the
models with their settings (API, reasoning effort). Answers and verdicts are cached, so
rerunning a finished run, or one where some calls failed, only pays for what is missing.

### Metrics

| Metric | Counted on | An item passes when |
|---|---|---|
| language resolved | all items | the language resolves to the item's glottocode |
| feature retrieved | all items | the expected feature is in the evidence shown to the model |
| correct | value items | right language, and the answer cites a value line of the expected feature with the recorded value |
| honest | no-data items | right language, feature retrieved, no coded value of it cited, and the answer says the data is missing |
| no invalid citations | all items | the answer cites only labels present in the evidence |
| judge (with `--judge`) | all items | an LLM, given the question, the recorded value and the answer, but not the evidence, says the answer states that value or admits the gap |

The judge is a cross-check, not the main metric: where it and the code-based checks
disagree, the answer is read by hand. That is how three more ways of saying "no data" were
found (see below).

### Results: retrieval

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
| Pool counts features, not documents; "noun classes → number of genders" | 40/41 | 31/33 | 8/8 | 0 | [4](eval/reports/4-pool-by-feature.json) |
| Hybrid search (`--hybrid`): 2 of 8 places for features only the embeddings find | **41/41** | **33/33** | **8/8** | **0** | [5](eval/reports/5-hybrid.json) |

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
- The last row was measured at commit `12ac20d`, with the judge too: it agreed on all 33
  values and, as before, rejected `kom-30A` and `abk-GB028` (see "One record per question").

### Hybrid search

BM25 matches words; embeddings compare meanings. The vector channel found WALS 30A first
for "number of genders noun classes", where BM25 put a dozen Grambank noun class features
ahead of it. It is weaker on its own: for "number of vowel phonemes" it ranked Grambank's
number markers and WALS consonant features above WALS 2A, which BM25 puts first. So the
question was how to fuse the two rankings. Retrieval only, the same rewritten queries:

| Fusion | Tuning set (41) | Held-out set (32) |
|---|---|---|
| BM25 alone | 40/41 | 32/32 |
| Reciprocal rank fusion (RRF, k = 60) | 38/41 | — |
| Reserved slots: BM25's first 6, then the best 2 vector-only features | **41/41** | **32/32** |

- RRF rewards agreement: a feature both channels rank 3rd to 5th beats the first feature of
  either one. It lost WALS 2A twice and did not win 30A.
- Reserved slots cannot push BM25's own choices out and still let in a feature only the
  vectors find. They won `en-swa-30A` and lost nothing; with answers, `en-swa-30A` became a
  correct "five or more genders" citing WALS 30A.
- The gain is one question of 73, and the slots were chosen on the tuning set; the held-out
  set was used a second time here, for retrieval only. BM25 stays the default, since hybrid
  search needs `glotto-lens embed` and an API call per question.
- `rus-117A` also passed in this run. Its evidence changed only in the last two places, and
  it had passed and failed before with the same features, so it is not counted as a gain
  of the hybrid search.

### Results: held-out set

Alice AI, the settings of the last retrieval row, judge DeepSeek V4.1 Flash; commit
`c6f6c82`, one run. Nothing was changed after it.

| Set | Retrieved | Correct | Honest | Invalid citations | Judge: value | Judge: no data | Report |
|---|---|---|---|---|---|---|---|
| Tuning set (41 questions, last retrieval row) | 40/41 | 31/33 | 8/8 | 0 | — | — | [4](eval/reports/4-pool-by-feature.json) |
| Held-out set (32 questions) | **32/32** | **23/23** | **6/9** | **0** | 23/23 | 7/9 | [heldout](eval/reports/heldout-aliceai-llm.json) |

What the numbers say:

- **Retrieval and value answers carry over.** Every feature was found, including WALS 30A
  for Zulu noun classes, the kind of question that fails for Swahili in the tuning set:
  "Сколько именных классов (родов)" is rewritten as "number of genders". Every value answer
  cites the right line.
- **Honesty does not carry over as well**, and the three misses are of different kinds:
  - `h-ita-13A` takes the missing value as evidence: "Italian has no tones, which is
    confirmed by the absence of a value in WALS". Both checks fail it; it is the one clear
    error.
  - `h-com-GB020` answers from WALS 37A, where Comanche has a value ("demonstrative word
    used as definite article"). The question does not name a database, so the answer is
    supported; the item is weaker than it looked when it was written.
  - `h-bak-81A` says "Данные о базовом порядке слов (SOV, SVO и т. д.) в WALS не
    зафиксированы". The judge accepts it; the phrase check misses it because the period in
    "т. д." ends its sentence window.
- The 8/8 honesty of the tuning set was partly tuned: the phrase check grew on its answers.
  On new questions a reasonable reading is 7–8 of 9, and the check alone gives 6.
- Neither miss was fixed here: changing the metric or the item after the run would make
  this row a tuning row.

### Results: answer models

Five models from the Yandex AI Studio catalogue answer the same 41 questions on the same
evidence: Alice AI rewrites every question (`--rewrite-model aliceai-llm`), so only the
answer differs. Temperature 0, reasoning off for DeepSeek and Qwen
(`--reasoning-effort none`), judge DeepSeek V4.1 Flash. Commit `e5a211b`.

| Answer model | Correct | Honest | Answers with invalid citations | Judge: value | Judge: no data | Output tokens | Report |
|---|---|---|---|---|---|---|---|
| Alice AI LLM (`aliceai-llm`) | 31/33 | 8/8 | 0 | 31/33 | 7/8 | ~2.2k¹ | [alice](eval/reports/cmp-aliceai-llm.json) |
| Alice AI LLM Flash (`aliceai-llm-flash`) | 31/33 | 8/8 | 0 | 32/33 | 8/8 | 2.2k | [flash](eval/reports/cmp-aliceai-llm-flash.json) |
| YandexGPT 5.1 Pro (`yandexgpt-5.1`) | **32/33** | 7/8 | 0 | 32/33 | 7/8 | 2.3k | [yandexgpt](eval/reports/cmp-yandexgpt-5.1.json) |
| DeepSeek V4.1 Flash (`deepseek-v4.1-flash`) | **32/33** | **8/8** | 0 | 32/33 | 8/8 | 7.9k | [deepseek](eval/reports/cmp-deepseek-v4.1-flash.json) |
| Qwen3.6-35B (`qwen3.6-35b-a3b`) | 31/33 | 8/8 | 1 | 32/33 | 7/7² | 6.4k | [qwen](eval/reports/cmp-qwen3.6-35b-a3b.json) |
| DeepSeek V4.1 Flash, reasoning on³ | **32/33** | **8/8** | 0 | 32/33 | 8/8 | 19.7k | [reasoning](eval/reports/cmp-deepseek-v4.1-flash-reasoning.json) |

¹ Alice's answers came from the cache, so the report has no token count; the figure is from
the last paid run and includes the rewrite step, which the other rows took from the cache. ² The judge's reply on `kom-30A` was cut off at 2000 tokens.
³ The model's default reasoning, commit `892fe3a`; two answers (`eng-49A`, `yor-13A`) were
cut off at the 1000-token limit after they had cited the value.

What the numbers say:

- **The models are within one answer of each other.** One answer is 3 points on 33
  questions, and every model was run once, so the table does not rank them. 32/33 is the
  ceiling: `en-swa-30A` fails in retrieval for everyone.
- **The misses are about citing, not knowing.** Alice and Alice Flash cite the Grambank
  lines instead of WALS 117A on `rus-117A`. Qwen writes "(V1)" without brackets on
  `bsq-37A` and cites a label that is not in the evidence on `geo-1A`. YandexGPT on `kom-30A`
  says Komi has no gender, where WALS has no value; it is the only dishonest answer.
- **DeepSeek and Qwen write three times as much** for the same scores, even with reasoning
  off.
- **Reasoning does not help here.** With its default reasoning DeepSeek gets exactly the same
  verdict on every one of the 41 questions, from the checks and from the judge, for 2.5 times
  the output tokens. These questions ask for a value that is already in the evidence, so the
  work is finding and citing the right line, not inferring anything; questions that need
  several facts combined could behave differently.
- **The judge found three gaps in the honesty check**: "WALS does not provide a value",
  "значение не установлено", and a negation 16 words after "значения". The check now
  accepts them; no other answer in any report changed its score.
- **The judge is stricter on hedged answers.** On `kom-30A` Alice says WALS has no value but
  concludes from Grambank that there is no gender; the phrase check passes it, the judge
  does not. On `fra-87A` it rejects "after the noun, and in both orders".

## Limitations

- **Tuned on the test set, checked once on a held-out set.** Retrieval settings, the rewrite
  examples and the honesty check were chosen while looking at the 41 tuning questions. The
  held-out set (32 questions, other languages) has been run once; it is small, and a second
  look at it would no longer be held out.
- **Small set.** On 33 value questions one answer is 3 percentage points, and answers vary
  between runs when the evidence changes. In the fourth row `rus-117A` (possession in
  Russian) failed because the answer cited the Grambank lines instead of the WALS 117A
  line, though the content was right; a run earlier it passed.
- **Lexical search by default.** BM25 matches words, so extra words in the rewritten query
  can push the right feature out (`en-swa-30A`). Hybrid search fixes that case, but it is
  opt-in, and its embedding model is served only as `text-search-doc/latest`: if Yandex
  updates it, stored vectors and the cache do not notice. After such an update, delete
  `data/cache/embeddings` and the `document_vectors` table and run `glotto-lens embed`.
- **Honesty is a phrase check.** It recognises Russian and English wordings of "no data"
  with a regular expression. It also accepts some claims about the language
  ("род неизвестен"), and it took several corrections to cover the phrasings the models
  actually use; the LLM judge now flags the phrasings it misses. Abbreviations with a
  period ("т. д.") end its sentence window (`h-bak-81A`).
- **The judge is a model too.** It sees no evidence, so it checks the recorded value, not
  whether the answer is supported. DeepSeek judges its own answers in its row, and on long
  answers its reply can be cut off (`kom-30A`). It reasons by default, so it uses several
  times more output tokens than the answer models.
- **One record per question.** Each item is scored against one feature, but a question that
  names no database can be answered from the other one: for `abk-GB028` and `h-com-GB020`
  Grambank has no value and WALS has one (39A, 37A). The judge sees only the item's record
  and counts such answers as claims. A prompt rule against answering from other features
  was tried on the tuning set: `kom-30A` kept its conclusion, `abk-GB028` lost its hedge,
  and the rule was dropped.
- **One language per question.** Comparisons ("Russian vs Polish") and questions about
  languages in general are not supported yet.
- **Coverage.** The databases have gaps (Grambank has no data for Standard German, for
  example), and 17 WALS languages have no glottocode, so their values cannot be looked up
  by name through Glottolog.
- **One run per model.** The comparison has no repeats, and reasoning was tried on one model
  only (DeepSeek).

## Project layout

```text
src/glotto_lens/
  sources.py     pinned dataset releases, download, reading CLDF tables
  database.py    build the SQLite database (atomic), sanity report
  chapters.py    parse WALS chapter HTML into blocks
  documents.py   searchable documents: chapter chunks, feature cards, distributions
  search.py      FTS5 search with BM25, per-feature mode
  vectors.py     embeddings in SQLite, vector search, hybrid fusion (reserved slots or RRF)
  languages.py   name normalisation, language lookup, language profiles
  context.py     evidence for a question: features, values, passages, labels
  prompts.py     rewrite and answer prompts
  llm.py         Yandex AI Studio client, disk cache, token log
  answer.py      the pipeline: rewrite, retrieve, answer, citation check, answer log
  evaluate.py    question set checks, metrics, reports
  cli.py         the glotto-lens command
eval/questions.jsonl   the question set
eval/reports/          the evaluation reports behind the results table
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
