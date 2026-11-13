# Item-projected sentiment — method note (slice 1)

Closed-form scoring only. No LLM, classifier, or learned rater was used for eligibility, work choice, chunk skip, or scores.

## Formula version

- `formula_version`: `slice1-v1`
- Window radius k = **10** tokens
- Chunk size = **500** tokens
- US public-domain cutoff year = **1930** (95-year rule as of 2026)
- Porter stemming of item seeds: **yes** (NLTK `PorterStemmer`)
- Stopwords: NLTK English list (198 types), frozen in the audit DB via seed generation; `god`/`faith`/`religion`/`divine`/`prayer`/`church` forced kept

## Items (published wording, not rewritten)

**Hoge (1972) Intrinsic Religious Motivation Scale** — 10 items. Reverse-coded items 8–10 (published extrinsic poles). Fetzer/NIA booklet Commitment chapter (Idler et al. 2003; Williams 1994) uses this instrument. Wording frozen as specified for this slice.

**Gorsuch & McPherson (1989) I/E-R** — full 14 items, not the two-item Fetzer Values short form.

- Intrinsic (8): items 1, 3, 4, 5, 7, 10, 12, 14; reverse 3, 10, 14
- Extrinsic-personal (3): items 6, 8, 9
- Extrinsic-social (3): items 2, 11, 13

Attendance/tithe/hours were not added as extra behavioral items. I/E-R Es items about *why* one goes to church are scored as published motivation items.

Seeds = item alphabetic tokens after stopword removal, then Porter stem. Apostrophe-s stripped before stemming. Seeds stored in `item_bank.seeds_json`.

## Sentiment (window valence)

For chunk c and item i with seed set S_i and reverse flag r_i in {+1, -1}:

1. Hit if any stemmed seed occurs in the chunk.
2. Window valence = MultiLexScaled over tokens within +/- k of each hit (mean of per-hit windows).
3. Chunk-item score: r_i * (seed density) * (window valence), with seed density = seed-token count / chunk token count.
4. Work-item score: mean over all chunks (no-hit and iSRM-skipped chunks contribute 0; skipped rows are stored).
5. Subscales: mean of published item groups.
6. `intrinsic` = mean(Hoge IRMS, I/E-R intrinsic).
7. `extrinsic` = mean of the six I/E-R Ep+Es items.
8. `I_minus_E` = intrinsic − extrinsic.

Religion-language floor: skip chunk iff iSRM token count = 0. Count and skip flag persisted.

### MultiLexScaled / component lexica

van der Veen & Bleich (2024), *PLoS ONE* 20(1): e0313092. USUK scaler file `Calibration_USUK_2021-12-10.csv`.

| Lexicon | Version / file | Role |
|---|---|---|
| Hu & Liu opinion lexicon | `positive-words.txt` / `negative-words.txt` | ±1 |
| labMT (Dodds et al.) | `labMT_English.tsv`, centered −5, |v|≥1 filter | LabMT_filtered |
| Lexicoder SD 2015 | quanteda `LSD2015.lc3` (Young & Soroka 2012) | ±1, prefix wildcards |
| MPQA subjectivity clues | Wilson, Wiebe & Hoffmann 2005 `subjclues.tff` | strong/weak × pos/neg |
| NRC Emotion Lexicon v0.92 | positive/negative; both-pos-and-neg dropped | ±1 |
| SentiWordNet 3.0 | mean synset pos−neg, |v|≥0.1 | SWN_filtered |
| SO-CAL (Taboada et al. 2011) | adj/adv/noun/verb 1.11 + intensifiers | noun plural / verb inflection heuristics (pattern.en not used) |
| WordStat | **not used** (proprietary; not obtained) | — |

iSRM: OSF `iSRM_dictionary_v1.0` (OSF fxhcm / drfzt), 1272 unique terms as released in the file (abstract states 888 validated terms; the released list is what was counted).

Window valence = mean of USUK-calibrated component raw scores (sum of matched valences / window tokens), then divided by published `stdev_of_avgs` = 0.8495984756427533. SO-CAL modifiers applied as 1 + w/2; MLS default negaters applied.

Lexicon sizes (frozen): {"HuLiu": {"n": 6786, "pos": 2003, "neg": 4783}, "LabMT_filtered": {"n": 3731, "pos": 2668, "neg": 1063}, "LexicoderSD": {"n": 4409, "pos": 1625, "neg": 2784}, "MPQA": {"n": 6447, "pos": 2298, "neg": 4149}, "NRC": {"n": 5474, "pos": 2231, "neg": 3243}, "SWN_filtered": {"n": 24127, "pos": 11081, "neg": 13046}, "SOCAL": {"n": 11020, "pos": 4045, "neg": 6970, "n_mods": 210}}

## Work eligibility and choice

Census: `christian-writers.sqlite` (not overwritten).

1. **PD full text** in Gutenberg English catalog, English Wikisource (non-encyclopedia), or Internet Archive identifier when present. Encyclopedia/DNB/Britannica Wikisource pages are not works.
2. **English** original (`en` / English / American English / …) **or** the retrieved copy is an English Gutenberg/Wikisource text (PD translation of a non-English original).
3. **Year ≤ 1930.** Census year if parseable; else author `death_date` ≤ 1930 used as latest bound when a PD catalog copy exists (`year_source=author_death_latest`). Missing year without that bound → `missing_year`.
4. **Citation:** OpenAlex `cited_by_count` by Wikidata work id only (no title search); else Wikipedia `linkshere` count (namespace 0, cap 500); else Wikidata sitelink count. Authors with more than 25 eligible copies: API cites are capped at 25 (preferring works that already have a Wikidata/Wikipedia id); remaining eligible copies are stored with `citation_score` 0. Ties → earlier year, then lower census `work_id`.
5. Empty eligible set → exclude author. Reason codes stored on `author_decisions`.
6. No invented works. Census rows only. Missing catalog data → ineligible, logged on `work_candidates`.

## Where these scores go in the current train

The current fine-tune is v3 ([`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb)). It keeps the 20 works with the highest `ier_intrinsic` score from this method (no I−E filter). v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 rewrites the v2 traces into one weighed turn, drops UltraChat, and keeps the 300-row format replay. Three arms: weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, and a size-matched control. R-DPO α=0.01, β=0.1. Bidirectional rejected answers. Seeds 0–2. Reported score: WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. The v1, v1_fixed, and v2 notebooks stay untouched. This page is the scoring method. The mix is in [`colab/data/v3/`](../colab/data/v3/).

## Audit database

`intrinsic-motivation-scores.sqlite` tables: `runs`, `item_bank`, `author_decisions`, `work_candidates`, `works_scored`, `chunks`, `chunk_item_scores`, `work_item_scores`, `work_subscores`, `api_errors`. JSONL is a view over `works_scored` + `work_item_scores`.

## References

- Hoge, D. R. (1972). A validated intrinsic religious motivation scale. *JSSR, 11*(4), 369–376.
- Gorsuch, R. L., & McPherson, S. E. (1989). Intrinsic/extrinsic measurement: I/E-revised and single-item scales. *JSSR, 28*(3), 348–354.
- Williams, D. R. (1994). The measurement of religion in epidemiologic studies. In *Religion in aging and health*.
- Idler, E. L., et al. (2003). Measuring multiple dimensions of religion and spirituality for health research. *Research on Aging*. Fetzer/NIA booklet.
- van der Veen, A. M., & Bleich, E. (2024). The advantages of lexicon-based sentiment analysis in an age of machine learning. *PLoS ONE*.
- iSRM Dictionary v1.0, OSF 10.17605/osf.io/fxhcm.
