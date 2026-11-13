# Item-projected sentiment (slice 1: published rubrics + math only)

**Scope:** already-published intrinsic-motivation items (Williams and the instruments he uses) + already-published sentiment lexica → functions that score one English public-domain work per author. No new scale, no training, no LLM.

Deferred on this scoring slice: GPT item rater, human dual-coding, Quest/fundamentalism filters, SDT.

The fine-tuning inclusion policy is no longer deferred. The current run is v3 ([`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb)): top 20 works by `ier_intrinsic`, v2 traces rewritten into one weighed turn, no UltraChat, 300-row format replay kept. v1 trains two arms with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 trains three arms (weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, size-matched control) with R-DPO α=0.01 β=0.1, bidirectional rejected answers, and seeds 0–2. The reported score is WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. The v1, v1_fixed, and v2 notebooks stay untouched.

## Language: English original or PD English translation

Limit to English. The published sentiment lexica (MultiLexScaled and its component dictionaries, iSRM) are English word lists.

Eligible text for an author:

- Original English and US public domain and retrievable full text, or
- A public-domain English translation of a non-English original (same PD + full-text tests).

If several PD English texts exist, pick the most cited among that set. If none exist, exclude the author.

Do not score non-English text. Do not use a modern (in-copyright) translation even if the original is PD.

## No LLM anywhere in this slice

- Work eligibility: year + catalog flags + language code
- Work choice: argmax citation
- Author drop: empty eligible set
- Chunk skip: religiosity token count = 0
- Item/subscale scores: seed counts × lexicon valence × reverse-code

## Rubric (published items only)

Williams 1994; Fetzer/NIA booklet (Idler et al. 2003). Williams wrote Commitment and pointed to:

- Hoge 1972 Intrinsic Religious Motivation Scale (10 items)
- Gorsuch & McPherson 1989 I/E-R (Fetzer Values short form)
- Allport & Ross 1967 (parent I/E)

Freeze items with original Likert keys and reverse-codes. Seeds = item tokens after stopword removal (optionally Porter-stemmed). No rewritten items. Do not score attendance/tithe/hours on text.

## Sentiment functions (closed form)

For chunk c and item i with seed set S_i and reverse flag r_i in {+1,-1}:

- Hit if any seed in S_i occurs in c
- Window valence = MultiLexScaled over tokens within ±k of a hit
- Chunk-item score: r_i × (seed density) × (window valence)
- Work-item score: mean over chunks (no-hit chunks contribute 0)
- Subscales: mean of published item groups (Hoge intrinsic, I/E-R extrinsic personal, I/E-R extrinsic social)
- I − E = intrinsic mean − extrinsic mean

Religion-language floor: skip chunk iff iSRM (or equivalent published list) count = 0. Persist both the skip and the count.

## One work per author

From `/cursor/stores/self/docs/christian-writers.sqlite`:

1. PD: publication year ≤ 1930 (US 95-year rule as of 2026) and full text in Gutenberg, Wikisource, or Internet Archive pdusa. For translations, the translation’s year/catalog flags matter.
2. English: original language en or the copy scored is an English translation (lang=en).
3. Citation: OpenAlex cited_by_count if present; else Wikipedia inlinks to the work; else Wikidata sitelinks. Ties → earlier publication year.
4. Select argmax citation on the eligible set. Empty set → exclude author.

## Audit database

Tables: runs, item_bank, author_decisions, work_candidates, works_scored, chunks, chunk_item_scores, work_item_scores, work_subscores.

No score or exclusion exists only in a report. JSON exports are views over these tables.
