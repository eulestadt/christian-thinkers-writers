# Intrinsic motivation scoring — coverage

Generated: 2026-10-27T22:11:05Z
Formula: `slice1-v1`; k=10; chunk=500 tokens; PD cutoff=1930.

The counts below are the scoring census (527 works). The current training run is **v3**, which keeps the top 20 of those works by `ier_intrinsic`. v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 rewrites the v2 traces into one weighed turn, drops UltraChat, and keeps the 300-row format replay. Three arms: weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, and a size-matched control. R-DPO α=0.01, β=0.1. Bidirectional rejected answers. Seeds 0–2. Reported score: WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. Notebook: [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb). The v1, v1_fixed, and v2 notebooks stay untouched. Coverage: [`cai-3way-colab.md`](cai-3way-colab.md).

## Authors

| Metric | Value |
|---|---|
| Census writers | 14128 |
| Authors with a keep/exclude row | 14128 |
| Kept (one English PD work scored) | 527 |
| Excluded | 13601 |
| Census works considered (`work_candidates`) | 163063 |
| Eligible work-copies | 5412 |
| Works scored | 527 |
| Chunks | 99637 |
| Chunks skipped (iSRM count = 0) | 86 |
| API errors logged | 4350 |

Every census author has exactly one `author_decisions` row.

## Exclusions by reason

| reason_code | authors |
|---|---|
| `no_pd_text` | 8566 |
| `no_works_listed` | 4804 |
| `text_fetch_failed` | 217 |
| `no_english_pd_text` | 9 |
| `missing_year` | 3 |
| `year_after_1930` | 2 |

## API errors by service

| service | n |
|---|---|
| openalex | 2925 |
| wikisource | 892 |
| wikidata | 428 |
| wikipedia | 94 |
| gutenberg | 8 |
| pipeline | 3 |

## Systematic gaps

- WordStat (8th MultiLexScaled lexicon) was not available; valence uses the other seven published component lexica with the USUK scaler.
- OpinionFinder 2.0 `mpqapolarity` file was not obtained; MPQA uses the published Wilson et al. 2005 subjectivity clues with the same strong/weak × polarity mapping.
- SO-CAL verb/noun expansion uses closed-form English inflection heuristics, not `pattern.en`.
- Gutenberg matching is deterministic title/author string overlap against `pg_catalog.csv` (English texts only). False negatives occur when census titles are non-English, fragmentary, or do not overlap the Gutenberg title.
- Wikisource encyclopedia/DNB/Britannica pages in the census are not treated as authorial works.
- Internet Archive `pdusa` is used only when a catalog identifier is already attached; there is no free-text IA search, so IA is a smaller source than Gutenberg.
- Wikipedia inlink counts cap at 500 (`lhlimit`).
- OpenAlex is queried by Wikidata work id only (title search was not used). Unmatched works fall back to Wikipedia inlinks then Wikidata sitelinks, else citation_score 0. Authors with >25 eligible copies are API-cited for at most 25 copies.
- Authors with `works_status=none_listed` are excluded as `no_works_listed` (census listed no works; none were invented).
- Year-missing works are eligible only when a PD catalog copy exists **and** author death year ≤ 1930 (latest publication bound). Ancient writers without death dates and without census/Wikidata years are dropped as `missing_year` even if a Gutenberg name match exists.
- Non-English PD originals without a retrieved English Gutenberg/Wikisource copy are `no_english_pd_text` or `no_pd_text`.
- iSRM released file has 1272 terms; the paper abstract states 888. The file on disk is what was counted.

## Outputs

- Method: `/cursor/stores/self/docs/intrinsic-motivation-scoring.md`
- JSONL (kept authors): `/cursor/stores/self/docs/intrinsic-motivation-scores.jsonl`
- Audit SQLite: `/cursor/stores/self/docs/intrinsic-motivation-scores.sqlite`
- Compressed copy of that SQLite: `/cursor/stores/self/docs/intrinsic-motivation-scores.sqlite.gz`
- This coverage file.

`chunk_item_scores` is queryable as `(chunk_id, item_id, hit, seed_density, window_valence, score)` (view over integer `item_nid` storage plus `item_nid_map`). `work_candidates`, `works_scored`, `work_subscores`, `chunks`, and `author_decisions` are physical tables. Every census author has one `author_decisions` row.
