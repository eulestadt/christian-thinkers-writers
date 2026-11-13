# CAI coverage (Steps 0–4) — legacy Gemini page

Generated: 2026-10-28T10:37:53Z

This page is the discarded Gemini data build. It was regenerated with DeepSeek and is not the current training run.

The current Colab run is **v3** ([`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb)). v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 keeps the top 20 books by `ier_intrinsic`, rewrites the v2 traces into one weighed turn, and does not use UltraChat. The 300-row format replay stays. Three arms: weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, and a size-matched control. R-DPO α=0.01, β=0.1. Bidirectional rejected answers. Seeds 0–2. Reported score: WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. The v1, v1_fixed, and v2 notebooks stay untouched. Coverage: [`cai-3way-colab.md`](cai-3way-colab.md).

The counts below are the old Steps 0–4 record, not the v3 mix. Ranked by I/E-R intrinsic only (no I−E filter).
Extraction, critique, revision, and judge prompts use author-text vocabulary only.
Gemini-generated principles, prompts, SL-CAI, and RLAIF labels were discarded and regenerated.

## Models

- Single model (Steps 1–4): `deepseek-flash`
- Thinking: disabled (`thinking: {type: disabled}`; no reasoning_effort)
- Client: OpenAI-compatible `https://api.deepseek.com`
- Rate limit: 40 RPM, TPM cap 2500000, max in-flight 6

## Counts

| Metric | Value |
|---|---|
| Books fetched | 40 |
| Books extracted (`deepseek-flash`) | 40 |
| Raw principles (pre-cluster) | 3139 |
| Elicited prompts (raw sum) | 1992 |
| Constitution principles | 2756 |
| Prompt set total | 2500 |
| Prompt set elicited | 2000 |
| Prompt set UltraChat | 500 |
| SL-CAI rows ok | 2500 |
| SL-CAI rows error | 0 |
| RLAIF rows ok | 2500 |
| RLAIF rows error | 0 |
| API calls ok | 3133 |
| API calls non-ok | 33 |
| API errors logged | 27 |

## API errors by type

| Type | n |
|---|---|
| `JSONDecodeError` | 13 |
| `APIConnectionError` | 7 |
| `APITimeoutError` | 7 |

## Successful calls by model

| Model | n |
|---|---|
| `deepseek-flash` | 3133 |

## Notes

- Step 4 wrote 2500 RLAIF DPO pairs to /cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl.
- DeepSeek-only Steps 0–4 on this discarded page are a historical record. The current Colab train is v3. See [`cai-3way-colab.md`](cai-3way-colab.md).

## Outputs

- `/cursor/stores/self/cai-pipeline/data/corpus_manifest.json`
- `/cursor/stores/self/cai-pipeline/data/raw_principles.jsonl`
- `/cursor/stores/self/cai-pipeline/data/constitution.json`
- `/cursor/stores/self/cai-pipeline/data/prompts_dataset.jsonl`
- `/cursor/stores/self/cai-pipeline/data/sl_cai_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/state.db`
