# CAI coverage (Steps 0–4) — DeepSeek

Generated: 2026-10-28T10:37:53Z

DeepSeek-only constitution / SL-CAI / RLAIF data build over the I/E-R intrinsic top 40.
No Colab, GPU, Unsloth, or Drive. Ranked by I/E-R intrinsic only (no I−E filter).
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
- DeepSeek-only Steps 0–4 complete. Step 5 (Colab/GPU/Unsloth/Drive) was not run.

## Outputs

- `/cursor/stores/self/cai-pipeline/data/corpus_manifest.json`
- `/cursor/stores/self/cai-pipeline/data/raw_principles.jsonl`
- `/cursor/stores/self/cai-pipeline/data/constitution.json`
- `/cursor/stores/self/cai-pipeline/data/prompts_dataset.jsonl`
- `/cursor/stores/self/cai-pipeline/data/sl_cai_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/rlaif_dpo_train.jsonl`
- `/cursor/stores/self/cai-pipeline/data/state.db`
